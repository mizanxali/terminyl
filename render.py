"""Draw the record in two pieces, matching how kitty can actually paint it.

A real LP's grooves, sheen and rim are rotationally symmetric -- spinning them
changes nothing on screen. What does turn is the paper label and the wipe marks
on the vinyl, which sweep through the highlight as the record goes round. So:

  * the grooves, sheen and rim are drawn once and live in kitty's background
    image layer
  * the label and the wipe marks are drawn once per rotation step, composited
    into one disc-sized frame and streamed to the window logo layer, which sits
    above the background image and below the text

That split is not just cheaper, it is also what a turntable actually looks like:
the lamp's reflection stays put on the disc while the record turns underneath
it. Streaming the whole disc every frame would move the highlight too, which
would read as a spinning lamp rather than a spinning record.

kitty has exactly one background image layer, so when the desk is switched on
the wood and the record are composited into a single image and uploaded
together -- still once, not per frame.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter

# Geometry as a fraction of the disc radius, roughly a 12" LP.
LABEL_R = 0.325
HOLE_R = 0.024
GROOVE_OUTER = 0.972
GROOVE_INNER = 0.365

# Radii of the shiny "band gaps" that separate tracks on a side.
BAND_GAPS = (0.44, 0.545, 0.63, 0.72, 0.80, 0.885, 0.945)

# Groove pitch in device pixels. Three is about the floor: any tighter and the
# rings alias into moire the moment kitty scales the image.
GROOVE_PITCH = 3.0

NEUTRAL_SHEEN = np.array([0.85, 0.86, 0.95], dtype=np.float32)

# Unlit vinyl is not neutral -- it is a very dark blue-black, and it is the
# blue that keeps it from reading as flat grey once the highlights come off.
VINYL_RGB = np.array([0.90, 0.94, 1.08], dtype=np.float32)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _png(img: Image.Image, compress_level: int = 1) -> bytes:
    """Level 1 for anything on the frame path -- the socket is never the
    bottleneck, and the encoder would be. Uploads that happen once can afford
    to be squeezed properly."""
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=compress_level)
    return buf.getvalue()


# Blur, in device pixels, applied to the background image layer -- the wood and
# the record -- and to nothing else. It is a legibility control, not a look: the
# grooves are a GROOVE_PITCH ripple, which is the highest-frequency thing on
# screen and lands directly under the text. Softening it costs the disc almost
# nothing at a glance and buys the terminal a quiet field to sit on.
#
# The cover art is not blurred at any radius. It lives in the window logo layer,
# which is composited after the background, so it stays as sharp as it arrived.


def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _blur(a: np.ndarray, mode: str, radius: float) -> np.ndarray:
    """Gaussian blur of a float array, by way of PIL.

    Pillow's blur only takes 8-bit bands, so this quantises on the way in --
    which costs nothing real, since everything here is bound for a PNG at the
    same depth a few lines later.
    """
    return np.asarray(Image.fromarray(_u8(a), mode).filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float32) / 255.0


def _blur_rgb(rgb: np.ndarray, radius: float) -> np.ndarray:
    return rgb if radius <= 0.0 else _blur(rgb, "RGB", radius)


def _blur_rgba(arr: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur that respects transparency.

    Blurring the colour channels on their own drags whatever is sitting in the
    fully transparent pixels into everything near an edge. Off the rim of a bare
    disc that is a dark halo, and around the punched label hole it is a dark ring
    right where the artwork will land. Premultiplying by alpha first, blurring,
    then dividing back out is what keeps the edges clean.
    """
    if radius <= 0.0:
        return arr
    alpha = arr[..., 3:4]
    blurred = _blur(np.concatenate([arr[..., :3] * alpha, alpha], axis=-1), "RGBA", radius)
    out = blurred[..., 3:4]
    rgb = blurred[..., :3] / np.maximum(out, 1.0 / 255.0)
    return np.concatenate([np.clip(rgb, 0.0, 1.0), out], axis=-1)


def label_diameter(disc_size: int) -> int:
    """Label diameter in pixels for a disc of the given size. Always even."""
    d = int(round(disc_size * LABEL_R))
    return d - (d % 2)


def dominant_color(img: Image.Image) -> np.ndarray:
    """Average of the artwork's saturated pixels, normalised to full brightness."""
    a = np.asarray(img.convert("RGB").resize((48, 48), Image.BILINEAR), dtype=np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    weight = (mx - mn) ** 1.5 * (mx > 0.15)
    total = float(weight.sum())
    if total < 1e-3:
        return NEUTRAL_SHEEN.copy()
    col = (a * weight[..., None]).sum((0, 1)) / total
    peak = float(col.max())
    return (col / peak).astype(np.float32) if peak > 1e-6 else NEUTRAL_SHEEN.copy()


def load_artwork(data: bytes | None) -> Image.Image | None:
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------
# the disc (background image layer)


# Resolution of the polar noise texture the wipe marks are sampled from. The
# width also sets how finely the record can be turned: one column is 360/1024
# of a revolution, which is an order of magnitude finer than any frame count
# worth animating.
NOISE_H, NOISE_W = 256, 1024


def _polar_texture(cells_t: int, cells_r: int, seed: int) -> np.ndarray:
    """Smooth noise on a (theta, r) grid rather than an (x, y) one.

    Cells that are wide in theta and short in r come out as arcs following the
    grooves, which is how wipe marks actually land on a record -- Cartesian
    noise gives blotches, and blotches read as dirt on the lens instead.

    The grid's last column repeats its first, so theta wraps without leaving a
    seam down one radius of the record.
    """
    grid = np.random.default_rng(seed).random((max(2, cells_r), cells_t + 1), dtype=np.float32)
    grid[:, -1] = grid[:, 0]
    img = Image.fromarray((grid * 255.0).astype(np.uint8), "L").resize(
        (NOISE_W, NOISE_H), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _polar_index(r: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Texture coordinates for every pixel, computed once and reused per frame."""
    ti = np.clip((theta + math.pi) * (NOISE_W / (2.0 * math.pi)), 0, NOISE_W - 1).astype(np.int32)
    ri = np.clip(r * (NOISE_H - 1), 0, NOISE_H - 1).astype(np.int32)
    return ri, ti


def _sample_polar(tex: np.ndarray, ri: np.ndarray, ti: np.ndarray, turn: int = 0) -> np.ndarray:
    """Sample the texture, optionally turned `turn` columns about the centre.

    Turning by whole columns is what makes rotation nearly free: no resampling,
    no interpolation, just reading the same texture at an offset.
    """
    return tex[ri, (ti + turn) % NOISE_W]


def _disc_geometry(size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalised radius, angle and pixel size for a disc `size` pixels across."""
    radius = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = (size - 1) / 2.0
    dx, dy = (xx - centre) / radius, (yy - centre) / radius
    return np.hypot(dx, dy), np.arctan2(dy, dx), 1.0 / radius


def _grooved(r: np.ndarray) -> np.ndarray:
    """1 across the recorded band, falling to 0 on the dead wax and the margin."""
    return _smoothstep(GROOVE_INNER, GROOVE_INNER + 0.02, r) * (
        1.0 - _smoothstep(GROOVE_OUTER, GROOVE_OUTER + 0.012, r)
    )


def _disc_rgba(size: int, opacity: float = 0.60, sheen_deg: float = 35.0,
               sheen_rgb: np.ndarray | None = None, recess: bool = True) -> np.ndarray:
    """Float RGBA of the record, by default with a hole for the label.

    The lighting model is the one thing about a record that is worth getting
    right: the surface is a single concentric groove, so it reflects
    anisotropically. A point source does not put a round highlight on an LP, it
    puts two opposed lobes across it, at their brightest where the grooves run
    square to the light. Everything here follows from that -- the lobes, the
    grooves only really showing up inside them, the scuffs only glinting there.

    `recess=False` keeps the label area solid. Whatever is behind the disc then
    cannot show through the seam between the punched hole and the label that
    covers it -- which matters as soon as there is a desk back there, because
    that seam stops being invisibly dark and becomes a ring of bright wood.
    """
    sheen_rgb = NEUTRAL_SHEEN if sheen_rgb is None else sheen_rgb
    r, theta, px = _disc_geometry(size)
    radius = size / 2.0

    # Cosine of the angle off the light axis. Squaring loses the sign, which is
    # what makes one light give two lobes instead of one.
    axial = np.cos(theta - math.radians(sheen_deg))
    lit = np.abs(axial)

    grooved = _grooved(r)
    # The groove profile itself: one cycle per pitch, in device pixels so it
    # stays the same width however big the record is drawn.
    wave = 0.5 + 0.5 * np.cos(r * radius * (2.0 * math.pi / GROOVE_PITCH))

    # --- diffuse: nearly black, and what little there is falls off outwards.
    lum = np.full_like(r, 0.036)
    lum += 0.020 * wave * grooved
    for gap in BAND_GAPS:
        lum += 0.038 * np.exp(-(((r - gap) / (2.0 * px)) ** 2)) * grooved
    # Smooth run-out land just outside the label, where the groove stops.
    lum += 0.030 * np.exp(-(((r - (GROOVE_INNER - 0.018)) / 0.020) ** 2))
    lum *= 1.0 - 0.26 * _smoothstep(0.45, 1.0, r)
    # A gentle tilt towards the light, so the disc reads as an object under a
    # lamp rather than a flat cut-out with rings on it.
    lum *= 1.0 + 0.30 * axial * r

    rgb = lum[..., None] * VINYL_RGB[None, None, :]

    # --- specular: the two lobes, a broad haze plus a tight core. Both get
    # brighter towards the rim, where the light is glancing off the surface
    # rather than coming at it square.
    envelope = np.exp(-(((r - 0.74) / 0.46) ** 2)) * (1.0 + 0.55 * _smoothstep(0.4, 1.0, r))
    lobe = 0.085 * lit ** 2.0 + 0.150 * lit ** 5.0 + 0.300 * lit ** 13.0
    # Inside a lobe the grooves are what is doing the reflecting, so this is
    # where the ring structure gets its contrast -- and it vanishes in the dark
    # quadrants, exactly as it does on a real record.
    spec = lobe * envelope * grooved * (0.22 + 1.05 * wave ** 1.4)
    for gap in BAND_GAPS:
        # Band gaps are unmodulated vinyl: mirror-smooth, so they flare.
        spec += 0.26 * lit ** 6.0 * np.exp(-(((r - gap) / (1.7 * px)) ** 2)) * envelope

    # Ungrooved vinyl -- the dead wax inside, the label collar, the margin out
    # past the last track -- is smooth, so it catches the light in a tighter and
    # brighter arc than the grooved area ever does.
    land = np.exp(-(((r - (GROOVE_INNER - 0.016)) / 0.024) ** 2))
    land += 0.7 * np.exp(-(((r - (LABEL_R + 0.014)) / 0.016) ** 2))
    land += _smoothstep(GROOVE_OUTER, GROOVE_OUTER + 0.006, r) * (
        1.0 - _smoothstep(0.982, 0.989, r))
    spec += 0.22 * lit ** 8.0 * land

    # No wipe marks here: they are the one part of the surface that is not
    # rotationally symmetric, so they live in the frames that turn instead.
    # See `render_spin_frames`.

    # The rim is a rolled lip, so it runs a thin bright line right round the
    # record -- brightest facing the light, but never fully dark, because at
    # that angle it is picking up whatever is above it.
    lip = np.exp(-(((r - 0.9895) / 0.0045) ** 2))
    spec += (0.075 + 0.50 * lit ** 2.5) * lip

    # Vinyl reflects the room, not the sleeve. Taking the artwork's colour neat
    # turns a black record into coloured plastic, so it only tints the sheen.
    tint = 0.42 * sheen_rgb + 0.58 * NEUTRAL_SHEEN
    rgb += np.maximum(spec, 0.0)[..., None] * tint[None, None, :]

    # A dark line right at the outside, so the edge reads as a thickness rather
    # than the picture simply stopping.
    rgb *= (1.0 - 0.55 * _smoothstep(0.9955, 1.0, r))[..., None]

    edge = 1.0 - _smoothstep(1.0 - 1.6 * px, 1.0, r)
    # Punch out the label recess; the logo layer paints the label on top.
    hole = _smoothstep(LABEL_R - 1.5 * px, LABEL_R + 0.5 * px, r) if recess else 1.0
    # The shading ring stays either way -- it is what makes the label read as
    # sunk into the disc rather than stuck on top of it.
    rgb *= (1.0 - 0.55 * np.exp(-(((r - LABEL_R) / (2.2 * px)) ** 2)))[..., None]
    # ...and the label sits proud enough to throw a short shadow away from the
    # light, which is what actually sells the step in the surface.
    rgb *= (1.0 - 0.34 * np.exp(-(((r - LABEL_R) / 0.014) ** 2))
            * _smoothstep(0.0, 0.8, -axial))[..., None]

    alpha = edge * hole * opacity
    # Soft shoulder instead of a hard clip: the lobes run hot where they cross
    # a band gap, and clipping there turns the sheen chalk white.
    rgb = 1.0 - np.exp(-np.maximum(rgb, 0.0) * 1.22)
    return np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)


def render_disc(size: int, opacity: float = 0.60, sheen_deg: float = 35.0,
                sheen_rgb: np.ndarray | None = None, blur: float = 0.0) -> bytes:
    """PNG of the record with a hole in the middle for the label to sit in."""
    out = _blur_rgba(_disc_rgba(size, opacity, sheen_deg, sheen_rgb), blur)
    return _png(Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA"))


# --------------------------------------------------------------------------
# the desk (background image layer, underneath the disc)

# Varnished timber under warm light. The wood is one colour lit unevenly, not a
# gradient between two colours -- that is what keeps it reading as wood.
#
# Dark walnut rather than oak, and deliberately so: this is backdrop, and every
# stop of desk brightness is contrast stolen from the text sitting on top of it.
# Terminal foregrounds are light, so the desk has to stay well below them.
WOOD = np.array([0.255, 0.108, 0.030], dtype=np.float32)
SUNLIGHT = np.array([1.00, 0.84, 0.52], dtype=np.float32)

# Light through a blind. The bars repeat every BLIND_PERIOD along the axis
# normal to them -- measured in window widths, so a wider window gets more bars
# rather than wider ones. Each slat then gets its own nudge, half-width and
# brightness from these, cycled; a real blind never lands evenly on a desk.
BLIND_PERIOD = 0.34
BLIND_SLATS = ((0.000, 0.36, 1.00), (0.055, 0.27, 0.84), (-0.045, 0.39, 0.93),
               (0.030, 0.30, 0.76), (-0.020, 0.34, 1.00), (0.065, 0.25, 0.86))

# Gaussian bars taper too gently to read as shafts of light; raising the
# exponent flattens the top and steepens the penumbra.
BLIND_FALLOFF = 3.0

# The top is boards, not one sheet. Board width as a fraction of the window
# width -- like the blinds, a wider window gets more boards, not wider ones.
PLANK_PERIOD = 0.235

# Bars descend left-to-right, like light from a window up and to the left.
BLIND_DEG = -33.0

# The desk is drawn at the window's own resolution, so the disc's grooves come
# out as fine as they do on their own. Past this it is scaled down and left to
# kitty to enlarge -- wood survives that far better than a 3px groove pitch.
DESK_MAX_DIM = 3200


def _value_noise(shape: tuple[int, int], cx: int, cy: int, seed: int) -> np.ndarray:
    """Smooth noise: a small random grid, bicubically stretched to size."""
    h, w = shape
    grid = np.random.default_rng(seed).random((max(2, cy), max(2, cx)), dtype=np.float32)
    img = Image.fromarray((grid * 255.0).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _fbm(shape: tuple[int, int], cx: int, cy: int, octaves: int, seed: int) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += amp * _value_noise(shape, cx * 2 ** o, cy * 2 ** o, seed + 17 * o)
        total += amp
        amp *= 0.5
    return out / total


def _desk_lighting(width: int, height: int, light: float) -> tuple[np.ndarray, np.ndarray]:
    """The blind bars and the lamp pool, as fields over the window.

    Computed apart from the wood so anything standing on the desk can be lit
    by the same light -- a plinth lit differently from the desk under it reads
    as pasted on, not standing there.
    """
    shape = (height, width)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = xx / max(width - 1, 1), yy / max(height - 1, 1)

    # Parallel bars, measured along the axis normal to them so the angle is one
    # number, then laid down slat by slat across however far that axis reaches.
    light = min(max(light, 0.0), 1.0)
    bars = np.full(shape, 0.5, dtype=np.float32)
    if light > 0.0:
        angle = math.radians(BLIND_DEG)
        s = u * math.cos(angle) + v * math.sin(angle)
        pattern = np.zeros(shape, dtype=np.float32)
        first = int(math.floor(float(s.min()) / BLIND_PERIOD)) - 1
        last = int(math.ceil(float(s.max()) / BLIND_PERIOD)) + 1
        for k in range(first, last + 1):
            nudge, halfwidth, strength = BLIND_SLATS[k % len(BLIND_SLATS)]
            centre = (k + nudge) * BLIND_PERIOD
            # Brightest slat wins rather than summing: overlapping penumbras
            # should not add up into a wash where the shadows ought to be.
            np.maximum(pattern, strength * np.exp(
                -np.abs((s - centre) / (halfwidth * BLIND_PERIOD)) ** BLIND_FALLOFF),
                out=pattern)
        # Fading towards 0.5 rather than 0 means no bars leaves the desk evenly
        # lit, not uniformly in shadow.
        bars += light * (pattern - 0.5)

    # A pool of lamp light rather than an even field: warm where it lands, and
    # the corners fall away harder.
    radial = np.hypot(u - 0.40, (v - 0.46) * 1.10) / 0.78
    pool = 1.0 - _smoothstep(0.22, 1.25, radial)
    return bars, pool


def _wood(width: int, height: int, seed: int, bars: np.ndarray,
          pool: np.ndarray) -> np.ndarray:
    """Linear-light RGB of a lacquered plank under the shared desk lighting.

    Returned before the highlight roll-off, so whatever furniture is going to
    stand on the desk can be composited in the same light and rolled off
    together with it.
    """
    shape = (height, width)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = xx / max(width - 1, 1), yy / max(height - 1, 1)

    # Each board is its own piece of timber: the grain phase jumps at the seam
    # and the stain takes a little differently board to board. That
    # discontinuity is most of what makes it read as a desk rather than a
    # texture fill.
    rng = np.random.default_rng(seed + 101)
    phase_tab = rng.random(64, dtype=np.float32)
    tone_tab = rng.random(64, dtype=np.float32) - 0.5
    plank = np.floor(u / PLANK_PERIOD).astype(np.int32) % 64
    phase, tone = phase_tab[plank], tone_tab[plank]

    # Timber grain: near-vertical lines that wander a little, because the saw
    # never cuts exactly along the fibre. Kept quiet -- on a lacquered plank the
    # grain is a texture, not a pattern.
    wander = _fbm(shape, 3, 2, 3, seed) - 0.5
    rings = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * 15.0 + wander * 2.8 + 0.18 * v
                                                + phase * 9.7))
    # Cubing pulls the cosine into thin dark lines on a broad even field. A
    # second, finer, differently-wandering set breaks up the corduroy that one
    # frequency on its own gives.
    fine = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u * 34.0 - wander * 4.5 + 0.30 * v
                                               + phase * 23.3))
    lines = 1.0 - (0.62 * rings ** 3 + 0.38 * fine ** 3)
    fibre = _fbm(shape, 300, 4, 2, seed + 3) - 0.5
    figure = _fbm(shape, 5, 3, 3, seed + 7) - 0.5

    # One wood colour, modulated around unity. Everything below is light.
    lum = 1.0 - 0.105 * lines + 0.085 * figure + 0.045 * fibre
    lum *= 1.0 + 0.13 * tone

    # The seams between boards: a dark gap, and just beside it the rounded edge
    # of the next board catching the light. Dark lines cost the text nothing.
    sp = (u / PLANK_PERIOD) % 1.0
    edge_px = np.minimum(sp, 1.0 - sp) * PLANK_PERIOD * max(width - 1, 1)
    lum *= 1.0 - 0.34 * np.exp(-((edge_px / 1.7) ** 2))
    lum += 0.045 * np.exp(-(((edge_px - 4.5) / 2.6) ** 2))
    # Boards cup a little as they dry -- barely brighter down the middle, barely
    # darker at the edges -- which is what lets them read as timber rather than
    # stripes painted on a flat sheet.
    lum *= 1.0 - 0.025 * np.cos(2.0 * math.pi * sp)

    rgb = WOOD[None, None, :] * lum[..., None]

    # Shade is 45% of full light; the lit strips run hot and pick up the colour
    # of the light itself. With no bars this all lands on the flat 0.5.
    rgb *= (0.45 + 0.95 * bars)[..., None]
    rgb += (0.055 * bars)[..., None] * SUNLIGHT[None, None, :]
    # Varnish: a broad reflected sheen, strongest where the light lands.
    rgb += (0.038 * bars * rings)[..., None] * SUNLIGHT[None, None, :]

    # The centre of the pool stays where it was -- brighter would cost the
    # text -- the shape comes from pushing the edges down.
    rgb *= (0.60 + 0.40 * pool)[..., None]
    rgb += (0.030 * pool ** 2)[..., None] * SUNLIGHT[None, None, :]
    # Shadows lose the lamp before they lose the sky, so they cool off as they
    # darken -- red falls away fastest, blue not at all. Warm pool against cool
    # corners is most of what makes the light feel like a lamp.
    rgb *= 1.0 - (1.0 - pool)[..., None] * np.array([0.055, 0.030, 0.0],
                                                    dtype=np.float32)[None, None, :]
    return rgb


# --------------------------------------------------------------------------
# desk clutter (background image layer, standing on the wood)

# The things that live on a desk beside a turntable: the mug the listener is
# working through, the pack they keep meaning to give up, a small plant. All
# of it is backdrop under text just like the wood, so the albedos stay muted,
# and every item is lit by the same bars and lamp pool as the desk it stands
# on -- an object carrying its own light reads as a sticker, not as standing
# there.
#
# Positions are window fractions and sizes fractions of the window width,
# like the planks and the blinds: clutter belongs to the room, not the
# record. The mug and the pack keep to the left edge and the plant to the
# right, all clear of where the plinth reaches on any usably wide window; on
# a window narrow enough for the deck to sweep the whole desk the items land
# on the plinth and read as standing on it instead, which is where they
# would end up anyway.

MUG_POS = (0.135, 0.72)      # where the varnish ring used to be
MUG_R = 0.045                # rim radius
MUG_HANDLE_DEG = 155.0       # off to the left, the way a right hand parks it

PACK_POS = (0.150, 0.46)
PACK_HALF = (0.036, 0.056)   # half-extents of the flip-top face
PACK_DEG = 16.0              # tossed down, not squared to the desk edge

PLANT_POS = (0.905, 0.83)
PLANT_R = 0.042              # pot radius; the leaves reach past it

CERAMIC_RGB = np.array([0.620, 0.575, 0.500], dtype=np.float32)
COFFEE_RGB = np.array([0.100, 0.048, 0.020], dtype=np.float32)
PACK_WHITE = np.array([0.680, 0.650, 0.580], dtype=np.float32)
PACK_RED = np.array([0.480, 0.032, 0.026], dtype=np.float32)
TERRACOTTA = np.array([0.355, 0.135, 0.058], dtype=np.float32)
SOIL_RGB = np.array([0.058, 0.040, 0.024], dtype=np.float32)
LEAF_RGB = np.array([0.052, 0.148, 0.040], dtype=np.float32)


def _stand_light(rgb: np.ndarray, bars: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """The desk lighting applied to something standing on it.

    A touch gentler than the varnish response -- glaze, cellophane and leaf
    wax all scatter more than lacquered timber -- but the same fields, so a
    blind bar runs unbroken across the wood and whatever is sitting in it.
    """
    out = rgb * (0.52 + 0.86 * bars)[..., None]
    out = out + (0.014 * bars)[..., None] * SUNLIGHT[None, None, :]
    return out * (0.62 + 0.38 * pool)[..., None]


def _mug(xx: np.ndarray, yy: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The coffee mug, straight down into it: a glazed ring round a disc of
    coffee, with the handle plugging into the wall.

    Returns (rgb, alpha, shadow), the rgb in linear light before the desk
    lighting, like everything else that stands on the wood.
    """
    R = MUG_R * w
    dx = (xx - MUG_POS[0] * w) / R
    dy = (yy - MUG_POS[1] * h) / R
    r = np.hypot(dx, dy)
    e = 1.5 / R
    facing = np.clip(-(dx + dy) / np.maximum(r, 1e-6) * 0.707, 0.0, 1.0)

    # The handle goes down first so the body paints over its root and the
    # loop plugs into the wall instead of lying against it.
    ha = math.radians(MUG_HANDLE_DEG)
    hcx, hcy = 1.12 * math.cos(ha), 1.12 * math.sin(ha)
    rh = np.hypot(dx - hcx, dy - hcy)
    loop = np.abs(rh - 0.40)
    handle_a = 1.0 - _smoothstep(0.13 - e, 0.13 + e, loop)
    # A torus from above: bright along the crown of the tube, rolling dark to
    # either edge, the lamp side a step up on the far side.
    hfac = np.clip(-((dx - hcx) + (dy - hcy)) / np.maximum(rh, 1e-6) * 0.707, 0.0, 1.0)
    h_lum = (0.40 + 0.55 * np.exp(-((loop / 0.055) ** 2))) * (0.70 + 0.45 * hfac)
    handle_rgb = h_lum[..., None] * CERAMIC_RGB[None, None, :]

    body_a = 1.0 - _smoothstep(1.0 - e, 1.0 + e, r)

    # The wall seen end-on: a rounded lip catching the lamp, the outer face
    # rolling away dark, and inside the cup the far wall lit while the near
    # wall shades its own coffee.
    lum = np.full_like(r, 0.62)
    lum += 0.46 * np.exp(-(((r - 0.87) / 0.05) ** 2)) * (0.45 + 0.55 * facing)
    inner = _smoothstep(0.83, 0.77, r)
    lum *= 1.0 - 0.45 * inner * facing
    lum += 0.22 * inner * (1.0 - facing)
    lum *= 1.0 - 0.42 * _smoothstep(0.94, 1.005, r)
    wall_rgb = lum[..., None] * CERAMIC_RGB[None, None, :]

    # The coffee: near-black liquid that shows almost nothing of itself, just
    # one soft stretch of the lamp off its surface.
    glint = np.exp(-(((dx + 0.24) / 0.30) ** 2 + ((dy + 0.28) / 0.15) ** 2))
    coffee_rgb = COFFEE_RGB[None, None, :] * (0.80 + 0.40 * facing[..., None] * r[..., None])
    coffee_rgb = coffee_rgb + (0.16 * glint)[..., None] * SUNLIGHT[None, None, :]
    coffee = 1.0 - _smoothstep(0.76 - 2.0 * e, 0.76 + 2.0 * e, r)
    body_rgb = wall_rgb * (1.0 - coffee[..., None]) + coffee_rgb * coffee[..., None]

    rgb = np.zeros(dx.shape + (3,), dtype=np.float32)
    alpha = np.zeros_like(dx)
    for a, c in ((handle_a, handle_rgb), (body_a, body_rgb)):
        rgb = rgb * (1.0 - a[..., None]) + c * a[..., None]
        alpha = alpha + a * (1.0 - alpha)

    # Cast down-right like everything under this lamp; the handle throws too.
    dxs, dys = dx - 0.28, dy - 0.34
    shadow = 1.0 - _smoothstep(0.90, 1.38, np.hypot(dxs, dys))
    shadow = np.maximum(shadow, 0.85 * (1.0 - _smoothstep(
        0.10, 0.34, np.abs(np.hypot(dxs - hcx, dys - hcy) - 0.40))))
    return rgb, alpha, shadow


def _pack(xx: np.ndarray, yy: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pack of reds, flip-top up, under its cellophane.

    The one piece of branding worth drawing at this size is the roof: red
    from the top edge down to the chevron's point mid-face, white below it.
    The name would be a smear of grey at this resolution, so a smear of grey
    is exactly what stands in for it.
    """
    hx_, hy_ = PACK_HALF[0] * w, PACK_HALF[1] * w
    c, s = math.cos(math.radians(PACK_DEG)), math.sin(math.radians(PACK_DEG))
    ox, oy = xx - PACK_POS[0] * w, yy - PACK_POS[1] * h
    lx, ly = ox * c + oy * s, -ox * s + oy * c
    corner = 0.16 * hx_

    def sdf(x, y):
        qx = np.abs(x) - (hx_ - corner)
        qy = np.abs(y) - (hy_ - corner)
        return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
                + np.minimum(np.maximum(qx, qy), 0.0) - corner)

    d = sdf(lx, ly)
    alpha = 1.0 - _smoothstep(-1.2, 1.2, d)

    boundary = hy_ * (-0.06 - 0.42 * np.abs(lx) / hx_)
    red = 1.0 - _smoothstep(boundary - 1.2, boundary + 1.2, ly)
    rgb = (PACK_WHITE[None, None, :] * (1.0 - red[..., None])
           + PACK_RED[None, None, :] * red[..., None])

    text = np.exp(-(((ly - 0.30 * hy_) / (0.045 * hy_)) ** 2)) \
        * (1.0 - _smoothstep(0.50 * hx_, 0.62 * hx_, np.abs(lx)))
    rgb *= 1.0 - 0.22 * text[..., None]

    # Cellophane: one broad diagonal streak of the room, laid over red and
    # white alike -- it is the wrapper reflecting, not the print.
    gloss = np.exp(-(((lx / hx_ - 0.55 * ly / hy_) - 0.35) / 0.55) ** 2)
    rgb += (0.10 * gloss)[..., None] * NEUTRAL_SHEEN[None, None, :]

    # Bevelled edges, lit on the sides facing the lamp -- the box normal
    # rotated back into screen space, same trick as the plinth.
    nlx, nly = lx / hx_, ly / hy_
    nsx, nsy = nlx * c - nly * s, nlx * s + nly * c
    bevel = np.exp(-(((d + 2.5) / 2.2) ** 2))
    rgb += (0.14 * bevel * np.clip(-(nsx + nsy) * 0.707, 0.0, 1.0))[..., None] \
        * SUNLIGHT[None, None, :]
    rgb *= (1.0 - 0.35 * bevel * np.clip((nsx + nsy) * 0.707, 0.0, 1.0))[..., None]

    # A box this shallow throws a tight shadow; the small rotation makes the
    # down-right offset in local axes close enough to the true one.
    shadow = 1.0 - _smoothstep(-1.0, 0.30 * hx_, sdf(lx - 0.20 * hx_, ly - 0.24 * hx_))
    return rgb, alpha, shadow


def _plant(xx: np.ndarray, yy: np.ndarray, w: int, h: int,
           seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small potted succulent, straight down onto the rosette.

    Two whorls of leaves over a terracotta pot: the outer whorl goes down
    first so the inner one overlaps it, the way a rosette actually stacks,
    and the longest leaves overhang the rim onto the desk.
    """
    Rp = PLANT_R * w
    dx = (xx - PLANT_POS[0] * w) / Rp
    dy = (yy - PLANT_POS[1] * h) / Rp
    r = np.hypot(dx, dy)
    e = 1.5 / Rp
    facing = np.clip(-(dx + dy) / np.maximum(r, 1e-6) * 0.707, 0.0, 1.0)

    rgb = np.zeros(dx.shape + (3,), dtype=np.float32)
    alpha = np.zeros_like(dx)

    def put(a, c):
        nonlocal rgb, alpha
        rgb = rgb * (1.0 - a[..., None]) + c * a[..., None]
        alpha = alpha + a * (1.0 - alpha)

    # The pot: a lit terracotta rim round a disc of soil sitting in its own
    # shade -- the leaves shadow it from barely above.
    pot_a = 1.0 - _smoothstep(1.0 - e, 1.0 + e, r)
    lum = np.full_like(r, 0.55)
    lum += 0.40 * np.exp(-(((r - 0.90) / 0.06) ** 2)) * (0.40 + 0.60 * facing)
    lum *= 1.0 - 0.40 * _smoothstep(0.95, 1.005, r)
    pot_rgb = lum[..., None] * TERRACOTTA[None, None, :]
    soil = _smoothstep(0.86, 0.80, r)
    pot_rgb = pot_rgb * (1.0 - soil[..., None]) \
        + (SOIL_RGB * 0.9)[None, None, :] * soil[..., None]
    put(pot_a, pot_rgb)

    rng = np.random.default_rng(seed + 201)
    for n, reach, width_, root in ((9, 1.35, 0.30, 0.16), (6, 0.78, 0.25, 0.05)):
        start = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(n):
            th = start + 2.0 * math.pi * k / n + rng.uniform(-0.16, 0.16)
            leaf_l = reach * rng.uniform(0.80, 1.12)
            leaf_w = width_ * rng.uniform(0.85, 1.15)
            ux, uy = math.cos(th), math.sin(th)
            ls = dx * ux + dy * uy - root
            lt = dy * ux - dx * uy
            q = ((ls - 0.5 * leaf_l) / (0.5 * leaf_l)) ** 2 + (lt / leaf_w) ** 2
            leaf_a = 1.0 - _smoothstep(1.0 - 4.0 * e, 1.0 + 4.0 * e, q)
            # Lit as a whole by which way it points, shaded down into the
            # rosette at its root, with a waxy line along the midrib.
            fac = 0.72 + 0.42 * max(-(ux + uy) * 0.707, 0.0)
            leaf_lum = fac * (0.55 + 0.45 * np.clip(ls / leaf_l, 0.0, 1.0))
            leaf_lum = leaf_lum * (1.0 + 0.35 * np.exp(-((lt / (0.28 * leaf_w)) ** 2)))
            # Each leaf its own green, drifting a little towards yellow --
            # a rosette of one flat colour reads as plastic.
            tint = LEAF_RGB * rng.uniform(0.78, 1.15) + rng.uniform(-0.012, 0.020) \
                * np.array([1.0, 0.6, -0.3], dtype=np.float32)
            put(leaf_a, leaf_lum[..., None] * np.clip(tint, 0.0, None)[None, None, :])

    # The foliage floats above the desk, so the pool of shadow is wide and
    # soft rather than a tight ring at the pot's foot.
    shadow = 1.0 - _smoothstep(0.75, 1.55, np.hypot(dx - 0.30, dy - 0.36))
    return rgb, alpha, shadow


def _clutter(w: int, h: int, bars: np.ndarray, pool: np.ndarray,
             seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Everything standing on the desk that is not the deck: plant, pack, mug.

    Returns (rgb, alpha, shadow) over the whole window, already sitting in
    the shared desk lighting, ready to composite before the highlight
    roll-off so the clutter tones down with the wood it stands on.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    shadow = np.zeros((h, w), dtype=np.float32)
    for item_rgb, item_a, item_sh in (_plant(xx, yy, w, h, seed),
                                      _pack(xx, yy, w, h),
                                      _mug(xx, yy, w, h)):
        rgb = rgb * (1.0 - item_a[..., None]) \
            + _stand_light(item_rgb, bars, pool) * item_a[..., None]
        alpha = alpha + item_a * (1.0 - alpha)
        shadow = np.maximum(shadow, item_sh)
    return rgb, alpha, shadow


# --------------------------------------------------------------------------
# the deck (turntable hardware, background image layer)

# Nothing on the deck moves -- the platter turns, but it is rotationally
# symmetric like the grooves, so the plinth, platter edge and tonearm
# all live in the background image alongside the desk. The one place the deck
# touches the animation is the tonearm's footprint over the vinyl: the wipe
# glints stream in the layer above, so `_glint_mask` blanks them out under the
# arm -- a scuff glinting on top of the arm would put the record above it.
#
# Geometry is in units of the disc radius, origin on the spindle, +y down,
# matching `_disc_geometry`. The plinth is off-centre on purpose -- platter
# front-left, arm rear-right, like any deck -- because the record itself has
# to stay centred in the window: the streamed frames land there, and nothing
# else about the layout is negotiable.

PLATTER_R = 1.045          # machined edge peeking out from under the record

PLINTH_CENTRE = (0.065, -0.10)
PLINTH_HALF = (1.385, 1.42)
PLINTH_CORNER = 0.14

ARM_PIVOT = (0.98, -1.08)
ARM_STYLUS = (0.55, 0.28)  # parked mid-side; the tube clears the label region
ARM_BASE_R = 0.15
ARM_TUBE_R = 0.023

# Light in this scene falls from up-left -- the disc's shadow goes down-right
# -- so everything the deck casts is offset the same way.
DECK_SHADOW_OFF = (0.055, 0.065)

# Satin black, linear light: the plinth is backdrop under text, so it sits a
# step darker than the walnut and shows the lamp only as a sheen.
PLINTH_RGB = np.array([0.052, 0.054, 0.060], dtype=np.float32)
STEEL_RGB = np.array([0.92, 0.94, 1.00], dtype=np.float32)


def _capsule_dist(dx: np.ndarray, dy: np.ndarray,
                  a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Distance from every pixel to the segment a-b, in disc-radius units."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    t = np.clip(((dx - a[0]) * ux + (dy - a[1]) * uy)
                / max(ux * ux + uy * uy, 1e-9), 0.0, 1.0)
    return np.hypot(dx - (a[0] + t * ux), dy - (a[1] + t * uy))


def _plinth(dx: np.ndarray, dy: np.ndarray, bars: np.ndarray, pool: np.ndarray,
            px: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The deck body, in the same linear light as the wood it stands on.

    Returns (rgb, alpha, shadow): the shadow is where the plinth falls on the
    desk, already soft, to be multiplied in before the plinth goes down.
    """
    cx, cy = PLINTH_CENTRE
    hx, hy = PLINTH_HALF

    def sdf(x, y):
        qx = np.abs(x - cx) - (hx - PLINTH_CORNER)
        qy = np.abs(y - cy) - (hy - PLINTH_CORNER)
        return (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
                + np.minimum(np.maximum(qx, qy), 0.0) - PLINTH_CORNER)

    d = sdf(dx, dy)
    alpha = 1.0 - _smoothstep(-px, px, d)

    # Brushed satin rather than dead flat: faint streaks, no figure. Enough
    # texture to read as a surface, not enough to compete with the text.
    streak = _value_noise(dx.shape, 220, 3, seed + 91) - 0.5
    rgb = PLINTH_RGB[None, None, :] * (1.0 + 0.16 * streak)[..., None]
    # Satin responds to the room light more gently than varnished timber does.
    rgb *= (0.55 + 0.80 * bars)[..., None]
    rgb += (0.010 * bars)[..., None] * SUNLIGHT[None, None, :]
    rgb *= (0.62 + 0.38 * pool)[..., None]

    # Bevelled edge: a lit line on the sides facing the lamp, a darker one on
    # the sides facing away. The normal is approximated from the box axes,
    # which is exact on the flats and close enough round the corners.
    nxp, nyp = (dx - cx) / hx, (dy - cy) / hy
    bevel = np.exp(-(((d + 0.020) / 0.014) ** 2))
    facing = np.clip(-(nxp + nyp) * 0.707, 0.0, 1.0)
    rgb += (0.040 * bevel * facing)[..., None] * SUNLIGHT[None, None, :]
    rgb *= (1.0 - 0.45 * bevel * np.clip((nxp + nyp) * 0.707, 0.0, 1.0))[..., None]

    shadow = 1.0 - _smoothstep(-0.02, 0.16, sdf(dx - 0.05, dy - 0.06))
    return rgb, alpha, shadow


def _tonearm(dx: np.ndarray, dy: np.ndarray,
             px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The arm at rest over the grooves: counterweight, tube, headshell, base.

    Returns (rgb, alpha, shadow). Painted into the background by
    `render_desk`, and consulted -- alpha and shadow only -- by `_glint_mask`,
    so the streamed wipe glints never light up on top of the arm.
    """
    ax, ay = ARM_PIVOT
    sx, sy = ARM_STYLUS
    ux, uy = sx - ax, sy - ay
    reach = math.hypot(ux, uy)
    ux, uy = ux / reach, uy / reach
    nx, ny = -uy, ux
    edge = 1.6 * px

    def along(s: float) -> tuple[float, float]:
        return ax + s * ux, ay + s * uy

    # Signed distance off the arm's axis, shared by every tubular part. The
    # specular line sits on whichever side of the tube faces the light.
    perp = (dx - ax) * nx + (dy - ay) * ny
    hlt = -0.35 if (nx + ny) > 0.0 else 0.35

    def cyl(radius: float, base: float, gloss: float, tint: np.ndarray) -> np.ndarray:
        t = np.clip(perp / radius, -1.5, 1.5)
        lum = base * (0.55 + 0.45 * np.clip(1.0 - t * t, 0.0, 1.0)) \
            + gloss * np.exp(-((t - hlt) / 0.5) ** 2)
        return lum[..., None] * tint[None, None, :]

    def capsule(a, b, radius: float) -> np.ndarray:
        return 1.0 - _smoothstep(radius - edge, radius + edge,
                                 _capsule_dist(dx, dy, a, b))

    # Bottom to top: whatever is painted later sits on top.
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    parts.append((capsule(along(-0.26), along(-0.08), 0.052),
                  cyl(0.052, 0.26, 0.30, STEEL_RGB)))
    parts.append((capsule(along(0.0), along(reach - 0.13), ARM_TUBE_R),
                  cyl(ARM_TUBE_R, 0.40, 0.50, STEEL_RGB)))
    # The headshell: the wider end that carries the cartridge, a step brighter
    # than the vinyl under it or the arm just stops instead of ending.
    parts.append((capsule(along(reach - 0.15), along(reach - 0.01), 0.038),
                  cyl(0.038, 0.17, 0.22, STEEL_RGB)))

    # The pivot base: a dark housing with a lit rim, a bearing collar, and a
    # cap over the bearing itself.
    rb = np.hypot(dx - ax, dy - ay)
    radial_lit = np.clip(-((dx - ax) + (dy - ay)) / np.maximum(rb, 1e-6) * 0.707,
                         0.0, 1.0)
    lum = np.full_like(dx, 0.11)
    lum += 0.24 * np.exp(-(((rb - (ARM_BASE_R - 0.020)) / 0.012) ** 2)) * radial_lit
    lum += 0.20 * np.exp(-(((rb - 0.062) / 0.012) ** 2))
    lum += 0.10 * np.exp(-((rb / 0.028) ** 2))
    parts.append((1.0 - _smoothstep(ARM_BASE_R - edge, ARM_BASE_R + edge, rb),
                  lum[..., None] * STEEL_RGB[None, None, :]))

    # The stylus itself: one bright point where the needle meets the groove,
    # which is what says "playing" rather than "lying nearby".
    rs = np.hypot(dx - sx, dy - sy)
    parts.append((0.60 * np.exp(-((rs / (2.4 * px)) ** 2)),
                  np.ones(dx.shape + (3,), dtype=np.float32)))

    rgb = np.zeros(dx.shape + (3,), dtype=np.float32)
    alpha = np.zeros_like(dx)
    for a, c in parts:
        rgb = rgb * (1.0 - a[..., None]) + c * a[..., None]
        alpha = alpha + a * (1.0 - alpha)

    # The shadow, cast down-right onto whatever the arm crosses -- mostly the
    # record. One soft union of the same shapes; the arm floats well clear of
    # the surface, so the penumbra is wide and the offset generous.
    ox, oy = DECK_SHADOW_OFF
    dxs, dys = dx - ox, dy - oy

    def scap(a, b, radius: float) -> np.ndarray:
        return 1.0 - _smoothstep(radius - 0.005, radius + 0.045,
                                 _capsule_dist(dxs, dys, a, b))

    shadow = scap(along(-0.26), along(-0.08), 0.052)
    shadow = np.maximum(shadow, scap(along(0.0), along(reach - 0.13), ARM_TUBE_R))
    shadow = np.maximum(shadow, scap(along(reach - 0.15), along(reach - 0.01), 0.038))
    shadow = np.maximum(shadow, 1.0 - _smoothstep(
        ARM_BASE_R - 0.005, ARM_BASE_R + 0.05, np.hypot(dxs - ax, dys - ay)))
    return rgb, alpha, shadow


def render_desk(width: int, height: int, disc_size: int, opacity: float = 0.55,
                sheen_deg: float = 35.0, sheen_rgb: np.ndarray | None = None,
                seed: int = 7, light: float = 1.0, brightness: float = 1.0,
                blur: float = 0.0, turntable: bool = False) -> bytes:
    """PNG of the record on a wooden desk -- by default on a turntable
    standing there -- sized to the window.

    Uploaded once, so it is worth compressing properly. Send it with a
    `cscaled` layout: the record stays round however the window is resized.
    """
    scale = min(1.0, DESK_MAX_DIM / max(width, height, 1))
    w, h = max(2, int(width * scale)), max(2, int(height * scale))
    d = max(16, int(disc_size * scale))
    d -= d % 2

    bars, pool = _desk_lighting(w, h, light)
    rgb = _wood(w, h, seed, bars, pool)

    # Everything from here down works in disc-radius units about the spindle,
    # the same frame the deck geometry is defined in.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    radius = d / 2.0
    dxn = (xx - (w - 1) / 2.0) / radius
    dyn = (yy - (h - 1) / 2.0) / radius
    pxn = 1.0 / radius

    if turntable:
        p_rgb, p_a, p_sh = _plinth(dxn, dyn, bars, pool, pxn, seed)
        rgb *= (1.0 - 0.36 * p_sh)[..., None]
        rgb = rgb * (1.0 - p_a[..., None]) + p_rgb * p_a[..., None]

    # The clutter goes down after the deck: on a window too narrow to keep
    # them apart the items stand on the plinth rather than vanishing under it.
    c_rgb, c_a, c_sh = _clutter(w, h, bars, pool, seed)
    rgb *= (1.0 - 0.38 * c_sh * (1.0 - c_a))[..., None]
    rgb = rgb * (1.0 - c_a[..., None]) + c_rgb * c_a[..., None]

    # Roll the highlights off instead of clipping them: past 1.0 a hard clip
    # loses red first and the wood turns green.
    rgb = 1.0 - np.exp(-1.28 * np.maximum(rgb * max(brightness, 0.0), 0.0))

    if turntable:
        # The platter, of which only a machined sliver shows past the record.
        # Its edge is a turned circle, so it reflects anisotropically like the
        # grooves do -- same light, same two lobes.
        rr_w = np.hypot(dxn, dyn)
        lit = np.abs(np.cos(np.arctan2(dyn, dxn) - math.radians(sheen_deg)))
        plat_a = 1.0 - _smoothstep(PLATTER_R - pxn, PLATTER_R + pxn, rr_w)
        lum = 0.30 + 0.22 * lit ** 2
        lum += 0.45 * np.exp(-(((rr_w - (PLATTER_R - 0.014)) / (2.5 * pxn)) ** 2)) \
            * (0.35 + 0.65 * lit)
        lum *= 1.0 - 0.60 * _smoothstep(PLATTER_R - 3.0 * pxn, PLATTER_R, rr_w)
        plat_rgb = lum[..., None] * STEEL_RGB[None, None, :]
        # The platter stands on the plinth, so its shadow lands there...
        rsh = np.hypot(dxn - 0.030, dyn - 0.035)
        rgb *= (1.0 - 0.38 * (1.0 - _smoothstep(PLATTER_R * 0.99, PLATTER_R * 1.16, rsh))
                * (1.0 - plat_a))[..., None]
        rgb = rgb * (1.0 - plat_a[..., None]) + plat_rgb * plat_a[..., None]
        # ...while the record lies flat on the platter, so its own shadow is a
        # tight seam at the edge, not the broad pool it throws on a bare desk.
        rsh = np.hypot(dxn - 0.008, dyn - 0.010)
        rgb *= (1.0 - 0.35 * (1.0 - _smoothstep(0.992, 1.03, rsh)))[..., None]
    else:
        # The record throws a shadow down and to the right of the light, offset
        # by a few percent of its own diameter -- it is lying on the desk, not
        # floating. Two parts, like any contact shadow: a hard dark line right
        # where the disc meets the wood, and a broad soft penumbra around it.
        r = np.hypot(dxn - 0.032, dyn - 0.032)
        rgb *= (1.0 - 0.55 * (1.0 - _smoothstep(0.985, 1.05, r)))[..., None]
        r = np.hypot(dxn - 0.090, dyn - 0.090)
        rgb *= (1.0 - 0.30 * (1.0 - _smoothstep(0.97, 1.34, r)))[..., None]
        # The well of the spindle hole. The dot of desk at the bottom of it
        # sits in the record's shadow, but it is lit from almost straight
        # above, so it keeps more light than the shadowed rim around the disc
        # does -- without this it reads as a hole in the render, not a hole in
        # the record.
        rc = np.hypot(dxn, dyn)
        rgb *= 1.0 + 1.7 * np.exp(-((rc / (0.8 * HOLE_R)) ** 2))[..., None]

    x0, y0 = (w - d) // 2, (h - d) // 2
    disc = _disc_rgba(d, 1.0, sheen_deg, sheen_rgb, recess=False)
    if not turntable:
        # The spindle hole goes all the way through: a dot of desk shows in the
        # middle, and it holds still while the label turns around it. Punched a
        # shade smaller than the hole in the label frames, so if the two layers
        # ever drift a pixel apart it is dark vinyl that peeks out, not bright
        # wood. On the deck the hole stays dark vinyl instead.
        rr, _, px = _disc_geometry(d)
        disc[..., 3] *= _smoothstep(0.8 * HOLE_R - 1.5 * px, 0.8 * HOLE_R + 0.5 * px, rr)
    a = disc[..., 3:4]
    patch = rgb[y0:y0 + d, x0:x0 + d]
    rgb[y0:y0 + d, x0:x0 + d] = patch * (1.0 - a) + disc[..., :3] * a

    # After the disc is down, so the blur crosses the join and the record settles
    # into the wood instead of sitting on it as a sharp cut-out. The desk may be
    # drawn smaller than the window and stretched back up by kitty, so the radius
    # has to travel with it to mean the same thing on screen.
    rgb = _blur_rgb(rgb, blur * scale)

    if turntable:
        # The arm goes on after the legibility blur: it is the nearest thing
        # to the viewer, in the same focal plane as the label that the logo
        # layer keeps sharp.
        arm_rgb, arm_a, arm_sh = _tonearm(dxn, dyn, pxn)
        rgb *= (1.0 - 0.30 * arm_sh * (1.0 - arm_a))[..., None]
        rgb = rgb * (1.0 - arm_a[..., None]) + arm_rgb * arm_a[..., None]

    # A whisper of grain, after the blur so it survives it. This is dither, not
    # decoration: the lamp pool and the shadow are long slow gradients across
    # dark browns, exactly where 8-bit banding shows.
    rgb += (np.random.default_rng(seed + 55).random((h, w, 1), dtype=np.float32)
            - 0.5) * (1.6 / 255.0)

    out = np.concatenate([rgb, np.full((h, w, 1), opacity, dtype=np.float32)], axis=-1)
    return _png(Image.fromarray((np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGBA"),
                compress_level=6)


# --------------------------------------------------------------------------
# what turns (window logo layer): the label, and the wipe marks around it


def _label_alpha(diameter: int, opacity: float) -> Image.Image:
    radius = diameter / 2.0
    yy, xx = np.mgrid[0:diameter, 0:diameter].astype(np.float32)
    centre = (diameter - 1) / 2.0
    r = np.hypot(xx - centre, yy - centre) / radius
    hole = HOLE_R / LABEL_R
    a = (1.0 - _smoothstep(1.0 - 2.0 / radius, 1.0, r)) * _smoothstep(
        hole - 1.5 / radius, hole + 1.5 / radius, r
    )
    return Image.fromarray((a * opacity * 255.0 + 0.5).astype(np.uint8), "L")


def _shade_label(img: Image.Image, sheen_rgb: np.ndarray) -> Image.Image:
    """Slight radial darkening plus a ring, so the paper reads as inset."""
    size = img.size[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = (size - 1) / 2.0
    r = np.hypot(xx - centre, yy - centre) / (size / 2.0)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a *= (1.0 - 0.16 * _smoothstep(0.60, 1.02, r))[..., None]
    # A lit paper edge, so the label stays legible as a label even when the
    # cover art itself is nearly black.
    a += (0.30 * np.exp(-(((r - 0.955) / 0.028) ** 2)))[..., None] * sheen_rgb[None, None, :]
    a += (0.10 * np.exp(-(((r - 0.055) / 0.045) ** 2)))[..., None] * sheen_rgb[None, None, :]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")


def _placeholder_art(sheen_rgb: np.ndarray, size: int = 512) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.array([0.15, 0.14, 0.13], dtype=np.float32)
    a = base[None, None, :] + (sheen_rgb * 0.28)[None, None, :] * (0.30 + 0.70 * yy)[..., None]
    a += 0.045 * np.sin(xx * 42.0)[..., None]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")


def render_label_frames(art_bytes: bytes | None, diameter: int, frames: int,
                        opacity: float = 0.60) -> tuple[list[Image.Image], np.ndarray]:
    """One RGBA label per rotation step, plus the artwork's dominant colour.

    Rotating a square about its centre leaves its inscribed circle intact, so
    the art is rotated first and circle-masked afterwards -- the mask edge stays
    perfectly clean instead of being resampled every frame.
    """
    art = load_artwork(art_bytes)
    sheen = dominant_color(art) if art is not None else NEUTRAL_SHEEN.copy()
    if art is None:
        art = _placeholder_art(sheen)

    master = max(diameter * 2, 256)
    art_master = art.resize((master, master), Image.LANCZOS)
    alpha = _label_alpha(diameter, opacity)

    out = []
    for i in range(frames):
        lab = _shade_label(
            art_master.rotate(-360.0 * i / frames, resample=Image.BICUBIC)
                      .resize((diameter, diameter), Image.LANCZOS), sheen
        ).convert("RGBA")
        lab.putalpha(alpha)
        out.append(lab)
    return out, sheen


# Wipe marks, tuned by eye against a dark desk. Barely there is the point: on a
# record that is actually kept well you catch them at the edge of noticing as
# they pass through the light, and that flicker is enough to read the whole disc
# as turning. Anything bolder and it stops being a record catching the lamp and
# starts being one that wants cleaning.
GLINT_GAIN = 0.056

# Vinyl reflects the room, not the sleeve, so the glints stay the colour of the
# rest of the sheen rather than picking up the cover art.
GLINT_RGB = NEUTRAL_SHEEN / NEUTRAL_SHEEN.max()


def _glint_mask(size: int, sheen_deg: float,
                turntable: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where on the disc a wipe mark could show, and the noise coordinates.

    Everything here is fixed to the lamp, not to the record: the mask says how
    brightly a scuff *would* glint if one happened to be at that pixel, and the
    frames then slide scuffs through it. That is the whole trick -- the light
    stays put, the marks go round.
    """
    r, theta, px = _disc_geometry(size)
    lit = np.abs(np.cos(theta - math.radians(sheen_deg)))
    envelope = np.exp(-(((r - 0.74) / 0.46) ** 2)) * (1.0 + 0.55 * _smoothstep(0.4, 1.0, r))

    # Only ever visible in the lobes -- a scuff you can see in the shadow half
    # is a scuff drawn on, not one caught by the light. Smooth vinyl shows a
    # mark more readily than grooved vinyl, so the dead wax and the run-out
    # margin are damped rather than excluded.
    mask = (0.60 + 0.40 * _grooved(r)) * lit ** 4.0 * envelope
    # Nothing off the edge of the record, and nothing under the paper.
    mask *= 1.0 - _smoothstep(1.0 - 1.6 * px, 1.0, r)
    mask *= _smoothstep(LABEL_R - 1.5 * px, LABEL_R + 1.5 * px, r)

    if turntable:
        # The tonearm sits in the background layer, visually *above* the vinyl
        # -- so the glints, which stream in the layer over it, have to go dark
        # under its footprint, and mostly-dark under its shadow, or the scuffs
        # would sweep across the top of the arm.
        _, arm_a, arm_sh = _tonearm(r * np.cos(theta), r * np.sin(theta), px)
        mask *= (1.0 - arm_a) * (1.0 - 0.55 * arm_sh)

    ri, ti = _polar_index(r, theta)
    return mask.astype(np.float32), ri, ti


def render_spin_frames(art_bytes: bytes | None, disc_size: int, frames: int,
                       label_opacity: float = 0.60, opacity: float = 0.60,
                       sheen_deg: float = 35.0, wipe: float = 1.0,
                       turntable: bool = False) -> tuple[list[bytes], np.ndarray]:
    """One disc-sized PNG per rotation step, plus the artwork's dominant colour.

    Everything on the record that actually turns, in a single frame: the paper
    label in the middle, and the wipe marks sweeping across the vinyl around it.
    The rest is transparent, because the grooves, the rim and the lamp's
    reflection are already sitting still in the background image underneath.

    Compositing can only lighten what is beneath it, which suits the subject:
    a scuff catches the light, it does not cast a shadow. So only the positive
    half of the noise is kept.

    `wipe` scales how worn the record looks; 0 is straight off the press, and
    with nothing left to catch the light only the label reads as turning.
    """
    label_d = label_diameter(disc_size)
    labels, sheen = render_label_frames(art_bytes, label_d, frames, label_opacity)

    mask, ri, ti = _glint_mask(disc_size, sheen_deg, turntable)
    mask *= max(min(opacity, 1.0), 0.0)
    coarse = _polar_texture(26, 150, 11)
    fine = _polar_texture(9, 420, 29)
    rgb = np.broadcast_to((GLINT_RGB * 255.0 + 0.5).astype(np.uint8),
                          (disc_size, disc_size, 3))
    inset = (disc_size - label_d) // 2

    out = []
    for i, label in enumerate(labels):
        # Negative, so the marks turn the same way the label does.
        turn = -int(round(i * NOISE_W / frames))
        scuff = _sample_polar(coarse, ri, ti, turn) - 0.5
        scuff += 0.6 * (_sample_polar(fine, ri, ti, turn) - 0.5)
        alpha = np.clip(GLINT_GAIN * max(wipe, 0.0) * np.maximum(scuff, 0.0) * mask, 0.0, 1.0)

        frame = Image.fromarray(np.dstack(
            [rgb, (alpha * 255.0 + 0.5).astype(np.uint8)[..., None]]), "RGBA")
        frame.alpha_composite(label, (inset, inset))
        # These are built once per track and then replayed, so they are not on
        # the frame path in the sense `_png` means -- squeezing them halves
        # what goes down the socket every frame for a few ms here.
        out.append(_png(frame, compress_level=6))
    return out, sheen
