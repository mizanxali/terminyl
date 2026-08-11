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


def _wood(width: int, height: int, seed: int, light: float,
          brightness: float = 1.0) -> np.ndarray:
    """Float RGB of a lacquered plank, optionally with light bars raked across.

    `brightness` scales the whole thing before the highlight roll-off, so
    turning it down darkens the wood without draining the colour out of it.
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

    # A mug has stood on this desk. The ring is a varnish stain: thin, broken
    # unevenly around its circumference, and kept out where the record goes.
    ring_r = np.hypot((u - 0.135) * width, (v - 0.72) * height) / (0.052 * width)
    wobble = 0.45 + 0.55 * _smoothstep(0.35, 0.75, _fbm(shape, 8, 5, 2, seed + 23))
    lum *= 1.0 - 0.11 * np.exp(-(((ring_r - 1.0) * (0.026 * width)) / 3.2) ** 2) * wobble

    rgb = WOOD[None, None, :] * lum[..., None]

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

    # Shade is 45% of full light; the lit strips run hot and pick up the colour
    # of the light itself. With no bars this all lands on the flat 0.5.
    rgb *= (0.45 + 0.95 * bars)[..., None]
    rgb += (0.055 * bars)[..., None] * SUNLIGHT[None, None, :]
    # Varnish: a broad reflected sheen, strongest where the light lands.
    rgb += (0.038 * bars * rings)[..., None] * SUNLIGHT[None, None, :]

    # A pool of lamp light rather than an even field: warm where it lands, and
    # the corners fall away harder. The centre stays where it was -- brighter
    # would cost the text -- the shape comes from pushing the edges down.
    radial = np.hypot(u - 0.40, (v - 0.46) * 1.10) / 0.78
    pool = 1.0 - _smoothstep(0.22, 1.25, radial)
    rgb *= (0.60 + 0.40 * pool)[..., None]
    rgb += (0.030 * pool ** 2)[..., None] * SUNLIGHT[None, None, :]
    # Shadows lose the lamp before they lose the sky, so they cool off as they
    # darken -- red falls away fastest, blue not at all. Warm pool against cool
    # corners is most of what makes the light feel like a lamp.
    rgb *= 1.0 - (1.0 - pool)[..., None] * np.array([0.055, 0.030, 0.0],
                                                    dtype=np.float32)[None, None, :]

    # Roll the highlights off instead of clipping them: past 1.0 a hard clip
    # loses red first and the wood turns green.
    return 1.0 - np.exp(-1.28 * np.maximum(rgb * max(brightness, 0.0), 0.0))


def render_desk(width: int, height: int, disc_size: int, opacity: float = 0.55,
                sheen_deg: float = 35.0, sheen_rgb: np.ndarray | None = None,
                seed: int = 7, light: float = 1.0, brightness: float = 1.0,
                blur: float = 0.0) -> bytes:
    """PNG of the record lying on a wooden desk, sized to the window.

    Uploaded once, so it is worth compressing properly. Send it with a
    `cscaled` layout: the record stays round however the window is resized.
    """
    scale = min(1.0, DESK_MAX_DIM / max(width, height, 1))
    w, h = max(2, int(width * scale)), max(2, int(height * scale))
    d = max(16, int(disc_size * scale))
    d -= d % 2

    rgb = _wood(w, h, seed, light, brightness)

    x0, y0 = (w - d) // 2, (h - d) // 2
    # The record throws a shadow down and to the right of the light, offset by a
    # few percent of its own diameter -- it is lying on the desk, not floating.
    # Two parts, like any contact shadow: a hard dark line right where the disc
    # meets the wood, and a broad soft penumbra spreading out from it.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    off = 0.016 * d
    r = np.hypot(xx - ((w - 1) / 2.0 + off), yy - ((h - 1) / 2.0 + off)) / (d / 2.0)
    rgb *= (1.0 - 0.55 * (1.0 - _smoothstep(0.985, 1.05, r)))[..., None]
    off = 0.045 * d
    r = np.hypot(xx - ((w - 1) / 2.0 + off), yy - ((h - 1) / 2.0 + off)) / (d / 2.0)
    rgb *= (1.0 - 0.30 * (1.0 - _smoothstep(0.97, 1.34, r)))[..., None]
    # The well of the spindle hole. The dot of desk at the bottom of it sits in
    # the record's shadow, but it is lit from almost straight above, so it keeps
    # more light than the shadowed rim around the disc does -- without this it
    # reads as a hole in the render, not a hole in the record.
    rc = np.hypot(xx - (w - 1) / 2.0, yy - (h - 1) / 2.0)
    rgb *= 1.0 + 1.7 * np.exp(-((rc / (0.8 * HOLE_R * d / 2.0)) ** 2))[..., None]

    disc = _disc_rgba(d, 1.0, sheen_deg, sheen_rgb, recess=False)
    # The spindle hole goes all the way through: a dot of desk shows in the
    # middle, and it holds still while the label turns around it. Punched a
    # shade smaller than the hole in the label frames, so if the two layers ever
    # drift a pixel apart it is dark vinyl that peeks out, not bright wood.
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


def _glint_mask(size: int, sheen_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    ri, ti = _polar_index(r, theta)
    return mask.astype(np.float32), ri, ti


def render_spin_frames(art_bytes: bytes | None, disc_size: int, frames: int,
                       label_opacity: float = 0.60, opacity: float = 0.60,
                       sheen_deg: float = 35.0,
                       wipe: float = 1.0) -> tuple[list[bytes], np.ndarray]:
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

    mask, ri, ti = _glint_mask(disc_size, sheen_deg)
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
