#!/usr/bin/env python3
"""A spinning vinyl record behind your kitty terminal, wearing the cover art of
whatever Apple Music is playing.

kitty paints, in order: background colour, background image, window logo, text.
That gives us two layers below the text, and the record splits neatly across
them -- the grooves, rim and lamp reflection go in the background image and are
uploaded once, and the parts that actually turn (the label and the wipe marks
on the vinyl) are streamed per frame. See render.py.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import signal
import sys
import termios
import threading
import time

import kitty_rc
import nowplaying
import render

KITTY_DIR = os.path.join(os.path.expanduser("~"), ".config", "kitty")
KITTY_CONF = os.path.join(KITTY_DIR, "kitty.conf")

BEGIN = "# >>> kitty-vinyl >>>"
END = "# <<< kitty-vinyl <<<"
CONF_BLOCK = f"""{BEGIN}
# Added by kitty-vinyl so the spinner can talk to kitty. Delete to uninstall.
allow_remote_control socket-only
listen_on unix:/tmp/kitty
{END}"""


def log(msg: str) -> None:
    print(f"[vinyl] {msg}", flush=True)


# --------------------------------------------------------------------------
# setup


def install_config() -> bool:
    os.makedirs(KITTY_DIR, exist_ok=True)
    existing = ""
    if os.path.exists(KITTY_CONF):
        with open(KITTY_CONF, encoding="utf-8") as fh:
            existing = fh.read()

    if BEGIN in existing and END in existing:
        head, _, rest = existing.partition(BEGIN)
        _, _, tail = rest.partition(END)
        updated = head + CONF_BLOCK + tail
        if updated == existing:
            log("kitty.conf already has the remote-control block")
            return False
    else:
        if existing:
            backup = KITTY_CONF + ".pre-vinyl"
            if not os.path.exists(backup):
                import shutil
                shutil.copy2(KITTY_CONF, backup)
                log(f"backed up kitty.conf -> {backup}")
        updated = existing.rstrip("\n") + ("\n\n" if existing else "") + CONF_BLOCK + "\n"

    with open(KITTY_CONF, "w", encoding="utf-8") as fh:
        fh.write(updated)
    log(f"patched {KITTY_CONF}")
    return True


def uninstall_config() -> None:
    if not os.path.exists(KITTY_CONF):
        return
    with open(KITTY_CONF, encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN in text and END in text:
        head, _, rest = text.partition(BEGIN)
        _, _, tail = rest.partition(END)
        with open(KITTY_CONF, "w", encoding="utf-8") as fh:
            fh.write(head.rstrip("\n") + "\n" + tail.lstrip("\n"))
        log("removed the kitty-vinyl block from kitty.conf")


def window_pixels(timeout: float = 0.4) -> tuple[int, int] | None:
    """Ask the terminal how big it is, in pixels (CSI 14 t)."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return None
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        os.close(fd)
        return None
    try:
        raw = list(old)
        raw[3] &= ~(termios.ICANON | termios.ECHO)  # lflag
        termios.tcsetattr(fd, termios.TCSANOW, raw)
        os.write(fd, b"\x1b[14t")
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], max(deadline - time.monotonic(), 0))
            if not ready:
                break
            buf += os.read(fd, 64)
            if b"t" in buf:
                break
        m = re.search(rb"\x1b\[4;(\d+);(\d+)t", buf)
        return (int(m.group(2)), int(m.group(1))) if m else None
    except OSError:
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        finally:
            os.close(fd)


# What the desk is drawn on when the window can't be measured (no tty under
# launchd). Only the aspect ratio matters -- kitty scales it to fit.
FALLBACK_WINDOW = (1600, 1000)


def parse_window(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", value.strip())
    if not m:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 2560x1600")
    return int(m.group(1)), int(m.group(2))


def choose_size(requested: int | None, window: tuple[int, int] | None) -> tuple[int, str]:
    if requested:
        return requested, "requested"
    if window:
        px = int(min(window) * 0.78)
        return max(420, min(px - px % 2, 1400)), f"auto from {window[0]}x{window[1]}px window"
    return 900, "default (could not measure the window)"


# --------------------------------------------------------------------------
# the daemon


class Vinyl:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rc = kitty_rc.KittyRC(args.socket_glob)
        self.fps = max(1.0, args.frames * args.rpm / 60.0)
        self.window = args.window or window_pixels()
        self.size, self.size_why = choose_size(args.size, self.window)
        self.label_d = render.label_diameter(self.size)

        self._lock = threading.Lock()
        self._frames: list[bytes] = []
        self._playing = False
        self._generation = 0

        self.stop = threading.Event()
        self._track_key: str | None = None

    # -- track watching ----------------------------------------------------

    def watch(self) -> None:
        rescan_at = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            if now >= rescan_at:
                self.rc.rescan()
                rescan_at = now + 5.0
            try:
                track = nowplaying.poll()
            except nowplaying.PermissionDenied as exc:
                log(str(exc))
                self.stop.set()
                return
            except Exception as exc:
                log(f"could not read Music: {exc}")
                track = None
            if track is not None:
                try:
                    self._apply(track)
                except Exception as exc:
                    log(f"track update failed: {exc}")
            self.stop.wait(self.args.poll)

    def _apply(self, track: nowplaying.Track) -> None:
        if not track.has_track:
            if self._track_key is not None:
                self._track_key = None
                with self._lock:
                    self._frames, self._playing = [], False
                log(f"{track.state} — clearing the record")
                self.rc.clear_logo(self.args.match)
            else:
                with self._lock:
                    self._playing = False
            return

        if track.key != self._track_key:
            self._load(track)
        with self._lock:
            self._playing = track.playing

    def _load(self, track: nowplaying.Track) -> None:
        started = time.monotonic()
        art = None
        if track.has_artwork:
            try:
                art = nowplaying.artwork()
            except Exception as exc:
                log(f"artwork fetch failed: {exc}")

        pngs, _sheen = render.render_spin_frames(
            art, self.size, self.args.frames, self.args.label_opacity,
            background_opacity(self.args), wipe=self.args.wipe
        )
        blobs = [kitty_rc.encode_logo(p, alpha=1.0, match=self.args.match) for p in pngs]

        with self._lock:
            self._frames = blobs
            self._generation += 1
        self._track_key = track.key
        wire = sum(len(b) for b in blobs) / len(blobs) / 1024
        log(f"{track}{'' if art else ' (no artwork)'} — "
            f"{self.args.frames} frames, {wire:.0f}KB each, {time.monotonic()-started:.1f}s")

    # -- animation ---------------------------------------------------------

    def put_background(self) -> None:
        opacity = background_opacity(self.args)
        if self.args.desk:
            w, h = self.window or FALLBACK_WINDOW
            png = render.render_desk(w, h, self.size, opacity,
                                     light=self.args.light, seed=self.args.desk_seed,
                                     brightness=self.args.desk_brightness,
                                     blur=self.args.blur)
            what = f"desk {w}x{h}px, disc {self.size}px ({self.size_why})"
        else:
            png = render.render_disc(self.size, opacity, blur=self.args.blur)
            what = f"disc {self.size}px ({self.size_why})"
        self.rc.write(kitty_rc.encode_background(png, layout=background_layout(self.args)))
        log(f"{what}, label {self.label_d}px, {len(png)/1024:.0f}KB")

    def spin(self) -> None:
        interval = 1.0 / self.fps
        phase = 0.0
        last_index = last_gen = -1
        last_send = 0.0
        prev = time.monotonic()

        while not self.stop.is_set():
            now = time.monotonic()
            delta, prev = now - prev, now

            with self._lock:
                frames, playing, gen = self._frames, self._playing, self._generation

            if frames:
                if playing:
                    phase = (phase + delta * self.fps) % len(frames)
                index = int(phase) % len(frames)
                # Re-send occasionally while paused so a kitty window opened
                # mid-track still gets the label. kitty skips identical uploads,
                # so this costs almost nothing.
                if index != last_index or gen != last_gen or now - last_send > 2.0:
                    self.rc.write(frames[index])
                    last_index, last_gen, last_send = index, gen, now

            self.stop.wait(max(interval - (time.monotonic() - now), 0.001))

    def shutdown(self) -> None:
        self.stop.set()
        try:
            if not self.rc.clear_logo(self.args.match):
                log("kitty would not lift the record off the window")
            if not self.rc.clear_background():
                log("kitty would not lift the desk off the window")
        except Exception as exc:
            log(f"cleanup failed: {exc}")
        self.rc.close()


def cmd_run(args: argparse.Namespace) -> int:
    vinyl = Vinyl(args)
    if not vinyl.rc.rescan():
        log(f"no kitty sockets matching {args.socket_glob!r}")
        log("run `vinyl setup`, then fully quit kitty (cmd+Q) and reopen it")
        return 1
    if not vinyl.rc.healthy():
        log("found a kitty socket but it would not answer — is allow_remote_control set?")
        return 1

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: vinyl.stop.set())

    vinyl.put_background()
    log(f"{args.rpm:g} rpm over {args.frames} frames = {vinyl.fps:.1f} fps")
    threading.Thread(target=vinyl.watch, daemon=True).start()
    try:
        vinyl.spin()
    finally:
        vinyl.shutdown()
        log("stopped, background cleared")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    if install_config():
        log("now fully quit kitty (cmd+Q) and reopen it, then: vinyl run")
    else:
        socks = kitty_rc.find_sockets(args.socket_glob)
        log(f"kitty sockets: {socks or 'none yet — quit and reopen kitty'}")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    uninstall_config()
    rc = kitty_rc.KittyRC(args.socket_glob)
    if rc.rescan():
        rc.clear_logo()
        rc.clear_background()
    log("done — restart kitty to close the remote-control socket too")
    return 0


def background_opacity(args: argparse.Namespace) -> float:
    """The desk is a whole wall of wood, the bare disc is a shape in the middle.

    The desk can afford to be more solid than the disc ever could -- there is
    much less of the terminal's own background left to show through either way.

    Neither figure touches the cover art, which is carried by the logo layer at
    `--label-opacity`. So this can be pulled back for the sake of the text
    without the label going with it.
    """
    if args.opacity is not None:
        return args.opacity
    return 0.62 if args.desk else 0.47


def background_layout(args: argparse.Namespace) -> str:
    """The desk has to cover the window; a bare disc just sits in the middle.

    `cscaled` keeps the aspect ratio, so the record stays round however the
    window is resized -- `scaled` would squash it.
    """
    return args.layout or ("cscaled" if args.desk else "centered")


def cmd_status(args: argparse.Namespace) -> int:
    print("kitty sockets:", kitty_rc.find_sockets(args.socket_glob) or "none")
    px = window_pixels()
    print("window size:  ", f"{px[0]}x{px[1]} px" if px else "unknown (not attached to a tty)")
    try:
        track = nowplaying.poll()
    except Exception as exc:
        print("music:         error:", exc)
        return 1
    print("music:        ", f"{track} [{track.state}]")
    if track.has_track:
        print("               album:", track.album, "| artwork:", track.has_artwork)
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Compose disc + label into a still PNG, so you can tune it without kitty."""
    import io
    from PIL import Image

    art = None
    try:
        track = nowplaying.poll()
        if track.has_track and track.has_artwork:
            art = nowplaying.artwork()
            print(f"artwork from: {track}")
    except Exception as exc:
        print(f"could not read Music ({exc}); using a placeholder label")

    window = args.window or window_pixels()
    size, why = choose_size(args.size, window)
    opacity = background_opacity(args)
    spin, _ = render.render_spin_frames(art, size, max(1, args.frames),
                                        args.label_opacity, opacity, wipe=args.wipe)
    # Any frame will do for a still; a later one shows the marks off the light
    # axis, which is a fairer picture of what most of a revolution looks like.
    turning = Image.open(io.BytesIO(spin[len(spin) // 3])).convert("RGBA")

    if args.desk:
        w, h = window or FALLBACK_WINDOW
        print(f"desk {w}x{h}px, disc {size}px ({why})")
        png = render.render_desk(w, h, size, opacity, light=args.light,
                                 seed=args.desk_seed, brightness=args.desk_brightness,
                                 blur=args.blur)
    else:
        print(f"disc {size}px ({why})")
        png = render.render_disc(size, opacity, blur=args.blur)

    top = Image.open(io.BytesIO(png)).convert("RGBA")
    # The desk is drawn small and left to kitty to scale; scale the turning
    # layer to match so the preview shows the same proportions as the real thing.
    spin_px = max(2, round(size * top.size[0] / (window or FALLBACK_WINDOW)[0])) \
        if args.desk else size
    if spin_px != size:
        turning = turning.resize((spin_px, spin_px), Image.LANCZOS)

    bg = Image.new("RGBA", top.size, _hex_rgba(args.bg))
    bg.alpha_composite(top)
    bg.alpha_composite(turning, ((top.size[0] - spin_px) // 2, (top.size[1] - spin_px) // 2))
    out = os.path.abspath(args.out)
    bg.convert("RGB").save(out)
    print(f"wrote {out}")
    return 0


def _hex_rgba(value: str) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vinyl", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--socket-glob", default=kitty_rc.DEFAULT_PATTERN,
                        help="glob for kitty's listen_on sockets (default: %(default)s)")

    look = argparse.ArgumentParser(add_help=False)
    look.add_argument("--size", type=int, default=None,
                      help="record diameter in device pixels (default: 78%% of the window)")
    look.add_argument("--opacity", type=float, default=None,
                      help="0-1, how strongly the background shows through "
                           "(default: 0.62 with the desk, 0.47 without)")
    look.add_argument("--label-opacity", type=float, default=0.85,
                      help="0-1, opacity of the cover art label (default: %(default)s)")
    look.add_argument("--blur", type=float, default=3.0,
                      help="how far, in device pixels, to soften the desk and the record "
                           "so text reads more easily over them; the cover art is never "
                           "blurred. 0 is a sharp record (default: %(default)s)")
    look.add_argument("--frames", type=int, default=75,
                      help="rotation steps per revolution; more is smoother but costs "
                           "a longer render on every track change (default: %(default)s)")
    look.add_argument("--rpm", type=float, default=12.0,
                      help="turntable speed, well under the 33.3333 of a real LP -- at "
                           "this size on screen that reads as frantic (default: %(default)s)")
    look.add_argument("--wipe", type=float, default=1.0,
                      help="how strongly the wipe marks catch the light as the record "
                           "turns; 0 is a pristine pressing (default: %(default)s)")
    look.add_argument("--layout", default=None,
                      choices=("centered", "scaled", "cscaled", "clamped", "tiled"),
                      help="kitty background_image_layout (default: cscaled with the desk, "
                           "centered without)")
    look.add_argument("--no-desk", dest="desk", action="store_false",
                      help="just the record on the terminal background, no wooden desk")
    look.add_argument("--light", type=float, default=0.0,
                      help="0-1, bars of light across the desk: 0 is evenly lit, "
                           "1 is full sun through a blind (default: %(default)s)")
    look.add_argument("--desk-brightness", type=float, default=1.0,
                      help="scales how brightly the desk is lit; below 1 is a darker "
                           "wood and more contrast under the text (default: %(default)s)")
    look.add_argument("--desk-seed", type=int, default=7,
                      help="reshuffles the wood grain (default: %(default)s)")
    look.add_argument("--window", type=parse_window, default=None,
                      help="window size in device pixels, WIDTHxHEIGHT (default: measured, "
                           "or 1600x1000 with no tty)")
    look.add_argument("--match", default="all",
                      help="which kitty windows get the turning layer (default: %(default)s)")
    look.add_argument("--poll", type=float, default=2.0,
                      help="seconds between Music.app checks (default: %(default)s)")

    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("run", parents=[common, look],
                   help="watch Music and spin the record").set_defaults(func=cmd_run)
    sub.add_parser("setup", parents=[common],
                   help="add remote control to kitty.conf").set_defaults(func=cmd_setup)
    sub.add_parser("uninstall", parents=[common],
                   help="undo setup and clear the record").set_defaults(func=cmd_uninstall)
    sub.add_parser("status", parents=[common],
                   help="show kitty sockets, window size and what Music is playing").set_defaults(func=cmd_status)
    prev = sub.add_parser("preview", parents=[common, look],
                          help="render a still, for tuning size and opacity")
    prev.add_argument("--bg", default="#1d2021", help="terminal background to preview against")
    prev.add_argument("--out", default="vinyl-preview.png", help="where to write the still")
    prev.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
