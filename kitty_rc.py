"""Minimal kitty remote-control client.

kitty speaks the same DCS-framed JSON on its UNIX socket as it does on a tty:

    ESC P @kitty-cmd {"cmd": ..., "payload": ...} ESC \\

Going straight to the socket instead of shelling out to `kitty @` matters here:
the animation loop sends a command every frame, and a process spawn per frame
would cost more than the frame itself.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import secrets
import socket
import stat

PREFIX = b"\x1bP@kitty-cmd"
SUFFIX = b"\x1b\\"
VERSION = [0, 26, 0]

# kitty reassembles image uploads from base64 chunks tagged with a stream id,
# terminated by an empty chunk. 2KiB of raw data per chunk is what kitty's own
# client uses.
CHUNK = 2048

DEFAULT_PATTERN = "/tmp/kitty-*"


def find_sockets(pattern: str = DEFAULT_PATTERN) -> list[str]:
    found = []
    for path in sorted(glob.glob(pattern)):
        try:
            if stat.S_ISSOCK(os.stat(path).st_mode):
                found.append(path)
        except OSError:
            pass
    return found


class KittyRC:
    """Talks to every listening kitty instance, reconnecting as they come and go."""

    def __init__(self, pattern: str = DEFAULT_PATTERN):
        self.pattern = pattern
        self._conns: dict[str, socket.socket] = {}

    def rescan(self) -> int:
        """Pick up kitty instances started since the last scan."""
        for path in find_sockets(self.pattern):
            self._conns.setdefault(path, None)  # type: ignore[arg-type]
        return len(self._conns)

    def close(self) -> None:
        for sock in self._conns.values():
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._conns.clear()

    def _connect(self, path: str) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(path)
        return sock

    def send(self, cmd: str, payload: dict | None = None, want_response: bool = False):
        """Broadcast a command. Returns a list of (socket_path, response|None)."""
        msg = {"cmd": cmd, "version": VERSION, "payload": payload or {}}
        if not want_response:
            msg["no_response"] = True
        return self.write(PREFIX + json.dumps(msg).encode() + SUFFIX,
                          want_response=want_response)

    def write(self, blob: bytes, want_response: bool = False):
        """Send raw pre-encoded command bytes to every kitty we know about."""
        results = []
        for path in list(self._conns):
            for attempt in (0, 1):  # one retry: kitty may have closed an idle conn
                sock = self._conns.get(path)
                try:
                    if sock is None:
                        sock = self._conns[path] = self._connect(path)
                    sock.sendall(blob)
                    results.append((path, self._read(sock) if want_response else None))
                    break
                except OSError:
                    if sock is not None:
                        try:
                            sock.close()
                        except OSError:
                            pass
                    self._conns[path] = None  # type: ignore[assignment]
                    if attempt:  # still dead after a reconnect: drop it
                        self._conns.pop(path, None)
        return results

    @staticmethod
    def _read(sock: socket.socket) -> dict | None:
        buf = b""
        while SUFFIX not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        start = buf.find(PREFIX)
        if start < 0:
            return None
        body = buf[start + len(PREFIX) : buf.find(SUFFIX, start)]
        try:
            return json.loads(body)
        except ValueError:
            return None

    # -- convenience wrappers -------------------------------------------------

    def clear_logo(self, match: str = "all") -> bool:
        # `alpha` is not optional, even for a removal: kitty calls float() on it
        # unconditionally, so leaving it out makes the command raise inside kitty
        # and the record stays on screen. -1 is kitty's "use the default".
        return self._clear("set-window-logo",
                           {"data": "-", "position": "center", "alpha": -1.0, "match": match})

    def clear_background(self) -> bool:
        return self._clear("set-background-image",
                           {"data": "-", "layout": "configured", "all": True})

    def _clear(self, cmd: str, payload: dict) -> bool:
        """Unlike the frame stream, clears wait for kitty to answer.

        They are the last thing we do on the way out, and a clear that kitty
        rejects leaves the record burned onto the window after we are gone --
        the one failure here worth hearing about rather than dropping.
        """
        return all(r[1] and r[1].get("ok")
                   for r in self.send(cmd, payload, want_response=True))

    def healthy(self) -> bool:
        return any(r[1] and r[1].get("ok") for r in self.send("ls", {}, want_response=True))


# -- image streaming ----------------------------------------------------------


def encode_image(cmd: str, png: bytes, extra: dict) -> bytes:
    """Pre-encode an image upload into the exact bytes to put on the socket.

    Doing this ahead of time means the animation loop only ever does a single
    `sendall` per frame -- no JSON, no base64, no image work at frame time.
    """
    stream_id = secrets.token_urlsafe(16)
    parts = [base64.standard_b64encode(png[i:i + CHUNK]).decode()
             for i in range(0, len(png), CHUNK)]
    parts.append("")  # empty chunk terminates the upload
    out = []
    for part in parts:
        msg = {"cmd": cmd, "version": VERSION, "stream": True, "stream_id": stream_id,
               "no_response": True, "payload": dict(extra, data=part)}
        out.append(PREFIX + json.dumps(msg).encode() + SUFFIX)
    return b"".join(out)


def encode_logo(png: bytes, position: str = "center", alpha: float = 1.0,
                match: str = "all") -> bytes:
    return encode_image("set-window-logo", png,
                        {"position": position, "alpha": alpha, "match": match})


def encode_background(png: bytes, layout: str = "centered") -> bytes:
    return encode_image("set-background-image", png, {"layout": layout, "all": True})
