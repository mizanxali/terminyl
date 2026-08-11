"""Read the current track out of the Apple Music app via AppleScript."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass

SEP = "|~|"

# `application "Music" is running` is checked first: talking to a non-running
# app would launch it, and nobody wants their terminal opening Music.
_STATE = f'''
if application "Music" is not running then return "notrunning"
tell application "Music"
    set pstate to (player state as text)
    if pstate is not "playing" and pstate is not "paused" then return pstate
    set tr to current track
    try
        set tid to (persistent ID of tr) as text
    on error
        set tid to ""
    end try
    try
        set tname to (name of tr) as text
    on error
        set tname to ""
    end try
    try
        set tartist to (artist of tr) as text
    on error
        set tartist to ""
    end try
    try
        set talbum to (album of tr) as text
    on error
        set talbum to ""
    end try
    try
        set nart to (count of artworks of tr)
    on error
        set nart to 0
    end try
    return pstate & "{SEP}" & tid & "{SEP}" & tname & "{SEP}" & tartist & "{SEP}" & talbum & "{SEP}" & (nart as text)
end tell
'''

_ARTWORK = '''
on run argv
    set outPath to item 1 of argv
    if application "Music" is not running then return "notrunning"
    tell application "Music"
        if player state is stopped then return "stopped"
        set tr to current track
        if (count of artworks of tr) is 0 then return "noart"
        set raw to (raw data of artwork 1 of tr)
    end tell
    set fh to open for access (POSIX file outPath) with write permission
    set eof fh to 0
    write raw to fh
    close access fh
    return "ok"
end run
'''


class PermissionDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class Track:
    state: str  # playing | paused | stopped | notrunning
    track_id: str
    name: str
    artist: str
    album: str
    has_artwork: bool

    @property
    def playing(self) -> bool:
        return self.state == "playing"

    @property
    def has_track(self) -> bool:
        return self.state in ("playing", "paused")

    @property
    def key(self) -> str:
        """Stable identity for a track, used to decide when to re-render."""
        if self.track_id:
            return self.track_id
        return hashlib.sha1(
            f"{self.name}\0{self.artist}\0{self.album}".encode()
        ).hexdigest()[:16]

    def __str__(self) -> str:
        if not self.has_track:
            return self.state
        return f"{self.name} — {self.artist}" if self.artist else self.name


IDLE = Track("stopped", "", "", "", "", False)


def _osascript(script: str, *args: str, timeout: float = 15.0) -> str:
    proc = subprocess.run(
        ["osascript", "-", *args],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "-1743" in err or "Not authorized" in err:
            raise PermissionDenied(
                "macOS denied automation access to Music. Grant it under System "
                "Settings > Privacy & Security > Automation, then restart."
            )
        raise RuntimeError(err or f"osascript exited {proc.returncode}")
    return proc.stdout.strip()


def poll() -> Track:
    out = _osascript(_STATE)
    if SEP not in out:
        return Track(out or "stopped", "", "", "", "", False)
    state, tid, name, artist, album, nart = (out.split(SEP) + [""] * 6)[:6]
    return Track(state, tid, name, artist, album, (nart or "0").strip() not in ("", "0"))


def artwork() -> bytes | None:
    """Raw bytes of the current track's cover art, or None."""
    fd, path = tempfile.mkstemp(prefix="kitty-vinyl-art-")
    os.close(fd)
    try:
        if _osascript(_ARTWORK, path) != "ok":
            return None
        with open(path, "rb") as fh:
            data = fh.read()
        return data or None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
