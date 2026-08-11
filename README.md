# kitty-vinyl

A vinyl record spinning behind your kitty terminal, lying on a wooden desk,
wearing the cover art of whatever Apple Music is playing.

```
./vinyl setup     # one-time: adds remote control to kitty.conf
                  # then fully quit kitty (cmd+Q) and reopen it
./vinyl run       # start spinning
```

`./vinyl status` shows what it can see; `./vinyl preview` writes a still PNG so
you can tune the look without kitty; `./vinyl uninstall` puts everything back.

## How it works

kitty paints a window in this order:

```
background colour  ->  background image  ->  window logo  ->  your text
```

That's two layers underneath the text, and the record splits across them:

- **What stays still** — wood grain, whatever light falls across it, then the
  grooves, band gaps, rim and specular sheen — is composited into a single
  **background image** and uploaded exactly once. kitty has only one background
  image layer, so it all has to travel together.
- **What turns** — the cover art label, and the wipe marks on the vinyl around
  it — goes in the **window logo** layer, one disc-sized frame per rotation
  step, streamed as the record spins.

Splitting it that way is not a shortcut, it is the physics: a real LP's grooves,
rim and highlight are rotationally symmetric, so spinning them would change
nothing on screen, while the lamp's reflection is fixed to the lamp and must not
turn at all. What you actually see move on a turntable is the label and the
scuffs sweeping through the light — so those are exactly what gets streamed. The
frames carry only those, transparent everywhere else, which keeps them at
~190 KB even though they are disc-sized.

Everything goes over kitty's remote-control socket as pre-encoded bytes, so the
animation loop does one `sendall` per frame — no image work at frame time.

### Why not the obvious approaches

- **`background_image` with a list of frames.** kitty 0.47 can hold many
  background images in VRAM and switch by index almost for free
  (~0.1 ms/frame). But the list is resolved at kitty *startup*: `load-config`
  does not re-read it, a new OS window doesn't pick up changes, and once a frame
  is loaded kitty never re-reads that file even if its contents change. So the
  images can't follow the track. Each distinct frame also pins ~3 MB of VRAM
  permanently — 48 frames measured at +152 MB.
- **Streaming the whole record, grooves and all.** Works, but a fully painted
  ~900px disc is ~1.3 MB every frame instead of ~190 KB, and rotating it turns
  the specular highlight with it — which reads as a spinning lamp rather than a
  spinning record.
- **The graphics protocol with negative z-index.** Draws below text, but it's
  anchored to cells (so it drifts as the screen scrolls), it's hidden when a
  full-screen app like vim takes the alternate screen, and writing escape codes
  into a tty someone is actively using invites corruption.

## Cost

Roughly **5–20% of one CPU core** while a track is playing, depending on frame
rate, disc size, and whether the window is actually visible (an occluded window
is much cheaper — kitty skips the redraw). Memory is flat; nothing accumulates.

When playback is **paused it drops to ~0** — the frame stops changing, and kitty
recognises a repeated upload and skips the work entirely.

Turn it down with fewer frames or a smaller disc:

```
./vinyl run --frames 50          # 10 fps instead of 15
./vinyl run --size 620           # smaller record
```

## Options

| flag | default | |
|---|---|---|
| `--size` | 78% of the window | record diameter in device pixels |
| `--opacity` | `0.62` desk, `0.47` bare | how strongly the background shows through |
| `--label-opacity` | `0.85` | opacity of the cover art |
| `--blur` | `3.0` | device pixels of blur over the desk and the record, so text reads more easily on top; the cover art is never blurred. `0` is a sharp record |
| `--frames` | `75` | rotation steps per revolution; more is smoother but costs a longer render on every track change |
| `--wipe` | `1.0` | how strongly the wipe marks catch the light as the record turns; `0` is a pristine pressing |
| `--rpm` | `12.0` | turntable speed; fps = frames x rpm / 60. Well under the `33.3333` of a real LP, which at this size on screen reads as frantic |
| `--no-desk` | | drop the wood, just the record as before |
| `--light` | `0` | bars of light across the desk: `0` evenly lit, `1` full sun through a blind |
| `--desk-brightness` | `1.0` | scales how brightly the desk is lit; below `1` is darker wood and more contrast under the text |
| `--desk-seed` | `7` | reshuffles the wood grain |
| `--window` | measured | window size in device pixels, `WIDTHxHEIGHT` |
| `--match` | `all` | which kitty windows get the label |
| `--poll` | `2.0` | seconds between Music.app checks |

Size and window are auto-detected (via `CSI 14 t`) when running attached to a
terminal. Under launchd there's no tty, so pass `--size` and `--window`
explicitly — otherwise the desk is drawn 16:10 and left to kitty to stretch.

The desk is drawn at the window's own resolution so the grooves stay as fine as
they are on their own, which makes it a ~2 MB upload. That happens once, at
startup; nothing about it is per-frame.

## Keeping the text readable

Two knobs, and they work on the background layer only — the cover art is carried
by the logo layer at `--label-opacity` and neither of them touches it, so the
record can be pushed as far back as you like without the album going with it.

`--blur` is the one that buys the most. The grooves are a 3px ripple, which is
the finest thing on screen and sits directly under the text; softening it costs
the record very little at a glance. `--opacity` then sets how far the whole
thing sinks into the terminal background.

```
./vinyl run --blur 6 --opacity 0.5      # further back
./vinyl run --blur 0 --opacity 0.72     # the old sharp record
```

This is deliberately not kitty's `background_blur`, which blurs the desktop
wallpaper behind the window rather than anything this draws.

## Notes

- **Splits.** The background image spans the whole OS window while the logo is
  per split, so with splits open you get one static disc and one turning layer
  in each pane. For a single unsplit window they line up exactly. `--match` can
  narrow which panes get the turning layer.
- **Resizing.** The desk is drawn to the window it measured at startup and
  scaled by kitty from there, so a resize moves the static disc out from under
  the turning layer by whatever the scale factor is. Restart the daemon to
  re-fit it.
- **Automation permission.** The first run makes macOS ask whether the terminal
  may control Music. It has to be allowed, or the track can't be read.
- **Music must already be running** — the daemon deliberately never launches it.
- Stopping with `ctrl-c` clears both layers.

## Running it in the background

```
nohup ./vinyl run >/tmp/vinyl.log 2>&1 &
```

Or load `com.local.kitty-vinyl.plist` (edit the `--size` in it first) to start
it at login:

```
cp com.local.kitty-vinyl.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.local.kitty-vinyl.plist
```

## Files

| | |
|---|---|
| `vinyl.py` | daemon, CLI, kitty.conf setup |
| `render.py` | draws the desk, the still disc and the frames that turn |
| `nowplaying.py` | reads the current track out of Music via AppleScript |
| `kitty_rc.py` | kitty remote-control client and image-upload encoder |
