# pixel-effect

Turn a video or photo into a glitched, live-stream-style piece: a tracked head (or the whole
person) dissolves into blocks and re-forms, with iridescence, glyph overlays, ghost trails,
halos, signal damage and an animated HUD with a scrolling chat.

Everything is one Python script, `pixel_effect.py`. Presets are small `.bat` files that set
its options, and `render.bat` runs a preset.

## Setup

Python 3.10+ and:

```bash
pip install -r requirements.txt
```

With Anaconda, make an environment for it:

```bash
conda create -n pixel-effect python=3.12
conda activate pixel-effect
pip install -r requirements.txt
```

Optional, to put the original audio back: [ffmpeg](https://ffmpeg.org) (`conda install -c conda-forge ffmpeg`).

## Quick start (Windows)

```bat
render.bat angelcore input.mp4 preview.mp4 draft
render.bat angelcore input.mp4 final.mp4
```

`draft` renders a quick, quarter-size preview of the first 4 seconds. Run `render.bat` with no
arguments to list the presets.

## Presets

| Preset | Look |
| --- | --- |
| `angelcore` | Angelcore HUD, body on black, Enochian glyphs, pink trails, crown halo |
| `broken` | Broken transmission: angelcore HUD, mixed glyphs, hard RGB tear across everything |
| `dream` | No HUD, halftone body on deep purple, oil-slick iridescence, long soft trails, ring halo |
| `ghost` | No HUD, black void, long body trails, iridescent oil colors |
| `hudless` | Head only, quiet iridescence, small crown, all glyphs, no HUD |
| `minimal` | Head only, quiet iridescence, small crown, no chat. Good first look at new footage |
| `terminal` | Hacker terminal: orange and green panels, hex and code glyphs, blurred background |

### Make your own

```bat
copy presets_template.bat presets_mylook.bat
```

`presets_template.bat` lists **every** option, grouped (`LOOK`, `HUD`, `FONTS`, `CHAT`, `PIXEL`,
`IRID`, `GLYPHS`, `ECHO`, `HALO`, `DAMAGE`, `SIGIL`, `BODY`, `BG`, `OVERLAY`, `CROP`, `FACE`,
`FULL`, `DRAFT`), with comments showing allowed values and on/off switches. Edit the values,
then `render.bat mylook input.mp4 out.mp4`.

Rules for preset files:

- No spaces around `=`: `set HUD=--no-hud`, never `set HUD = --no-hud`.
- `""` means "not set": the script uses its own default.
- If a flag appears twice, the last one wins.
- `%~dp0` means "this folder", so `--comments "%~dp0comments.txt"` works from anywhere.

## Control panel (GUI)

```bash
pip install -r requirements-gui.txt
python gui.py
```

Or double-click `gui.bat` on Windows. It opens in your browser and runs only on your computer.

- Every option is a control, grouped like `presets_template.bat`, with its help text.
- The preview renders one frame of your clip at the time you pick, through the same code as a real
  render, usually in under a second. The first preview of a clip that needs a background estimate
  takes longer (it reads the whole clip once).
- **Load preset** / **Save as preset** read and write the same `presets_*.bat` files `render.bat` uses.
- **Equivalent command** shows the settings as a command line you can copy.
- **Draft render** and **Full render** run the real script and show the result.

A single-frame preview can't show effects that build up over time (echo trails, mask smoothing);
use a draft render for those.

## Color looks, tone and wash

- `--look` picks a ready-made color look: `default`, `eve`, `rose`, `gold`, `teal`, `ember`, `vapor`, `noir`.
- `--tone-shadows`, `--tone-mids`, `--tone-highlights` (colors, dark to light) build your own look;
  `--tone-strength` sets how much it replaces the original colors.
- `--hue-shift`, `--saturation`, `--bloom`, `--chroma-shift` fine-tune any look.
- `--wash "#ff3fa4" --wash-opacity 0.3 --wash-blend softlight` lays a flat color over everything.

In the GUI every color option has a color wheel.

## Fonts

Font options take an installed font's **name** (`--hud-font "Segoe UI Bold"`), a font file name, or a
path. Fonts you don't want to install can go in `fonts\`. In the GUI, font options are searchable
dropdowns of everything installed.

## Video overlays

`--overlay` also takes a video (MP4, MOV...). It plays in sync with your clip and loops if shorter.
Videos have no transparency, so pick how it mixes in:

- `--overlay-blend screen` or `add`: black becomes see-through (light leaks, dust, sparks, particles).
- `--overlay-key "#00ff00"`: one color becomes see-through (green screen); `--overlay-key-tolerance` tunes it.
- `--overlay-start 2.5` starts the overlay video 2.5 s in.

## Running the script directly (any OS)

```bash
python pixel_effect.py input.mp4 out.mp4 --style eve --theme angelcore --max-seconds 5
python pixel_effect.py photo.jpg out.png --style eve --progress 0.8     # still image
python pixel_effect.py --help                                           # every option
```

Put the original audio back:

```bash
ffmpeg -i out.mp4 -i input.mp4 -map 0:v:0 -map 1:a:0? -c:v libx264 -crf 18 -c:a aac -shortest final.mp4
```

## Files

| File | What it is |
| --- | --- |
| `pixel_effect.py` | The effect script |
| `render.bat` | Runs a preset: `render.bat PRESET INPUT OUTPUT [draft]` |
| `gui.py`, `gui.bat` | The browser control panel (needs `requirements-gui.txt`) |
| `presets_*.bat` | Presets; `presets_template.bat` documents every option |
| `comments.txt` | Chat lines for the HUD, one `username: message` per line |
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (`--detector yunet`) |
| `human_segmentation_pphumanseg_2023mar.onnx` | Person segmentation (`--cutout neural` / `hybrid`) |
| `haarcascade_frontalface_default.xml` | OpenCV's Haar face detector (default detector) |
| `Enochian.otf`, `EnochianPlain.ttf` | Font for `--glyphs enochian` |

See `THIRD_PARTY.md` for where the model and font files come from.

## License

The code, presets and docs are under the [MIT License](LICENSE). The model and font files keep
their own licenses; see `THIRD_PARTY.md`.

## Tips

- Start every new clip with a `draft` render; tune, then render full quality.
- No head found? Try `--detector yunet`, lower `--yunet-score`, or `--debug` to see detections.
- `--target body` and `--background` with `--cutout bgsub` need a fixed camera; use `--cutout hybrid` otherwise.
