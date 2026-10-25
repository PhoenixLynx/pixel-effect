"""
gui.py: a browser control panel for pixel_effect.py.

    python gui.py            (opens http://127.0.0.1:7860 in your browser; runs only on this computer)

Every option of pixel_effect.py becomes a control, grouped like presets_template.bat. The preview renders one
frame of your clip at the moment you pick, through the same code as a real render. Presets load and save in the
template layout, so render.bat keeps working with them. Needs:  pip install gradio
"""
import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import gradio as gr

import pixel_effect as pe

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pixel_effect.py"
TEMPLATE = HERE / "presets_template.bat"
SET_RE = re.compile(r"^\s*set\s+([A-Za-z_]+)=(.*)$", re.I)
GROUP_ORDER = ["LOOK", "HUD", "FONTS", "CHAT", "PIXEL", "IRID", "GLYPHS", "ECHO", "HALO", "DAMAGE", "SIGIL",
               "BODY", "BG", "OVERLAY", "CROP", "FACE", "FULL"]
GROUP_TITLES = {"LOOK": "Look", "HUD": "HUD", "FONTS": "Fonts", "CHAT": "Chat", "PIXEL": "Pixel dissolve",
                "IRID": "Iridescence", "GLYPHS": "Glyphs", "ECHO": "Echo trails", "HALO": "Halo", "DAMAGE": "Signal damage",
                "SIGIL": "Sigils", "BODY": "Body and cutout", "BG": "Background", "OVERLAY": "Overlay art", "CROP": "Crop",
                "FACE": "Face finding", "FULL": "Full-render size", "OTHER": "Other"}
DRAFT_DEFAULT = "--scale 0.25 --max-seconds 4"
# options that take a file path -> (file types for the Browse dialog, mode: open one / open several / save)
_FONTS = [("Fonts", "*.ttf *.otf *.ttc")]
_IMAGES = [("Images", "*.png *.jpg *.jpeg *.webp *.bmp")]
PATH_FIELDS = {"--comments": ([("Text", "*.txt")], "open"), "--hud-font": (_FONTS, "open"), "--hud-title-font": (_FONTS, "open"),
               "--emoji-font": (_FONTS, "open"), "--enochian-font": (_FONTS, "open"), "--sigil-font": (_FONTS, "open"),
               "--sigil-image": ([("PNG", "*.png")], "many"), "--overlay": ([("PNG", "*.png")], "open"),
               "--bg-image": (_IMAGES, "open"), "--bg-replace": (_IMAGES, "open"), "--seg-model": ([("ONNX model", "*.onnx")], "open"),
               "--yunet-model": ([("ONNX model", "*.onnx")], "open"), "--mask-video": ([("MP4 video", "*.mp4")], "save"),
               "INPUT": ([("Video or image", "*.mp4 *.mov *.avi *.mkv *.webm *.png *.jpg *.jpeg *.webp *.bmp")], "open"),
               "OUTPUT": ([("MP4 video", "*.mp4"), ("PNG image", "*.png")], "save")}

_DIALOG = """
import sys, json, tkinter as tk, tkinter.filedialog as fd
types, mode, start = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
kw = dict(filetypes=[tuple(t) for t in types] + [("All files", "*.*")], initialdir=start or None, parent=root)
if mode == "many":
    print(",".join(fd.askopenfilenames(**kw)))
elif mode == "save":
    print(fd.asksaveasfilename(defaultextension=types[0][1].split()[0][1:], **kw))
else:
    print(fd.askopenfilename(**kw))
"""


def browse(key, current):
    """Open a Windows file dialog (in its own process, so it can't freeze the app) and return the chosen path."""
    types, mode = PATH_FIELDS[key]
    first = (current or "").split(",")[0].strip()
    start = os.path.dirname(first) if first and os.path.isdir(os.path.dirname(first)) else os.getcwd()
    if key in FONT_FLAGS and not (first and os.path.isfile(first)) and (HERE / "fonts").is_dir():
        start = str(HERE / "fonts")                    # Windows hides font files in C:\Windows\Fonts; fonts\ is easier
    out = subprocess.run([sys.executable, "-c", _DIALOG, json.dumps(types), mode, start], capture_output=True, text=True).stdout.strip()
    if not out:                                      # dialog cancelled: keep what was there
        return gr.skip()
    return ",".join(os.path.normpath(p) for p in out.split(",")) if mode == "many" else os.path.normpath(out)
RENDER_LOCK = threading.Lock()      # pixel_effect keeps its look in module globals: one render at a time
PREVIEW_GATE = threading.Lock()     # previews wait here in turn; stale ones are skipped (see do_preview)
TICKET_LOCK = threading.Lock()
TICKET = {"n": 0}


# ---------------------------------------------------------------- options, read from the script itself

def split_flags(value):
    """'--a 1 --b --c "x y"' -> [('--a', '1'), ('--b', None), ('--c', 'x y')], quotes removed."""
    out, toks = [], shlex.split(value, posix=False)
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out.append((toks[i], toks[i + 1].strip('"')))
            i += 2
        else:
            out.append((toks[i], None))
            i += 1
    return out


def template_groups():
    """flag -> group name, from the set lines and 'switches:' comments of presets_template.bat."""
    groups, pending, mentioned = {}, [], []
    for line in TEMPLATE.read_text().splitlines():
        if line.strip().lower().startswith("rem") and "switches:" in line:
            pending += [t for t in line.split("switches:")[1].split() if t.startswith("--")]
        elif line.strip().lower().startswith("rem"):          # options only described in comments (e.g. --hue-shift)
            mentioned += re.findall(r"--[a-z][a-z0-9-]*", line)
        m = SET_RE.match(line)
        if m:
            for flag in mentioned:
                groups.setdefault(flag, m.group(1).upper())
            mentioned = []
        if m and m.group(1).upper() == "DRAFT":         # DRAFT repeats FULL's flags; the render buttons handle it
            pending = []
        elif m:
            for flag, _ in split_flags(m.group(2)):
                groups[flag] = m.group(1).upper()
            for flag in pending:
                groups[flag] = m.group(1).upper()
            pending = []
    groups["--hud"] = groups["--no-hud"] = "HUD"
    return groups


class Spec:
    """One control: which flag it sets and how."""
    def __init__(self, action, group):
        self.flag, self.action, self.group = action.option_strings[0], action, group
        self.default = action.default
        self.help = (action.help or "").replace("%%", "%")
        if isinstance(action, argparse.BooleanOptionalAction):
            self.kind, self.default = "tri", "auto"
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            self.kind, self.default = "switch", False
        elif action.choices:
            self.kind = "choice"
        elif action.type in (int, float):
            self.kind = "number"
        else:
            self.kind = "text"

    def to_argv(self, v):
        if self.kind == "switch":
            return [self.flag] if v else []
        if self.kind == "tri":
            return {"on": ["--hud"], "off": ["--no-hud"]}.get(v, [])
        if v is None or v == self.default:
            return []
        if self.kind == "number":
            v = int(round(v)) if self.action.type is int else float(v)
        return [self.flag, str(v)]


def build_specs():
    groups = template_groups()
    specs = []
    for a in pe.build_parser()._actions:
        if not a.option_strings or a.option_strings[0] in ("-h", "--sigils"):    # --sigils only undoes --no-sigils
            continue
        specs.append(Spec(a, groups.get(a.option_strings[0], "OTHER")))
    return specs


SPECS = build_specs()
BY_FLAG = {s.flag: s for s in SPECS}
BY_FLAG["--no-hud"] = BY_FLAG["--hud"]


def argv_from(values):
    return [x for spec, v in zip(SPECS, values) for x in spec.to_argv(v)]


def command_line(src, out, argv):
    """The same settings as a command you could paste into a terminal (Windows-style quotes)."""
    return " ".join(f'"{x}"' if re.search(r'[\s#&|<>^]', x) or x == "" else x for x in ["python", "pixel_effect.py", src or "INPUT", out] + argv)


# ---------------------------------------------------------------- preview

_BG_CACHE = {}


def background_for(src, crop, thr):
    """The fixed-camera background estimate, cached per clip and settings (it reads the whole clip once)."""
    key = (src, os.path.getmtime(src), crop, thr)
    fresh = key not in _BG_CACHE
    if fresh:
        _BG_CACHE.clear()
        _BG_CACHE[key] = pe.build_background(src, None, "", crop, thr)
    return _BG_CACHE[key], fresh


def read_frame(src, t):
    """(frame, is_video, duration in seconds)."""
    if os.path.splitext(src)[1].lower() in pe.IMAGE_EXTS:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"Could not read {src}")
        return img, False, 0.0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / fps
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, min(t, dur - 1 / fps)) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame at {t:.2f} s")
    return frame, True, dur


def render_preview(src, t, width, values):
    if not src:
        return None, "Pick an input video or image first."
    if not os.path.exists(src):
        return None, f"File not found: `{src}`"
    started = time.perf_counter()
    try:
        with RENDER_LOCK:
            frame, is_video, _ = read_frame(src, t)
            args = pe.build_parser().parse_args([src, "preview.png"] + argv_from(values))
            if is_video:                                 # show this moment of the animation, like the video would
                args.hud_time = t
                args.progress = 0.5 - 0.5 * math.cos(2 * math.pi * t / args.period)
            ctx = pe.setup(args)
            if frame.shape[1] > width:
                frame = cv2.resize(frame, (int(width), int(frame.shape[0] * width / frame.shape[1]) // 2 * 2), interpolation=cv2.INTER_AREA)
            bg, fresh_bg = None, False
            if ctx.need_mask and ctx.need_bg and is_video and not args.bg_image:
                bg, fresh_bg = background_for(src, ctx.crop, args.bg_threshold)
            out = pe.render_still(frame, args, ctx, bg=bg)
    except SystemExit as e:
        return None, f"**Can't preview:** {e}"
    except Exception as e:                               # show the problem instead of crashing the app
        return None, f"**Error:** `{type(e).__name__}: {e}`"
    note = " (single frame: echo trails and smoothing only show in a render)" if is_video and (args.echo > 0 or args.mask_smooth > 0 and ctx.need_mask) else ""
    if fresh_bg:
        note += ". This included estimating the clip's background once; later previews of this clip are faster"
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), f"Rendered in {time.perf_counter() - started:.2f} s{note}"


def clip_length(src):
    try:
        _, is_video, dur = read_frame(src, 0)
        return gr.update(maximum=max(0.1, round(dur, 2)), interactive=is_video)
    except SystemExit:
        return gr.update()


# ---------------------------------------------------------------- presets

def preset_names():
    return sorted(p.stem[len("presets_"):] for p in HERE.glob("presets_*.bat"))


def load_preset(name):
    """Control values (and the preset's DRAFT line) from presets_NAME.bat, using batch rules: a later set wins."""
    path = HERE / f"presets_{name}.bat"
    env = {}
    for line in path.read_text().splitlines():
        m = SET_RE.match(line)
        if m:
            env[m.group(1).upper()] = m.group(2)
    values = {s.flag: s.default for s in SPECS}
    unknown = []
    for group in GROUP_ORDER:
        for flag, value in split_flags(env.get(group, "")):
            spec = BY_FLAG.get(flag)
            if flag == "--sigils":
                values["--no-sigils"] = False
            elif spec is None:
                unknown.append(flag)
            elif spec.kind == "switch":
                values[flag] = True
            elif spec.kind == "tri":
                values["--hud"] = "on" if flag == "--hud" else "off"
            elif value is not None:
                value = value.replace("%~dp0", str(HERE) + os.sep)
                values[flag] = (int(float(value)) if spec.action.type is int else float(value)) if spec.kind == "number" else value
    msg = f"Loaded **{name}**." + (f" Ignored unknown flags: {', '.join(unknown)}" if unknown else "")
    return [values[s.flag] for s in SPECS] + [env.get("DRAFT", DRAFT_DEFAULT), msg]


def batch_value(v):
    s = str(v)
    if s.startswith(str(HERE) + os.sep):                # keep presets portable: paths next to the tool become %~dp0
        s = "%~dp0" + s[len(str(HERE)) + 1:]
    return f'"{s}"' if s == "" or re.search(r'[\s#"&|<>^]', s) else s


def group_line(group, current):
    parts, switches = [], []
    for spec in SPECS:
        if spec.group != group:
            continue
        v = current[spec.flag]
        if spec.kind == "switch":
            switches += [spec.flag] if v else []
        elif spec.kind == "tri":
            switches += spec.to_argv(v)
        elif v is None:                                  # optional numbers left blank: not written (the look decides)
            continue
        else:
            if spec.kind == "number" and spec.action.type is int:
                v = int(round(v))
            parts.append(f"{spec.flag} {batch_value(v)}")
    return " ".join(parts + switches)


def save_preset(name, description, overwrite, draft, *values):
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "template":
        return gr.update(), "Preset names use letters, numbers, - and _ (and can't be 'template')."
    path = HERE / f"presets_{name}.bat"
    if path.exists() and not overwrite:
        return gr.update(), f"`{path.name}` already exists. Tick **Overwrite** to replace it."
    current = {s.flag: v for s, v in zip(SPECS, values)}
    lines = TEMPLATE.read_text().splitlines()
    first_set = next(i for i, l in enumerate(lines) if SET_RE.match(l))
    other = group_line("OTHER", current)
    out = ["@echo off", "rem " + "=" * 70, f"rem  Preset: {name}. {description or 'Saved from gui.py.'}",
           "rem  Same layout as presets_template.bat: every option is listed, with",
           "rem  this preset's values. Change a value or paste in a listed switch.",
           f"rem      render.bat {name} input.mp4 output.mp4 [draft]", "rem " + "=" * 70, ""]
    for line in lines[first_set:]:
        m = SET_RE.match(line)
        if not m:
            out.append(f"echo preset {name} loaded." if line.strip().lower().startswith("echo ") else line)
            continue
        g = m.group(1).upper()
        if g == "SCRIPT":
            out.append("set SCRIPT=pixel_effect.py")
        elif g == "DRAFT":
            out.append(f"set DRAFT={draft or DRAFT_DEFAULT}")
        else:
            out.append(f"set {m.group(1)}={group_line(g, current)}" + (f" {other}" if g == "LOOK" and other else ""))
    path.write_text("\r\n".join(out) + "\r\n", newline="")
    return gr.update(choices=preset_names(), value=name), f"Saved `{path.name}`. Render it with `render.bat {name} input.mp4 out.mp4`."


# ---------------------------------------------------------------- full renders

def run_render(src, out, draft_line, draft, *values):
    if not src or not os.path.exists(src):
        yield "Pick an input file first.", None
        return
    out = (out or "").strip() or "renders/out.mp4"
    if os.path.splitext(src)[1].lower() in pe.IMAGE_EXTS and not out.lower().endswith((".png", ".jpg")):
        out = os.path.splitext(out)[0] + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    argv = argv_from(values) + (shlex.split(draft_line or DRAFT_DEFAULT) if draft else [])   # later flags win
    proc = subprocess.Popen([sys.executable, "-u", str(SCRIPT), src, out] + argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    log, line = [], ""
    for ch in iter(lambda: proc.stdout.read(1), ""):
        if ch in "\r\n":
            if line.strip():
                if line.startswith("frame ") and log and log[-1].startswith("frame "):
                    log[-1] = line.strip()                # progress lines replace each other
                else:
                    log.append(line.strip())
                yield "\n".join(log[-14:]), None
            line = ""
        else:
            line += ch
    proc.wait()
    if line.strip():
        log.append(line.strip())
    ok = proc.returncode == 0 and os.path.exists(out)
    log.append(f"Finished: {os.path.abspath(out)}" if ok else f"Render failed (exit code {proc.returncode}).")
    yield "\n".join(log[-14:]), (os.path.abspath(out) if ok and out.lower().endswith(".mp4") else None)


# ---------------------------------------------------------------- the page

FONT_FLAGS = {"--hud-font", "--hud-title-font", "--emoji-font", "--sigil-font", "--enochian-font"}
COLOR_FLAGS = {"--accent", "--hud-color", "--echo-color", "--bg-color", "--tone-shadows", "--tone-mids", "--tone-highlights",
               "--wash", "--overlay-key"}
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def to_hex(value):
    """The color picker can report '#rrggbb', '#rrggbbaa' or 'rgba(r, g, b, a)'; presets store '#rrggbb'."""
    v = (value or "").strip()
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", v)
    if m:
        return "#" + "".join(f"{min(255, round(float(x))):02x}" for x in m.groups())
    return v[:7].lower() if re.match(r"^#[0-9a-fA-F]{6}", v) else v


def font_choices():
    """Installed fonts by name for the font dropdowns; typing filters the list."""
    return [("(none)", "")] + [(n, n) for n in sorted(pe.installed_fonts(), key=str.lower)]


def control_for(spec):
    label, info = spec.flag, spec.help or None
    if spec.flag in FONT_FLAGS:
        return gr.Dropdown(font_choices(), value=spec.default or "", label=label, allow_custom_value=True, filterable=True,
                           info=(info or "") + " Type to search installed fonts, or Browse for a font file.")
    if spec.kind == "switch":
        return gr.Checkbox(value=False, label=label, info=info)
    if spec.kind == "tri":
        return gr.Radio(["auto", "on", "off"], value="auto", label="--hud / --no-hud", info=info)
    if spec.kind == "choice":
        return gr.Dropdown(list(spec.action.choices), value=spec.default, label=label, info=info)
    if spec.kind == "number":
        return gr.Number(value=spec.default, label=label, info=info, precision=0 if spec.action.type is int else None)
    return gr.Textbox(value=spec.default or "", label=label, info=info, placeholder="(not set)")


def build_app():
    with gr.Blocks(title="pixel-effect") as app:
        gr.Markdown("# pixel-effect\nSet options on the left; the preview on the right renders one frame of your clip "
                    "through the same code as a real render.")
        draft_state = gr.State(DRAFT_DEFAULT)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    src = gr.Textbox(label="Input video or image", placeholder=r"C:\path\to\clip.mp4", scale=4)
                    src_browse = gr.Button("Browse…", scale=0, min_width=100)
                    upload = gr.File(label="or drop a file", type="filepath", scale=2, height=90)
                with gr.Row():
                    preset = gr.Dropdown(preset_names(), label="Preset", scale=3)
                    load_btn = gr.Button("Load preset", scale=1)
                controls, browse_buttons, pickers = {}, [], []
                groups = [g for g in GROUP_ORDER + ["OTHER"] if any(s.group == g for s in SPECS)]
                for g in groups:
                    with gr.Accordion(GROUP_TITLES.get(g, g), open=g in ("LOOK", "HUD")):
                        for spec in (s for s in SPECS if s.group == g):
                            if spec.flag in PATH_FIELDS:          # file paths get a Browse button beside the box
                                with gr.Row():
                                    controls[spec.flag] = control_for(spec)
                                    browse_buttons.append((spec.flag, gr.Button("Browse…", scale=0, min_width=100)))
                            elif spec.flag in COLOR_FLAGS:        # the hex box holds the value; the wheel fills it in
                                with gr.Row():
                                    controls[spec.flag] = control_for(spec)
                                    pickers.append((spec.flag, gr.ColorPicker(value=spec.default if HEX_RE.match(spec.default or "") else "#000000",
                                                                              label="pick", scale=0, min_width=80)))
                            else:
                                controls[spec.flag] = control_for(spec)
                with gr.Accordion("Save as preset", open=False):
                    save_name = gr.Textbox(label="Preset name", placeholder="mylook  (saves presets_mylook.bat)")
                    save_desc = gr.Textbox(label="One-line description")
                    overwrite = gr.Checkbox(label="Overwrite if it exists")
                    save_btn = gr.Button("Save preset")
            with gr.Column(scale=6):
                image = gr.Image(label="Preview", type="numpy", interactive=False)
                with gr.Row():
                    t_slider = gr.Slider(0, 10, value=1.0, step=0.04, label="Time in the clip (s)", scale=3)
                    width = gr.Slider(320, 1920, value=720, step=40, label="Preview width (px)", scale=2)
                with gr.Row():
                    live = gr.Checkbox(value=True, label="Live preview (re-render on every change)")
                    refresh = gr.Button("Update preview", variant="primary")
                status = gr.Markdown()
                cmd = gr.Textbox(label="Equivalent command", lines=3, interactive=False, buttons=["copy"])
                with gr.Accordion("Render", open=True):
                    with gr.Row():
                        out_path = gr.Textbox(value="renders/out.mp4", label="Output file", scale=4)
                        out_browse = gr.Button("Browse…", scale=0, min_width=100)
                    with gr.Row():
                        draft_btn = gr.Button("Draft render (quick)")
                        full_btn = gr.Button("Full render", variant="primary")
                    log = gr.Textbox(label="Render log", lines=8, interactive=False)
                    video = gr.Video(label="Result", interactive=False)

        values = [controls[s.flag] for s in SPECS]

        def do_preview(src_v, t, w, *vals):
            """Latest wins: every request takes a ticket; when its turn comes, only the newest ticket renders.
            Loading a preset changes ~80 controls at once, and this turns that burst into one preview."""
            with TICKET_LOCK:
                TICKET["n"] += 1
                mine = TICKET["n"]
            with PREVIEW_GATE:
                if mine != TICKET["n"]:
                    return gr.skip(), gr.skip(), gr.skip()
                img, msg = render_preview(src_v, t, w, vals)
            return img, msg, command_line(src_v, "OUTPUT", argv_from(vals))

        def maybe_preview(is_live, src_v, t, w, *vals):
            if not is_live:
                return gr.skip(), gr.skip(), command_line(src_v, "OUTPUT", argv_from(vals))
            return do_preview(src_v, t, w, *vals)

        # queue=False: requests arrive at once (each in its own thread), so the ticket check above can drop stale ones
        preview_io = dict(outputs=[image, status, cmd], queue=False)
        refresh.click(do_preview, [src, t_slider, width] + values, **preview_io)
        gr.on([c.change for c in values] + [t_slider.release, width.release], maybe_preview,
              [live, src, t_slider, width] + values, **preview_io)
        upload.change(lambda f: f or gr.skip(), upload, src)
        for flag, btn in browse_buttons:
            btn.click(lambda cur, f=flag: browse(f, cur), controls[flag], controls[flag])
        for flag, picker in pickers:
            box = controls[flag]
            picker.input(to_hex, picker, box)                                  # wheel -> hex box (and so the preview)
            box.change(lambda v: to_hex(v) if HEX_RE.match(to_hex(v)) else gr.skip(), box, picker)   # typed/loaded hex -> wheel
        src_browse.click(lambda cur: browse("INPUT", cur), src, src)
        out_browse.click(lambda cur: browse("OUTPUT", cur), out_path, out_path)
        src.change(clip_length, src, t_slider).then(maybe_preview, [live, src, t_slider, width] + values, **preview_io)
        load_btn.click(load_preset, preset, values + [draft_state, status])
        save_btn.click(save_preset, [save_name, save_desc, overwrite, draft_state] + values, [preset, status])
        draft_btn.click(run_render, [src, out_path, draft_state, gr.State(True)] + values, [log, video], concurrency_limit=1)
        full_btn.click(run_render, [src, out_path, draft_state, gr.State(False)] + values, [log, video], concurrency_limit=1)
    return app


if __name__ == "__main__":
    build_app().launch(inbrowser="--no-browser" not in sys.argv, server_name="127.0.0.1",
                       allowed_paths=[os.getcwd(), str(HERE)])
