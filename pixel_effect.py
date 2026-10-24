"""
pixel_effect.py: pixelate a tracked head so it dissolves into blocks and re-forms, with optional
binary-digit overlay, horizontal pixel smear, particle burst, and a tunable color look.

Setup:
    pip install opencv-python numpy

Video:
    python pixel_effect.py input.mp4 out.mp4 --max-seconds 5            # blocks + default look
    python pixel_effect.py input.mp4 out.mp4 --style eve                # blocks + digits + smear + particles + ice-blue look
    python pixel_effect.py input.mp4 out.mp4 --digits --particles       # mix and match
    python pixel_effect.py input.mp4 out.mp4 --hud --hud-title "MY TITLE"     # live-stream overlay on any clip
    python pixel_effect.py input.mp4 out.mp4 --style eve --comments mine.txt  # own comment lines ("user: message")
Still image (renders one moment of the animation; --progress 0 = clean, 1 = fully broken up):
    python pixel_effect.py photo.jpg out.png --style eve --progress 0.8

Put the original audio back (needs ffmpeg; conda install -c conda-forge ffmpeg):
    ffmpeg -i out.mp4 -i input.mp4 -map 0:v:0 -map 1:a:0? -c:v libx264 -crf 18 -c:a aac -shortest final.mp4

Notes:
  * OpenCV 5 moved the Haar detector to contrib and no longer ships its model file. Use OpenCV 4
    (pip install "opencv-python<5") or put haarcascade_frontalface_default.xml next to this script.
  * Head tracking is OpenCV's built-in Haar face detector: no downloads, but it only finds
    fairly frontal faces and can be jittery. Use --detector yunet (neural) for tilted or cropped faces, and --region / --skin-min / --rank to choose which face counts.
  * Tune the color looks in LOOK_DEFAULT / LOOK_EVE below. LOOK_EVE's gradient map was built from
    measurements of a reference frame (ice-blue, desaturated, deep black background).
"""
import argparse
import copy
import math
import re
import sys
import os
import time
import unicodedata
from types import SimpleNamespace

import cv2
import numpy as np

# ---- Color looks: tweak these, re-run on a short clip with --max-seconds 5 ----------
LOOK_DEFAULT = {
    "hue_shift": 0,        # degrees, -90..90 (rotates all colors)
    "saturation": 1.3,     # 1.0 = unchanged
    "chroma_shift": 3,     # pixels; splits red and blue channels (chromatic aberration)
    "bloom": 0.35,         # 0 = off; glow around bright areas
    "scanlines": 0.12,     # 0 = off; darkens every third row
    "posterize": 0,        # 0 = off; e.g. 6 limits each channel to 6 levels
    "gradient_stops": None,
    "gradient_strength": 0.0,
}

LOOK_EVE = {
    "hue_shift": 0,
    "saturation": 1.0,
    "chroma_shift": 2,
    "bloom": 0.2,
    "scanlines": 0.0,
    "posterize": 0,
    # Gradient map: luminance (0-255) -> RGB. Black stays black; mids are slate blue,
    # highlights pale lavender-gray. Blending 20% of the original keeps sparse pink accents.
    "gradient_stops": [(0, (0, 0, 0)), (70, (45, 67, 83)), (125, (102, 120, 134)),
                       (175, (168, 174, 186)), (255, (235, 231, 241))],
    "gradient_strength": 0.8,
}

DIGIT_BGR = (220, 240, 110)                       # mint-cyan digits
PARTICLE_BGR = [(224, 204, 217), (255, 255, 255), (230, 160, 190), (220, 230, 160)]


# ---- Pixelation -----------------------------------------------------------------------
def pixelate(img, block):
    """Downscale then upscale with nearest neighbor: the classic mosaic."""
    h, w = img.shape[:2]
    block = max(1, int(block))
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def dissolve_blocks(roi, progress, blocks_across, seed):
    """progress 0 -> original, 1 -> coarse blocks; each block flips at its own random moment."""
    h, w = roi.shape[:2]
    medium_block = max(1, w // (blocks_across * 2))
    coarse_block = max(1, w // blocks_across)
    medium, coarse = pixelate(roi, medium_block), pixelate(roi, coarse_block)
    rng = np.random.default_rng(seed)
    grid = rng.random((h // medium_block + 1, w // medium_block + 1)).astype(np.float32)
    r = cv2.resize(grid, (w, h), interpolation=cv2.INTER_NEAREST)
    out = roi.copy()
    m1 = r < progress
    out[m1] = medium[m1]
    m2 = r < progress * progress
    out[m2] = coarse[m2]
    return out


# ---- Extra effects --------------------------------------------------------------------
def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def halftone(roi, progress, cells_across):
    """Periodic structure: the image becomes a hexagonal lattice of dots, each the average color of its cell.
    Dot size follows brightness, and the lattice takes over as progress rises (0 = clean photo)."""
    h, w = roi.shape[:2]
    cell = max(5.0, w / max(4, cells_across))
    rows = int(h / (cell * 0.866)) + 2
    cols = int(w / cell) + 2
    small = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    canvas = np.zeros_like(roi)
    for r in range(rows):
        y = r * cell * 0.866
        for c in range(cols):
            x = (c + 0.5 * (r % 2)) * cell
            rad = cell * 0.62 * math.sqrt(lum[r, c]) + 0.6
            cv2.circle(canvas, (int(x), int(y)), int(round(rad)), tuple(int(v) for v in small[min(r, rows - 1), min(c, cols - 1)]), -1, cv2.LINE_AA)
    a = smoothstep(progress * 1.4)
    return cv2.addWeighted(roi, 1 - a, canvas, a, 0)


def blur_dissolve(roi, progress):
    """Gaussian blur that grows with progress (0 = sharp)."""
    k = int(progress * roi.shape[1] * 0.10) | 1
    return roi if k < 3 else cv2.GaussianBlur(roi, (k, k), 0)


_grid_cache = {}


def ripple(roi, progress, t):
    """Radial ripple: rings of displacement travel outward from the center and grow with progress."""
    h, w = roi.shape[:2]
    key = (h, w)
    if key not in _grid_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - w / 2, yy - h / 2
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        _grid_cache[key] = (xx, yy, dx / r, dy / r, r)
    xx, yy, ux, uy, r = _grid_cache[key]
    wavelength = w / 5.0
    amp = w * 0.045 * smoothstep(progress * 1.6)
    disp = amp * np.sin(2 * math.pi * (r / wavelength - 1.2 * t))
    return cv2.remap(roi, xx + ux * disp, yy + uy * disp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def moire(effect, progress, t):
    """Periodic structure: two slightly rotated line gratings interfere into broad moire bands that drift over time."""
    h, w = effect.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx - w / 2, yy - h / 2
    period = max(5.0, w / 28.0)
    th1, th2 = 0.15 * t, 0.15 * t + 0.09
    g1 = np.sin(2 * math.pi * (xx * math.cos(th1) + yy * math.sin(th1)) / period)
    g2 = np.sin(2 * math.pi * (xx * math.cos(th2) + yy * math.sin(th2)) / period + t)
    band = 0.5 + 0.5 * g1 * g2                                # 0..1, wide fringes where the gratings beat
    a = 0.75 * smoothstep(progress * 1.5)
    shade = 1 - a * (1 - band)[..., None]
    tint = np.array(DIGIT_BGR, np.float32) * 0.25 * a * (1 - band)[..., None]
    return np.clip(effect.astype(np.float32) * shade + tint, 0, 255).astype(np.uint8)


IRID_PALETTES = {   # (position 0..1, (R, G, B)); the last stop should equal the first so the colors wrap smoothly
    "cool": [(0.00, (12, 10, 70)), (0.18, (60, 50, 220)), (0.34, (130, 80, 255)), (0.48, (70, 190, 255)),
             (0.58, (255, 255, 255)), (0.70, (150, 70, 240)), (0.86, (40, 170, 190)), (1.00, (12, 10, 70))],
    "oil": [(0.00, (20, 10, 90)), (0.15, (70, 50, 230)), (0.30, (150, 80, 255)), (0.45, (60, 200, 255)),
            (0.55, (255, 255, 255)), (0.70, (230, 70, 200)), (0.88, (60, 220, 180)), (1.00, (20, 10, 90))],
}


def make_lut(name):
    stops = IRID_PALETTES[name]
    pos = [p for p, _ in stops]
    lut = np.zeros((256, 3), np.float32)
    for ch in range(3):                                # stored as BGR
        lut[:, 2 - ch] = np.interp(np.linspace(0, 1, 256), pos, [c[ch] for _, c in stops])
    return lut


IRID_LUTS = {k: make_lut(k) for k in IRID_PALETTES}


def iridescence_core(roi, matte, field_fn, wl, progress, t, strength, palette, glitch, tear=None, rings=1.0):
    """Colors a region with thin-film bands.
    field_fn(xs, yy, squash) gives each pixel's 'distance' (the rings follow its contours); wl is the ring spacing in pixels.
    Three independent dials:
      rings  0..1  how much ring/contour structure there is (0 = none; color comes from the picture and the glitch alone)
      glitch 0..1  warping of the contours by noise and shear, plus blocks where the colors jump
      tear   0..3  horizontal damage: bands slide sideways, bands get bright/dark/shifted color, smeared streaks, channel split
                   (None = same as glitch)"""
    h, w = roi.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = smoothstep(progress * 2.0)                          # damage builds with the dissolve
    g = glitch * ramp
    gt = (glitch if tear is None else tear) * ramp
    rng = np.random.default_rng(int(t * 12) + 7919)            # new random pattern about every 1/12 s

    # horizontal damage: rows are cut into bands; torn bands slide, change brightness and shift their color
    shift = np.zeros(h, np.float32)
    gain = np.ones(h, np.float32)
    phase = np.zeros(h, np.float32)
    y = 0
    while y < h:
        bh_ = int(rng.integers(max(3, h // 70), max(6, h // 7)))
        if rng.random() < min(0.95, 0.55 * gt):
            shift[y:y + bh_] = np.clip(rng.normal(0, 0.16 * w * gt), -0.45 * w, 0.45 * w)
            gain[y:y + bh_] = rng.choice([0.2, 0.5, 1.5, 2.2]) if gt > 0.8 else rng.uniform(0.6, 1.4)
            phase[y:y + bh_] = rng.uniform(-0.5, 0.5) * min(gt, 2.0)
        y += bh_
    xs = xx - shift[:, None]
    base = cv2.remap(roi, xs, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) if gt > 0 else roi
    lum = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, (0, 0), 2.0)

    # rings, warped by noise and shear
    r = field_fn(xs, yy, 1.0 - 0.55 * g)
    noise = cv2.resize(rng.random((max(3, h // 45), max(3, w // 45))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    d = rings * (r / wl - 1.2 * t) + 0.9 * lum + 0.15 * np.sin(yy / h * 9 + 2 * t) + (noise - 0.5) * 2.2 * g
    d = d + ((yy - h / 2) / h) * 1.5 * g * np.sin((xs - w / 2) / w * 5 + t) * rings
    d = d + phase[:, None]

    # blocks where the colors jump
    if g > 0:
        for _ in range(int(rng.integers(2, 3 + int(5 * g)))):
            bx, by = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw_, bh_ = int(rng.integers(w // 12, w // 3)), int(rng.integers(h // 40, h // 8))
            d[by:by + bh_, bx:bx + bw_] += rng.uniform(0.25, 0.75)

    idx = ((d % 1.0) * 255).astype(np.uint8)
    color = IRID_LUTS[palette][idx] * (0.55 + 0.8 * lum)[..., None]
    f = (d * 5) % 1.0
    lines = np.clip(1 - np.abs(f - 0.5) * 2 / 0.12, 0, 1)[..., None] * rings
    color = color + lines * np.array((255, 255, 220), np.float32) * 0.6
    color = color * gain[:, None, None]

    if gt > 0:
        for _ in range(int(rng.integers(1, 2 + min(14, int(5 * gt))))):          # smeared bright streaks
            sy, sh_ = int(rng.integers(0, h)), int(rng.integers(max(2, h // 90), max(4, h // 16)))
            k = int(rng.integers(w // 6, w // 1.5)) | 1
            color[sy:sy + sh_] = np.clip(cv2.blur(color[sy:sy + sh_], (k, 1)) * 1.35 + 20, 0, 255)
        split = min(40, int(2 + 8 * gt))                                          # channel split
        color[..., 0] = np.roll(color[..., 0], split, axis=1)
        color[..., 2] = np.roll(color[..., 2], -split, axis=1)
        for _ in range(min(12, int(3 * gt))):
            sy, sh_ = int(rng.integers(0, h)), int(rng.integers(h // 60 + 2, h // 14 + 3))
            color[sy:sy + sh_, :, 1] = np.roll(color[sy:sy + sh_, :, 1], int(rng.integers(-w // 12, w // 12)), axis=1)

    a = np.clip(strength * smoothstep(progress * 1.6) * matte, 0, 1)
    torn = base * (1 - a) + np.clip(color, 0, 255) * a
    mm = np.clip(matte, 0, 1)                                  # slid pixels must not leak outside the soft mask
    return np.clip(roi * (1 - mm) + torn * mm, 0, 255).astype(np.uint8)


def iridescence(frame, box, progress, t, strength=0.85, palette="cool", glitch=0.7, tear=None, rings=1.0, ring_size=1.0):
    """Head version: rings spread from the head's center. Run AFTER the color grade."""
    H, W = frame.shape[:2]
    cx, cy, bw, bh = box
    x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    x1, y1 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return frame
    roi = frame[y0:y1, x0:x1]
    h, w = roi.shape[:2]

    def field(xs, yy, squash):
        return np.sqrt(((xs - w / 2) * squash) ** 2 + (yy - h / 2) ** 2 + 1e-3)

    matte = make_matte(roi.shape, cx - x0, cy - y0, bw, bh)
    out = frame.copy()
    out[y0:y1, x0:x1] = iridescence_core(roi, matte, field, w / 5.0 * ring_size, progress, t, strength, palette, glitch, tear, rings)
    return out


def iridescence_body(frame, info, progress, t, strength=0.85, palette="cool", glitch=0.7, tear=None, rings=1.0, ring_size=1.0):
    """Body version: the bands follow the person's outline (distance from the silhouette edge)."""
    x0, y0, x1, y1 = info["rect"]
    roi = frame[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    sw, sh = info["small"]
    sroi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if (sw, sh) != (w, h) else roi
    sd = info["sd"]

    def field(xs, yy, squash):
        return -cv2.remap(sd, xs, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    res = iridescence_core(sroi, info["matte"][..., None], field, sw / 7.0 * ring_size, progress, t, strength, palette, glitch, tear, rings)
    if (sw, sh) != (w, h):
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)
        ma = cv2.resize(info["matte"], (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        res = (roi * (1 - ma) + res * ma).astype(np.uint8)
    out = frame.copy()
    out[y0:y1, x0:x1] = res
    return out


def make_matte(shape, cx, cy, bw, bh):
    """Soft-edged ellipse (0..1, shape HxWx1) that keeps head effects inside the box."""
    matte = np.zeros(shape[:2], np.float32)
    cv2.ellipse(matte, (int(cx), int(cy)), (int(bw / 2 * 0.8), int(bh / 2 * 0.8)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(matte, (0, 0), max(3.0, bw * 0.06))[..., None]


CODE_TOKENS = ["MOV", "XOR", "JMP", "PUSH", "POP", "CALL", "NOP", "LEA", "AND", "INT3", "EAX", "RSP", "0x7F", "0xFF",
               "0x00", "0xDE", "0xAD", "RET", "CMP", "JNZ"]
GLYPH_STYLES = ("binary", "hex", "code", "enochian")
ENOCH = {"path": "", "fonts": {}, "masks": {}}


def find_enochian_font(path=""):
    """An Enochian .ttf/.otf: the --enochian-font path, or any Enochian*.ttf/otf next to this script or in the current folder."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    for pat in ([path] if path else []) + [os.path.join(d, f"Enochian*.{e}") for d in (here, os.getcwd()) for e in ("otf", "ttf")]:
        for hit in sorted(glob.glob(pat)):
            return hit
    return ""


def enochian_mask(text, px):
    """Grayscale bitmap of text in the Enochian font at px pixels tall (cached)."""
    key = (text, px)
    if key not in ENOCH["masks"]:
        from PIL import Image, ImageDraw, ImageFont
        font = ENOCH["fonts"].setdefault(px, ImageFont.truetype(ENOCH["path"], px))
        l, t_, r, b = font.getbbox(text)
        im = Image.new("L", (max(1, r - l + 2), max(1, b - t_ + 2)), 0)
        ImageDraw.Draw(im).text((-l + 1, -t_ + 1), text, font=font, fill=255)
        ENOCH["masks"][key] = np.array(im)
    return ENOCH["masks"][key]


def blit_mask(layer, mask, x, y, color):
    """Draw a grayscale mask in a color onto layer with its top-left at (x, y), clipped to the layer."""
    H, W = layer.shape[:2]
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x][..., None].astype(np.float32) / 255.0
    region = layer[y0:y1, x0:x1].astype(np.float32)
    layer[y0:y1, x0:x1] = np.maximum(region, m * np.array(color, np.float32)).astype(np.uint8)


def parse_glyphs(text):
    """'binary', 'code,enochian', 'all' ... -> list of styles."""
    names = [n.strip() for n in text.replace("+", ",").split(",") if n.strip()]
    if "all" in names or "mix" in names:
        return list(GLYPH_STYLES)
    bad = [n for n in names if n not in GLYPH_STYLES]
    if bad or not names:
        raise SystemExit(f"--glyphs: unknown style {bad}. Choose from {', '.join(GLYPH_STYLES)}, comma-separated (e.g. code,enochian), or 'all'.")
    return names


GLYPH_SIZE = 1.0  # set by --glyph-size


def draw_digits(roi, box_in_roi, progress, seed, style="binary"):
    """Layer of random characters over the head box. style is a list (or comma string) of: binary (0/1), hex (byte pairs),
    code (assembly-like tokens), enochian (needs the font). With several styles, each token picks one at random."""
    styles = parse_glyphs(style) if isinstance(style, str) else style
    cx, cy, bw, bh = box_in_roi
    cell = max(8, int(bw / 16 * GLYPH_SIZE))
    scale = cell * 0.8 / 22.0
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(roi)
    if styles == ["binary"]:
        for gy in range(int(cy - bh / 2), int(cy + bh / 2), cell):
            for gx in range(int(cx - bw / 2), int(cx + bw / 2), cell):
                show, flip = rng.random(), rng.random()
                if show > progress * 0.8:
                    continue
                cv2.putText(layer, "1" if flip > 0.5 else "0", (gx, gy + cell),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, DIGIT_BGR, 1, cv2.LINE_AA)
        return layer
    for gy in range(int(cy - bh / 2), int(cy + bh / 2), cell):
        gx = int(cx - bw / 2)
        while gx < cx + bw / 2:
            show = rng.random()
            kind = styles[int(rng.integers(0, len(styles)))]
            if kind == "enochian":
                tok = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), int(rng.integers(1, 3))))
                mask = enochian_mask(tok, max(8, int(cell * 1.15)))
                if show <= progress * 0.8:
                    blit_mask(layer, mask, gx, gy + cell - mask.shape[0], DIGIT_BGR)
                gx += mask.shape[1] + int(cell * 0.4)
                continue
            if kind == "binary":
                tok = "1" if rng.random() > 0.5 else "0"
            elif kind == "hex":
                tok = f"{int(rng.integers(0, 256)):02X}"
            else:
                tok = CODE_TOKENS[int(rng.integers(0, len(CODE_TOKENS)))]
            w_tok = cv2.getTextSize(tok, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            if show <= progress * 0.8:
                cv2.putText(layer, tok, (gx, gy + cell), cv2.FONT_HERSHEY_SIMPLEX, scale, DIGIT_BGR, 1, cv2.LINE_AA)
            gx += w_tok + int(cell * 0.45)
    return layer


def smear(frame, box, s, seed):
    """Stretch the pixels at the head's right side into horizontal streaks (glitchy bands)."""
    if s <= 0.02:
        return frame
    H, W = frame.shape[:2]
    cx, cy, bw, bh = box
    x_src = int(min(W - 2, max(1, cx + 0.25 * bw)))
    y0, y1 = max(0, int(cy - bh / 2)), min(H, int(cy + bh / 2))
    band = max(3, int(bh / 60))
    reach = W - x_src
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for yb in range(y0, y1, band):
        ye = min(y1, yb + band)
        length = int(s * reach * (0.15 + 0.85 * rng.random()))
        if length < 4:
            continue
        col = frame[yb:ye, x_src].astype(np.float32)
        fade = ((1 - np.arange(length) / length) * 0.9)[None, :, None]
        seg = out[yb:ye, x_src:x_src + length].astype(np.float32)
        out[yb:ye, x_src:x_src + length] = (seg * (1 - fade) + col[:, None, :] * fade).astype(np.uint8)
    return out


def particles(frame, origin, s, seed, n=220):
    """Small colored squares that launch from the head and drift right."""
    if s <= 0.02:
        return frame
    H, W = frame.shape[:2]
    ox, oy = origin
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for _ in range(n):
        delay, size = rng.random(), int(rng.integers(2, 11))
        vx, vy = 0.3 + rng.random(), (rng.random() - 0.5) * 0.5
        col = PARTICLE_BGR[int(rng.integers(len(PARTICLE_BGR)))]
        wobble = (rng.random() - 0.5) * 20
        t = s - 0.6 * delay
        if t <= 0:
            continue
        x, y = int(ox + vx * t * W * 0.6), int(oy + vy * t * H * 0.6 + wobble)
        if 0 <= x < W - size and 0 <= y < H - size:
            cv2.rectangle(out, (x, y), (x + size, y + size), col, -1)
    return out


# ---- Head tracking and compositing ----------------------------------------------------
CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + CASCADE_NAME
cascade = None


def load_cascade():
    """Find the Haar face model. Checks next to this script first, then OpenCV's own data folder."""
    if not hasattr(cv2, "CascadeClassifier"):
        raise SystemExit(
            f"OpenCV {cv2.__version__} has no CascadeClassifier (OpenCV 5 moved it to contrib).\n"
            "Fix: pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n"
            '     pip install "opencv-python<5"')
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, CASCADE_NAME)]
    data = getattr(cv2, "data", None)
    if data is not None:
        candidates.append(os.path.join(data.haarcascades, CASCADE_NAME))
    for path in candidates:
        if os.path.exists(path):
            model = cv2.CascadeClassifier(path)
            if not model.empty():
                return model
    raise SystemExit(
        f"Could not find {CASCADE_NAME} (OpenCV {cv2.__version__}).\n"
        "Option 1 (simplest): use OpenCV 4:  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n"
        '                                    pip install "opencv-python<5"\n'
        f"Option 2: download the model file into the folder with this script:\n"
        f"    curl -L -o {CASCADE_NAME} {CASCADE_URL}")


ROTATIONS = [
    # (cv2 rotate code or None, function mapping a point in the rotated image back to the original, swap w/h)
    (None, lambda x, y, w, h: (x, y), False),
    (cv2.ROTATE_90_CLOCKWISE, lambda x, y, w, h: (y, h - x), True),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y, w, h: (w - y, x), True),
    (cv2.ROTATE_180, lambda x, y, w, h: (w - x, h - y), False),
]


YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_NAME
yunet = None
last_candidates = []     # every detection from the latest detect_heads call: (box, kept?, note), for --debug


def load_yunet(path, score):
    """YuNet is a small neural face detector. It copes better with tilted, partly cropped and bespectacled faces."""
    if not hasattr(cv2, "FaceDetectorYN"):
        raise SystemExit(f"OpenCV {cv2.__version__} has no FaceDetectorYN. Use the Haar detector (default) or OpenCV 4.5.4+.")
    if path and not os.path.exists(path):
        raise SystemExit(f"--yunet-model {path} does not exist.")
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([path] if path else []) + [os.path.join(here, YUNET_NAME)]:
        if p and os.path.exists(p):
            return cv2.FaceDetectorYN.create(p, "", (320, 320), score, 0.3, 5000)
    raise SystemExit(f"Could not find the YuNet model {YUNET_NAME}. Download it next to this script:\n"
                     f"    curl -L -o {YUNET_NAME} {YUNET_URL}\n"
                     "(about 230 KB), or pass --yunet-model path/to/file.onnx")


def skin_fraction(frame, box):
    """Share of skin-colored pixels in the middle of a face (0..1). Printed or monochrome faces score near 0."""
    cx, cy, bw, bh = box
    fw, fh = bw / 1.7, bh / 2.0
    fcy = cy + 0.12 * fh
    x0, x1 = int(cx - 0.3 * fw), int(cx + 0.3 * fw)
    y0, y1 = int(fcy - 0.3 * fh), int(fcy + 0.3 * fh)
    H, W = frame.shape[:2]
    patch = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if patch.size == 0:
        return 0.0
    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    return float(cv2.inRange(ycc, (40, 133, 77), (255, 173, 127)).mean() / 255.0)


def detect_heads(frame, min_neighbors=5, try_rotations=False, max_faces=1, min_face=1 / 14, detect_width=480,
                 detector="haar", region=None, skin_min=0.0, rank="score"):
    """Return up to max_faces head boxes [(cx, cy, w, h), ...].
    Steps: find candidate faces -> drop those outside --region or below --skin-min -> order by --rank -> merge overlaps."""
    global last_candidates
    h, w = frame.shape[:2]
    scale = detect_width / w if w > detect_width else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    sh, sw = small.shape[:2]
    candidates = []                                        # (score, (cx, cy, w, h) in full-frame pixels)
    if detector == "yunet":
        yunet.setInputSize((sw, sh))
        _, found = yunet.detect(small)
        for f in ([] if found is None else found):
            fx, fy, fw, fh, score = f[0], f[1], f[2], f[3], f[14]
            if fw < sw * min_face:
                continue
            cx, cy = fx + fw / 2, fy + fh / 2 - 0.12 * fh
            candidates.append((float(score), (cx / scale, cy / scale, fw * 1.7 / scale, fh * 2.0 / scale)))
    else:
        gray0 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for code, back, swap in (ROTATIONS if try_rotations else ROTATIONS[:1]):
            gray = gray0 if code is None else cv2.rotate(gray0, code)
            faces, _, weights = cascade.detectMultiScale3(
                gray, 1.1, min_neighbors, minSize=(max(12, int(gray.shape[1] * min_face)),) * 2, outputRejectLevels=True)
            if len(faces) == 0:
                continue
            for (fx, fy, fw, fh), weight in zip(faces, weights):
                cx, cy = fx + fw / 2, fy + fh / 2 - 0.12 * fh      # shift up to include hair, in the rotated frame
                bw, bh = fw * 1.7, fh * 2.0
                cx, cy = back(cx, cy, sw, sh)
                if swap:
                    bw, bh = bh, bw
                candidates.append((float(weight), (cx / scale, cy / scale, bw / scale, bh / scale)))
    candidates.sort(key=lambda c: -c[0])

    survivors, report = [], []
    for score, box in candidates:
        note = f"score {score:.2f}"
        reason = None
        if region is not None:
            x0, y0, x1, y1 = region
            if not (x0 * w <= box[0] <= x1 * w and y0 * h <= box[1] <= y1 * h):
                reason = "outside region"
        sk = skin_fraction(frame, box)                     # always measured, so --debug can show it
        note += f" skin {sk:.2f}"
        if reason is None and sk < skin_min:
            reason = "skin"
        if reason:
            report.append((box, False, f"{note} X {reason}"))
        else:
            survivors.append((score, box, note))
    if rank == "size":
        survivors.sort(key=lambda c: -c[1][2])
    kept = []
    for score, box, note in survivors:
        if len(kept) < max_faces and all(math.hypot(box[0] - k[0], box[1] - k[1]) > 0.5 * min(box[2], k[2]) for k in kept):
            kept.append(box)
            report.append((box, True, note))
    last_candidates = report[:12]
    return kept


class FaceTracker:
    """Keeps a small list of tracks. Each detection is matched to the nearest existing track
    (smoothing its box); unmatched detections start new tracks; tracks unseen for too long are dropped."""

    def __init__(self, max_faces=1, hold=15, confirm=1):
        self.max_faces, self.hold, self.confirm = max_faces, hold, confirm
        self.tracks, self.next_id = [], 0

    def update(self, detections):
        if detections is None:                      # detection was not run on this frame
            return
        pairs = sorted((math.hypot(t["box"][0] - d[0], t["box"][1] - d[1]), ti, di)
                       for ti, t in enumerate(self.tracks) for di, d in enumerate(detections))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            t, d = self.tracks[ti], detections[di]
            if ti in used_t or di in used_d or dist > max(t["box"][2], d[2]):
                continue
            t["box"] = tuple(0.65 * b + 0.35 * f for b, f in zip(t["box"], d))
            t["missed"], t["hits"], t["fresh"] = 0, t["hits"] + 1, True
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1
                t["fresh"] = False
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            if len(self.tracks) >= self.max_faces:  # full: replace the stalest track, if any is stale
                stalest = max(self.tracks, key=lambda t: t["missed"])
                if stalest["missed"] == 0:
                    continue
                self.tracks.remove(stalest)
            self.tracks.append({"id": self.next_id, "box": d, "missed": 0, "hits": 1, "fresh": True})
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.hold]

    def active(self):
        return [t for t in self.tracks if t["hits"] >= self.confirm and t["missed"] <= self.hold]


def base_dissolve(src, progress, opts, seed, density=1):
    mode = opts.get("mode", "blocks")
    if mode == "blocks":
        return dissolve_blocks(src, progress, opts["blocks_across"] * density, seed)
    if mode == "halftone":
        return halftone(src, progress, opts["blocks_across"] * density)
    if mode == "blur":
        return blur_dissolve(src, progress)
    return src.copy()


def apply_head_effect(frame, box, progress, opts, seed, cycle):
    H, W = frame.shape[:2]
    cx, cy, bw, bh = box
    x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    x1, y1 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return frame
    roi = frame[y0:y1, x0:x1]
    t = opts.get("t", 0.0)
    src = ripple(roi, progress, t) if opts.get("ripple") else roi
    effect = base_dissolve(src, progress, opts, seed)
    if opts.get("moire"):
        effect = moire(effect, progress, t)
    matte = make_matte(roi.shape, cx - x0, cy - y0, bw, bh)
    if opts["digits"]:
        layer = draw_digits(roi, (cx - x0, cy - y0, bw, bh), progress, seed, opts.get("glyphs", "binary"))
        cover = (layer.max(axis=2, keepdims=True) > 0).astype(np.float32) * 0.85
        effect = (effect * (1 - cover) + layer * cover).astype(np.uint8)
    out = frame.copy()
    out[y0:y1, x0:x1] = (roi * (1 - matte) + effect * matte).astype(np.uint8)
    s = progress * progress
    if opts["smear"]:
        out = smear(out, box, s, seed)
    if opts["particles"]:
        out = particles(out, (cx, cy), s, cycle)
    return out


# ---- Whole-body effect via background subtraction -------------------------------------
BG_WIDTH = 480


def small_size(w, h):
    sc = BG_WIDTH / w if w > BG_WIDTH else 1.0
    return int(w * sc), int(h * sc)


def crop_rect(w, h, crop):
    """crop = (x0, y0, x1, y1) as fractions of the frame -> even pixel rectangle (x0, y0, x1, y1)."""
    if crop is None:
        return 0, 0, w, h
    x0, y0, x1, y1 = int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)
    x1, y1 = x0 + (x1 - x0) // 2 * 2, y0 + (y1 - y0) // 2 * 2
    return x0, y0, x1, y1


def build_background(path, limit_frames, bg_image="", crop=None, thr=28):
    """The empty scene as seen by a fixed camera: the per-pixel median of ~40 frames spread across the clip,
    then refined twice: pixels where the person stood are left out of the median, which removes the 'ghost' that
    appears when you stay in one place for a while. Or a photo of the empty room via --bg-image."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0, cx1, cy1 = crop_rect(w0, h0, crop)
    sw, sh = small_size(cx1 - cx0, cy1 - cy0)
    if bg_image:
        img = cv2.imread(bg_image)
        if img is None:
            raise SystemExit(f"Could not read --bg-image {bg_image}")
        if abs(img.shape[1] / img.shape[0] - w0 / h0) > 0.02:
            raise SystemExit("--bg-image must have the same shape (aspect ratio) as the video.")
        cap.release()
        img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)[cy0:cy1, cx0:cx1]
        return cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    if limit_frames:
        total = min(total, limit_frames)
    picks = np.linspace(0, total - 1, min(40, total)).astype(int)
    frames = []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f[cy0:cy1, cx0:cx1], (sw, sh), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 5:
        raise SystemExit("Not enough frames to estimate the background. Use --bg-image with a photo of the empty scene.")
    stack = np.stack(frames)
    bg = np.median(stack, axis=0).astype(np.uint8)
    k = max(3, sw // 30) | 1
    blurred = np.stack([cv2.GaussianBlur(f, (5, 5), 0) for f in frames])
    for _ in range(2):
        bgb = cv2.GaussianBlur(bg, (5, 5), 0)
        person = np.stack([cv2.dilate((cv2.absdiff(f, bgb).max(axis=2) > thr).astype(np.uint8), np.ones((k, k), np.uint8)) for f in blurred])
        data = stack.astype(np.float32)
        data[person.astype(bool)] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refined = np.nanmedian(data, axis=0)
        bad = np.isnan(refined)                     # pixels covered by the person in every sample keep the old estimate
        refined[bad] = bg[bad]
        bg = refined.astype(np.uint8)
    return bg


SEG_NAME = "human_segmentation_pphumanseg_2023mar.onnx"
SEG_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/" + SEG_NAME
SEG = {"net": None}


def load_seg_model(path=""):
    """PP-HumanSeg: a small neural network that finds people in any frame (no fixed camera needed)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if path and not os.path.exists(path):
        raise SystemExit(f"--seg-model {path} does not exist.")
    for cand in ([path] if path else []) + [os.path.join(here, SEG_NAME), os.path.join(os.getcwd(), SEG_NAME)]:
        if os.path.exists(cand):
            SEG["net"] = cv2.dnn.readNet(cand)
            return
    raise SystemExit(f"Could not find the person-segmentation model {SEG_NAME}. Download it next to this script (6 MB):\n"
                     f"    curl -L -o {SEG_NAME} {SEG_URL}\nor pass --seg-model path/to/file.onnx")


def neural_mask(frame):
    """Float 0..1 person probability at full-frame size."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 127.5, (192, 192), (127.5, 127.5, 127.5), swapRB=True)
    SEG["net"].setInput(blob)
    out = SEG["net"].forward()[0]
    e = np.exp(out - out.max(axis=0, keepdims=True))
    prob = (e[1] / e.sum(axis=0)).astype(np.float32)
    return np.clip(cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def compute_mask(frame, bg_small, thr, state, cutout="bgsub", smooth=0.2):
    """Person mask (0..1). bgsub = differs from the empty scene; neural = a network finds people;
    hybrid = background subtraction for sharp edges, limited to where the network sees a person (kills ghosts)."""
    H, W = frame.shape[:2]
    if cutout == "neural":
        mask = (neural_mask(frame) > 0.5).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, W * 0.002))
    elif cutout == "bgsub":
        mask = person_mask(frame, bg_small, thr, state.setdefault("b", {}), smooth=0.0)
    else:
        nm = (neural_mask(frame) > 0.5).astype(np.uint8)
        k = max(3, int(W * 0.03)) | 1
        near = cv2.dilate(nm, np.ones((k, k), np.uint8)).astype(np.float32)          # allow edges to extend a little past the network's guess
        core = cv2.erode(nm, np.ones((k, k), np.uint8)).astype(np.float32)           # trust the network deep inside the person
        bgm = person_mask(frame, bg_small, thr, state.setdefault("b", {}), smooth=0.0)
        mask = np.clip(bgm * near + core, 0, 1)
    prev = state.get("final")
    if smooth > 0 and prev is not None and prev.shape == mask.shape:
        mask = (1 - smooth) * mask + smooth * prev
    state["final"] = mask
    return mask


def person_mask(frame, bg_small, thr, state, smooth=0.35):
    """Float mask (0..1, full-frame size) of what differs from the background. state keeps the previous mask for smoothing."""
    H, W = frame.shape[:2]
    sh, sw = bg_small.shape[:2]
    small = cv2.GaussianBlur(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), (5, 5), 0)
    diff = cv2.absdiff(small, cv2.GaussianBlur(bg_small, (5, 5), 0)).max(axis=2)
    m = (diff > thr).astype(np.uint8)
    k = max(3, sw // 90) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 4 + 1, k * 4 + 1), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) > 0.015 * sw * sh:
            cv2.drawContours(clean, [c], -1, 1, -1)          # filled, so holes inside the person close up
    mask = cv2.resize(clean.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    prev = state.get("mask")
    if smooth > 0 and prev is not None and prev.shape == mask.shape:
        mask = (1 - smooth) * mask + smooth * prev
    state["mask"] = mask
    return mask


BODY_WORK_WIDTH = 720      # body effects are computed at this width (then scaled up): much faster, slightly softer


def parse_color(text):
    """'#RRGGBB' or 'R,G,B' -> BGR tuple."""
    text = text.strip()
    try:
        if text.startswith("#") and len(text) == 7:
            r, g, b = int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
        else:
            r, g, b = (int(v) for v in text.split(","))
        return (b, g, r)
    except ValueError:
        raise SystemExit("--bg-color needs #RRGGBB (e.g. #0a0e1a) or R,G,B (e.g. 10,14,26).")


def replace_background(frame, mask, mode, color=(0, 0, 0), image=None):
    """Keep the person, change everything else: black | color | dim | blur | image."""
    H, W = frame.shape[:2]
    m = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), max(1.5, W * 0.002))[..., None]
    if mode == "black":
        fill = np.zeros_like(frame)
    elif mode == "color":
        fill = np.empty_like(frame)
        fill[:] = color
    elif mode == "dim":
        fill = (frame * 0.25).astype(np.uint8)
    elif mode == "blur":
        k = max(15, int(W * 0.03)) | 1
        fill = cv2.GaussianBlur(frame, (k, k), 0)
    else:
        fill = image
    return (frame * m + fill * (1 - m)).astype(np.uint8)


def apply_body_effect(frame, mask, progress, opts, t, seed, outline=False):
    """Dissolve, ripple, digits and smear applied to the silhouette. Returns (frame, info for the iridescence pass)."""
    H, W = frame.shape[:2]
    ys, xs_ = np.where(mask > 0.5)
    if len(xs_) < 400:
        return frame, None
    mx, my = int(0.06 * (xs_.max() - xs_.min())) + 4, int(0.04 * (ys.max() - ys.min())) + 4
    x0, x1 = max(0, xs_.min() - mx), min(W, xs_.max() + mx)
    y0, y1 = max(0, ys.min() - my), min(H, ys.max() + my)
    roi = frame[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    f = min(1.0, BODY_WORK_WIDTH / w)
    sw, sh = max(16, int(w * f)), max(16, int(h * f))
    sroi = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if f < 1 else roi
    m = cv2.resize(mask[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_AREA) if f < 1 else mask[y0:y1, x0:x1]
    mb = (m > 0.5).astype(np.uint8)
    sd = cv2.distanceTransform(mb, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - mb, cv2.DIST_L2, 3)   # + inside, - outside
    src = sroi
    if opts.get("ripple"):
        sdb = cv2.GaussianBlur(sd, (0, 0), 3)
        gx, gy = cv2.Sobel(sdb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sdb, cv2.CV_32F, 0, 1, ksize=3)
        n = np.sqrt(gx * gx + gy * gy) + 1e-3
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        amp = sw * 0.03 * smoothstep(progress * 1.6)
        disp = amp * np.sin(2 * math.pi * (sd / (sw / 7.0) - 1.2 * t))
        src = cv2.remap(sroi, xx + gx / n * disp, yy + gy / n * disp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    effect = base_dissolve(src, progress, opts, seed, density=2)
    if opts.get("moire"):
        effect = moire(effect, progress, t)
    if opts["digits"]:
        layer = draw_digits(sroi, (sw / 2, sh / 2, sw, sh), progress, seed, opts.get("glyphs", "binary"))
        cover = (layer.max(axis=2, keepdims=True) > 0).astype(np.float32) * 0.85
        effect = (effect * (1 - cover) + layer * cover).astype(np.uint8)
    feather = cv2.GaussianBlur(m, (0, 0), max(2.0, sw * 0.008))
    if outline:
        band = np.exp(-(sd / max(5.0, sw * 0.05)) ** 2)
        matte2 = np.clip(cv2.GaussianBlur(band, (0, 0), 2.0) * feather + band * (1 - feather) * 0.8, 0, 1)
    else:
        matte2 = np.clip(feather, 0, 1)
    up = (lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)) if f < 1 else (lambda a: a)
    matte = up(matte2)[..., None]
    out = frame.copy()
    out[y0:y1, x0:x1] = (roi * (1 - matte) + up(effect) * matte).astype(np.uint8)
    box = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    if opts["smear"]:
        out = smear(out, box, progress * progress, seed)
    if opts["particles"]:
        out = particles(out, (box[0], y0 + 0.12 * (y1 - y0)), progress * progress, seed)
    imatte = matte2 if outline else np.clip(feather, 0, 1)
    return out, {"rect": (x0, y0, x1, y1), "sd": sd.astype(np.float32), "matte": imatte.astype(np.float32),
                 "box": box, "small": (sw, sh)}


# ---- Color look -----------------------------------------------------------------------
def gradient_map(frame, stops, strength):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    xs = [s[0] for s in stops]
    lut = np.zeros((256, 1, 3), np.uint8)
    for c in range(3):                                   # BGR channel c uses RGB index 2 - c
        lut[:, 0, c] = np.interp(np.arange(256), xs, [s[1][2 - c] for s in stops])
    mapped = cv2.LUT(cv2.merge([gray, gray, gray]), lut)
    return cv2.addWeighted(mapped, strength, frame, 1 - strength, 0)


def grade(frame, look):
    out = frame
    if look.get("gradient_stops") and look.get("gradient_strength"):
        out = gradient_map(out, look["gradient_stops"], look["gradient_strength"])
    if look["hue_shift"] or look["saturation"] != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + look["hue_shift"] / 2) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * look["saturation"], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    s = int(look["chroma_shift"])
    if s:
        b, g, r = cv2.split(out)
        size = (out.shape[1], out.shape[0])
        r = cv2.warpAffine(r, np.float32([[1, 0, s], [0, 1, 0]]), size, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, np.float32([[1, 0, -s], [0, 1, 0]]), size, borderMode=cv2.BORDER_REPLICATE)
        out = cv2.merge([b, g, r])
    if look["bloom"]:
        bright = np.clip(out.astype(np.float32) - 160, 0, None) * (255 / 95)
        glow = cv2.GaussianBlur(bright, (0, 0), 15)
        out = np.clip(out.astype(np.float32) + glow * look["bloom"], 0, 255).astype(np.uint8)
    if look["posterize"]:
        n = int(look["posterize"])
        out = (np.round(out / 255.0 * (n - 1)) / (n - 1) * 255).astype(np.uint8)
    if look["scanlines"]:
        out = out.astype(np.float32)
        out[::3] *= 1 - look["scanlines"]
        out = out.astype(np.uint8)
    return out


# ---- Live-stream style HUD -------------------------------------------------------------
HUD_WHITE = (245, 240, 235)
HUD_GRAY = (170, 165, 160)
HUD_PINK = (170, 110, 255)       # BGR
HUD_CYAN = (255, 230, 130)       # BGR
HUD_RED = (60, 35, 250)
AVATAR_COLORS = [(190, 120, 255), (255, 210, 120), (200, 170, 255), (140, 230, 170)]
DEFAULT_COMMENTS = [
    ("neon_fox", "this is incredible"), ("pixelqueen", "wait what was that effect"),
    ("static_dream", "play it again"), ("moonbeam", "the colors!!"),
    ("glitchgarden", "how did you make this"), ("nightowl", "love this so much"),
]
HUD_TEXT_SCALE = 1.0   # set by --hud-text-scale
TITLE_SCALE = 1.0      # set by --hud-title-scale (extra size for the big title only)
# Fonts for the HUD text. "body" and "title" come from --hud-font / --hud-title-font; "fallback" (any Unicode-capable
# font) and "emoji" (a colour emoji font) are found automatically unless --emoji-font is given.
HUD_FONT = {"body": "", "title": "", "fallback": "", "emoji": "", "fonts": {}, "cache": {}}
UNICODE_OK = False
TITLE_FACES = (cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX)   # the built-in faces the HUD uses for titles
FALLBACK_FONTS = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                  "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
EMOJI_FONTS = ["C:/Windows/Fonts/seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc",
               "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf"]


def first_existing(paths):
    return next((p for p in paths if os.path.isfile(p)), "")


def is_emoji_char(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or o in (0x2B50, 0x2B55, 0x200D, 0xFE0F, 0x20E3)


def clean_text(text):
    """Keep full Unicode when a Pillow font is available, otherwise fold to plain ASCII."""
    return text if UNICODE_OK else ascii_safe(text)


def _load_font(path, px):
    from PIL import ImageFont
    key = (path, px)
    if key not in HUD_FONT["fonts"]:
        HUD_FONT["fonts"][key] = ImageFont.truetype(path, px)
    return HUD_FONT["fonts"][key]


def _emoji_image(run, px):
    """Colour emoji run as an RGBA numpy image about px tall. Bitmap emoji fonts (Noto, Apple) only come in one size, so they are scaled."""
    from PIL import Image, ImageDraw
    path, font, size = HUD_FONT["emoji"], None, px
    for size in (px, 109, 160, 136, 128):
        try:
            font = _load_font(path, size)
            break
        except OSError:
            continue
    w, h = int(font.getlength(run)) + 4, int(size * 1.4)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), run, font=font, embedded_color=True)
    k = px / float(size)
    if abs(k - 1) > 0.02:
        im = im.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    return np.array(im)


def _render(txt, px, path, stroke):
    """Text as a grayscale mask plus optional colour-emoji layer, cached. Returns (mask, emoji_rgba, width, ascent)."""
    key = (txt, px, path, stroke)
    cache = HUD_FONT["cache"]
    if key in cache:
        return cache[key]
    from PIL import Image, ImageDraw
    font = _load_font(path, px)
    runs = []
    for ch in txt:
        e = is_emoji_char(ch)
        if runs and runs[-1][0] == e:
            runs[-1][1] += ch
        else:
            runs.append([e, ch])
    asc, desc = int(px * 1.0) + 3, int(px * 0.4) + 3
    items, x = [], 0
    for e, run in runs:
        if e:
            if not HUD_FONT["emoji"]:
                continue                      # no emoji font found: skip the emoji rather than draw boxes
            arr = _emoji_image(run, px)
            items.append((True, x, arr))
            x += arr.shape[1] + int(px * 0.08)
        else:
            items.append((False, x, run))
            x += int(math.ceil(font.getlength(run)))
    W, Hc = max(1, x + 4), asc + desc
    mask = Image.new("L", (W, Hc), 0)
    d = ImageDraw.Draw(mask)
    emo = None
    for is_e, x0, item in items:
        if is_e:
            if emo is None:
                emo = np.zeros((Hc, W, 4), np.uint8)
            y0 = asc - int(px * 0.92)
            h, w = item.shape[:2]
            ys, xs = max(0, y0), x0
            ye, xe = min(Hc, y0 + h), min(W, x0 + w)
            if ye > ys and xe > xs:
                emo[ys:ye, xs:xe] = np.maximum(emo[ys:ye, xs:xe], item[ys - y0:ye - y0, 0:xe - xs])
        else:
            d.text((x0, asc), item, font=font, fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
    if len(cache) > 600:
        cache.clear()
    cache[key] = (np.array(mask), emo, W, asc)
    return cache[key]


def _font_for(font, txt):
    """Path of the TrueType font to draw this text with, or '' to use OpenCV's built-in font."""
    if font in TITLE_FACES and HUD_FONT["title"]:
        return HUD_FONT["title"]
    if HUD_FONT["body"]:
        return HUD_FONT["body"]
    if UNICODE_OK and not txt.isascii():
        return HUD_FONT["fallback"]
    return ""


def hud_text_size(txt, font, scale, thick):
    """Like cv2.getTextSize, but honours --hud-text-scale, custom fonts, Unicode and emoji."""
    scale = scale * HUD_TEXT_SCALE * (TITLE_SCALE if font in TITLE_FACES else 1.0)
    path = _font_for(font, txt)
    if not path:
        return cv2.getTextSize(txt, font, scale, thick)
    px = max(6, int(scale * 30))
    _, _, w, _ = _render(txt, px, path, 1 if thick >= 3 else 0)
    return (w, int(px * 0.75)), int(px * 0.25)


def put_text(img, txt, org, font, scale, color, thick=1, line=cv2.LINE_AA):
    """Like cv2.putText, but honours --hud-text-scale, custom fonts, Unicode and colour emoji (via Pillow)."""
    k = HUD_TEXT_SCALE * (TITLE_SCALE if font in TITLE_FACES else 1.0)
    scale = scale * k
    path = _font_for(font, txt)
    if not path:
        cv2.putText(img, txt, org, font, scale, color, max(1, int(round(thick * k))) if k > 1 else thick, line)
        return
    px = max(6, int(scale * 30))
    mask, emo, _, asc = _render(txt, px, path, 1 if thick >= 3 else 0)
    x, y = int(org[0]), int(org[1]) - asc
    blit_mask(img, mask, x, y, color)
    if emo is not None:
        H, W = img.shape[:2]
        h, w = emo.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            piece = emo[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
            a = piece[..., 3:4] / 255.0
            rgb = piece[..., 2::-1] * a                       # RGBA -> BGR, premultiplied
            region = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = np.maximum(region, rgb).astype(np.uint8)


FONT = cv2.FONT_HERSHEY_SIMPLEX


def tk(font_scale, bold=1.0):
    """Line thickness that suits a text size, so small text stays readable at any video size."""
    return max(1, int(round(font_scale * 1.5 * bold)))


def rounded_rect(img, p1, p2, r, color, thick):
    (x1, y1), (x2, y2) = p1, p2
    r = int(max(1, min(r, (y2 - y1) // 2, (x2 - x1) // 2)))
    A = cv2.LINE_AA
    for a, b in [((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                 ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))]:
        cv2.line(img, a, b, color, thick, A)
    for c, start in [((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                     ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)]:
        cv2.ellipse(img, c, (r, r), 0, start, start + 90, color, thick, A)


def heart_points(cx, cy, size):
    t = np.linspace(0, 2 * np.pi, 60)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return np.stack([cx + x * size / 34, cy + y * size / 34], axis=1).astype(np.int32)


def draw_heart(img, cx, cy, size, color, thick=0):
    pts = heart_points(cx, cy, size)
    if thick:
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    else:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def text_size(txt, scale, thick):
    return hud_text_size(txt, FONT, scale, thick)[0]


def draw_comment(layer, x, y, user, msg, k, s, max_px=10 ** 6):
    r = int(16 * s)
    cv2.circle(layer, (x + r, y), r, AVATAR_COLORS[k % len(AVATAR_COLORS)], -1, cv2.LINE_AA)
    cv2.circle(layer, (x + r, y), r, HUD_WHITE, max(1, int(s)), cv2.LINE_AA)
    tx = x + 2 * r + int(12 * s)
    put_text(layer, user, (tx, y - int(3 * s)), FONT, 0.55 * s, HUD_WHITE, tk(0.55 * s, 1.25), cv2.LINE_AA)
    msg = fit_text(msg, FONT, 0.5 * s, max(1, int(s)), max_px - (tx - x))
    put_text(layer, msg, (tx, y + int(20 * s)), FONT, 0.5 * s, HUD_WHITE, tk(0.5 * s, 1.0), cv2.LINE_AA)


def draw_hud(frame, t, frac, fps, cfg):
    """Draw the overlay on a black layer, then screen-blend it so it never leaves dark halos."""
    H, W = frame.shape[:2]
    s = H / 720.0
    th = max(1, int(round(s)))
    A = cv2.LINE_AA
    layer = np.zeros_like(frame)
    mx, my = int(0.04 * W), int(0.06 * H)
    frame = darken_rect(frame, (0.03, 0.54, 0.72, 0.84), s)

    # title with underline (top left)
    put_text(layer, cfg["title"], (mx, my + int(10 * s)), cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, HUD_WHITE, tk(0.9 * s, 1.25), A)
    cv2.line(layer, (mx, my + int((10 + 16 * TITLE_SCALE) * s)), (mx + int(0.32 * W), my + int((10 + 16 * TITLE_SCALE) * s)), HUD_CYAN, th, A)

    # corner brackets framing the picture
    bx0, by0, bx1, by1 = int(0.05 * W), int(0.15 * H), int(0.95 * W), int(0.86 * H)
    L = int(28 * s)
    for (x, y, dx, dy) in [(bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1)]:
        cv2.line(layer, (x, y), (x + dx * L, y), HUD_GRAY, th + 1, A)
        cv2.line(layer, (x, y), (x, y + dy * L), HUD_GRAY, th + 1, A)

    # LIVE badge and viewer count (top right)
    pill_h = int(30 * s)
    viewers = f"{(1200 + 250 * t) / 1000:.1f}k"
    vw = text_size(viewers, 0.55 * s, th)[0] + int(34 * s)
    lw = text_size("LIVE", 0.55 * s, th + 1)[0] + int(46 * s)
    x2 = W - mx
    rounded_rect(layer, (x2 - vw, my - int(8 * s)), (x2, my - int(8 * s) + pill_h), pill_h // 2, HUD_CYAN, th + 1)
    put_text(layer, viewers, (x2 - vw + int(17 * s), my + int(13 * s)), FONT, 0.55 * s, HUD_WHITE, tk(0.55 * s, 1.0), A)
    x1 = x2 - vw - int(12 * s) - lw
    rounded_rect(layer, (x1, my - int(8 * s)), (x1 + lw, my - int(8 * s) + pill_h), pill_h // 2, HUD_PINK, th + 1)
    if int(t * 2) % 2 == 0:                                   # blinking dot
        cv2.circle(layer, (x1 + int(18 * s), my + int(7 * s)), int(5 * s), HUD_PINK, -1, A)
    put_text(layer, "LIVE", (x1 + int(30 * s), my + int(13 * s)), FONT, 0.55 * s, HUD_WHITE, tk(0.55 * s, 1.25), A)

    # scrolling comment feed (bottom left): a new one every 1.6 s, each lives about 6.4 s
    comments, interval, life = cfg["comments"], 1.6, 6.4
    base_y, line_h = int(0.80 * H), int(58 * s)
    first = max(0, int((t - life) // interval) + 1)
    for k in range(first, int(t // interval) + 1) if comments else ():   # empty with --no-chat
        age = t - k * interval
        if not (0 <= age <= life):
            continue
        alpha = min(1.0, age / 0.35) * min(1.0, (life - age) / 1.2)
        slot = int(t // interval) - k                                          # 0 = newest comment
        y = base_y - int(slot * line_h)
        tmp = np.zeros_like(frame)
        user, msg = comments[k % len(comments)]
        draw_comment(tmp, mx + int(8 * s), y, user, msg, k, s, max_px=int(0.70 * W) - mx)
        layer = cv2.add(layer, cv2.convertScaleAbs(tmp, alpha=alpha))

    # message box with smiley, send arrow and heart (bottom)
    by_top, by_bot = int(0.905 * H), int(0.905 * H) + int(44 * s)
    rounded_rect(layer, (mx, by_top), (W - int(0.15 * W), by_bot), int(22 * s), HUD_CYAN, th + 1)
    sx, sy, sr = mx + int(26 * s), (by_top + by_bot) // 2, int(11 * s)
    cv2.circle(layer, (sx, sy), sr, HUD_WHITE, th, A)
    for dx in (-0.35, 0.35):
        cv2.circle(layer, (sx + int(dx * sr), sy - int(0.25 * sr)), max(1, int(1.2 * s)), HUD_WHITE, -1, A)
    cv2.ellipse(layer, (sx, sy + int(0.1 * sr)), (int(0.55 * sr), int(0.45 * sr)), 0, 20, 160, HUD_WHITE, th, A)
    put_text(layer, cfg["prompt"], (sx + int(26 * s), sy + int(4 * s)), FONT, 0.4 * s, HUD_CYAN, tk(0.4 * s, 1.0), A)
    px, py, ps = W - int(0.12 * W), sy, int(18 * s)
    plane = np.array([(0, -0.5), (1, 0), (0, 0.5), (0.3, 0)]) * ps + (px, py)
    cv2.polylines(layer, [plane.astype(np.int32)], True, HUD_WHITE, th + 1, A)
    draw_heart(layer, W - int(0.05 * W), py, int(30 * s), HUD_PINK, th + 1)

    # progress bar along the very bottom
    barY = int(0.885 * H)
    cv2.line(layer, (0, barY), (W, barY), HUD_GRAY, max(1, th), A)
    cv2.line(layer, (0, barY), (int(W * frac), barY), HUD_RED, th + 2, A)
    cv2.circle(layer, (int(W * frac), barY), int(6 * s), HUD_RED, -1, A)

    # status line (bottom left): the fps is real, the rest is decoration
    signal = 96 + int(3 * abs(math.sin(t * 1.7)))
    put_text(layer, f"NET//STABLE   {fps:.0f}FPS   SIGNAL {signal}%", (mx, int(0.975 * H)),
                FONT, 0.38 * s, HUD_CYAN, tk(0.38 * s, 1.0), A)

    # floating hearts (right edge), each rising and fading over 2.6 s
    for k in range(int(t / 0.35) - 8, int(t / 0.35) + 1):
        if k < 0:
            continue
        rng = np.random.default_rng(k)
        age = t - k * 0.35
        if not (0 <= age <= 2.6):
            continue
        a = min(1.0, age / 0.3) * min(1.0, (2.6 - age) / 0.8)
        x = int(W * (0.90 + 0.06 * rng.random()))
        y = int(H * 0.80 - age / 2.6 * H * 0.4)
        tmp = np.zeros_like(frame)
        size = int((16 + 14 * rng.random()) * s)
        if rng.random() > 0.5:
            draw_heart(tmp, x, y, size, HUD_WHITE)
        else:
            draw_heart(tmp, x, y, size, HUD_WHITE, th + 1)
        layer = cv2.add(layer, cv2.convertScaleAbs(tmp, alpha=a))

    prod = cv2.multiply(cv2.bitwise_not(frame), cv2.bitwise_not(layer), scale=1 / 255.0)
    return cv2.bitwise_not(prod)                                         # screen blend


# ---- HUD themes: angelcore and terminal ---------------------------------------------------
ANGEL_WHITE = (250, 247, 245)
ANGEL_DIM = (175, 170, 168)
ANGEL_RED = (45, 35, 235)            # BGR
TERM_ORANGE = (60, 120, 255)         # BGR
TERM_TEXT = [(255, 235, 200), (230, 200, 255), (255, 255, 255), (200, 255, 240)]

# Where things sit, as fractions of the frame (x, y). hud_template.py draws these as guides for your own artwork.
ANGEL_SLOTS = {
    "tagline": (0.50, 0.035),         # tiny tracked caps, centered
    "title": (0.07, 0.125),           # big serif, two lines
    "filename": (0.07, 0.268),         # icon + .exe style tag
    "vertical": (0.945, 0.62),        # rotated text along the right edge (x, bottom y)
    "chat": (0.04, 0.60, 0.62, 0.845),        # x0, y0, x1, y1
    "input": (0.04, 0.88, 0.62, 0.925),
    "date": (0.04, 0.965),
    "timecode": (0.62, 0.965),
    "viewers": (0.64, 0.075),
    "sigil_a": (0.945, 0.76, 34),     # x, y of the center, size (centered on the vertical text's line)
    "sigil_b": (0.07, 0.50, 20),
}


def screen_blend(frame, layer):
    prod = cv2.multiply(cv2.bitwise_not(frame), cv2.bitwise_not(layer), scale=1 / 255.0)
    return cv2.bitwise_not(prod)


def spaced_text(img, text, org, scale, color, thick, spacing, font=None):
    """Text with extra letter-spacing (OpenCV has none). Returns the x where it ended."""
    font = FONT if font is None else font
    x, y = org
    for ch in text:
        put_text(img, ch, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)
        x += hud_text_size(ch, font, scale, thick)[0][0] + spacing
    return x


def vertical_text(img, text, x, y_bottom, scale, color, thick, spacing):
    """Text running upward (rotated 90 degrees counter-clockwise), its baseline starting at (x, y_bottom)."""
    tw = int(sum(hud_text_size(c, FONT, scale, thick)[0][0] + spacing for c in text)) + 8
    th_ = int(40 * scale) + 12
    tmp = np.zeros((th_, tw, 3), np.uint8)
    spaced_text(tmp, text, (2, th_ - 10), scale, color, thick, spacing)
    rot = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = rot.shape[:2]
    x0, y0 = int(x - w / 2), int(y_bottom - h)
    H, W = img.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c > x0c and y1c > y0c:
        img[y0c:y1c, x0c:x1c] = np.maximum(img[y0c:y1c, x0c:x1c], rot[y0c - y0:y1c - y0, x0c - x0:x1c - x0])


def draw_sigil(img, cx, cy, size, color, th, seed, t=0.0):
    """Spiky, mirror-symmetric cross sigil, centered on (cx, cy). Procedural, no fonts.
    Animated: it breathes, each thorn and barb grows and retracts on its own rhythm, and it
    occasionally jitters sideways and leaves a faint ghost copy."""
    if SIGIL_MASKS:
        return draw_custom_sigil(img, cx, cy, size, color, th, seed, t)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * math.pi, 12)
    speeds = rng.uniform(0.8, 2.2, 12)
    lens = rng.uniform(0.35, 0.8, 3)
    rises = rng.uniform(0.1, 0.4, 3)
    barb = rng.uniform(0.15, 0.35, 2)
    A = cv2.LINE_AA

    def one(ox, col):
        sz = size * (1 + 0.05 * math.sin(0.9 * t + phases[0]))
        osc = lambda i: 0.62 + 0.38 * math.sin(speeds[i] * t + phases[i])
        x = cx + ox
        top, bot = cy - sz * 1.1, cy + sz * 1.1                       # centered on cy
        cv2.line(img, (int(x), int(top)), (int(x), int(bot)), col, th, A)
        for j, (frac, half) in enumerate(((0.30, 0.55), (0.64, 0.34))):   # crossbars with barbs
            y = top + (bot - top) * frac
            cv2.line(img, (int(x - sz * half), int(y)), (int(x + sz * half), int(y)), col, th, A)
            for sgn in (-1, 1):
                tip = (x + sgn * sz * (half + 0.12), y - sz * barb[j] * osc(1 + j))
                tri = [(x + sgn * sz * (half - 0.08), y - 1), tip, (x + sgn * sz * half, y + 2)]
                cv2.fillPoly(img, [np.array(tri, np.int32)], col, A)
        for k in range(3):                                               # thorns off the shaft
            y = top + (bot - top) * (0.34 + 0.2 * k)
            length, rise = sz * lens[k] * osc(3 + k), sz * rises[k] * osc(6 + k)
            for sgn in (-1, 1):
                tri = np.array([(x, y - 3), (x + sgn * length, y - rise), (x, y + 3)], np.int32)
                cv2.fillPoly(img, [tri], col, A)
        d = sz * 0.2 * osc(9)                                            # diamond at the top
        cv2.fillPoly(img, [np.array([(x, top - d), (x + d * 0.4, top), (x, top + d), (x - d * 0.4, top)], np.int32)], col, A)

    tick = int(t * 10)
    if np.random.default_rng(tick * 7 + seed).random() < 0.07:           # brief jitter + ghost
        off = size * 0.25 * (1 if tick % 2 else -1)
        one(off, tuple(int(c * 0.35) for c in color))
        one(-off * 0.3, color)
    else:
        one(0, color)


def floating_hearts(layer, t, s, th, x0, x1, y_base, rise, colors):
    """Hearts that float up and fade, between x0 and x1 (pixels), starting at y_base and rising by rise pixels."""
    for k in range(int(t / 0.35) - 8, int(t / 0.35) + 1):
        if k < 0:
            continue
        rng = np.random.default_rng(k + 500)
        age = t - k * 0.35
        if not (0 <= age <= 2.6):
            continue
        a = min(1.0, age / 0.3) * min(1.0, (2.6 - age) / 0.8)
        x = int(x0 + (x1 - x0) * rng.random() + 6 * s * math.sin(age * 3 + k))
        y = int(y_base - age / 2.6 * rise)
        tmp = np.zeros_like(layer)
        col = colors[int(rng.integers(0, len(colors)))]
        draw_heart(tmp, x, y, int((14 + 12 * rng.random()) * s), col, 0 if rng.random() > 0.5 else max(1, th))
        cv2.add(layer, cv2.convertScaleAbs(tmp, alpha=a), dst=layer)


def visible_comments(t, comments):
    """[(k, user, message, alpha, slot)], slot 0 = newest. A new comment every 1.6 s, each lives 6.4 s."""
    interval, life = 1.6, 6.4
    out = []
    if not comments:                                          # --no-chat
        return out
    first = max(0, int((t - life) // interval) + 1)
    for k in range(first, int(t // interval) + 1):
        age = t - k * interval
        if 0 <= age <= life:
            alpha = min(1.0, age / 0.35) * min(1.0, (life - age) / 1.2)
            user, msg = comments[k % len(comments)]
            out.append((k, user, msg, alpha, int(t // interval) - k))
    return out


def lock_on_boxes(layer, boxes, s, t, color, th, body=None):
    if body is not None:                                       # whole-body target: corner brackets only
        cx, cy, bw, bh = body
        x0, y0, x1, y1 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        L = int(min(bw, bh) * 0.08)
        for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(layer, (x, y), (x + dx * L, y), color, th + 1, cv2.LINE_AA)
            cv2.line(layer, (x, y), (x, y + dy * L), color, th + 1, cv2.LINE_AA)
        put_text(layer, "BODY LOCK", (x0 + L + int(8 * s), y0 + int(14 * s)), FONT, 0.38 * s, color, tk(0.38 * s, 1.0), cv2.LINE_AA)
    for i, (cx, cy, bw, bh) in enumerate(boxes):
        side = int(max(bw, bh) * 0.78)
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        cv2.rectangle(layer, (x0, y0), (x0 + side, y0 + side), color, th + 1, cv2.LINE_AA)
        put_text(layer, f"LOCK {i + 1:02d}", (x0, y0 - int(8 * s)), FONT, 0.38 * s, color, tk(0.38 * s, 1.0), cv2.LINE_AA)
        if int(t * 3) % 2 == 0:
            cv2.rectangle(layer, (x0 + side + int(6 * s), y0), (x0 + side + int(12 * s), y0 + int(6 * s)), color, -1)


def glitch_bars(layer, W, H, t, color, n=7):
    rng = np.random.default_rng(int(t * 12) + 31)
    for _ in range(n):
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, int(W * 0.8)))
        ln = int(rng.integers(W // 12, W // 2))
        cv2.line(layer, (x, y), (min(W - 1, x + ln), y), color, 1)


def timecode(t, fps):
    return f"{int(t // 3600):02d}:{int(t // 60) % 60:02d}:{int(t) % 60:02d}:{int((t % 1) * fps):02d}"


def draw_hud_angelcore(frame, t, frac, fps, cfg):
    """Monochrome white, soft glow, serif title, vertical text, sigils, red lock-on boxes, framed chat window."""
    H, W = frame.shape[:2]
    s = H / 720.0
    th = max(1, int(round(s)))
    A = cv2.LINE_AA
    layer = np.zeros_like(frame)
    P = lambda key, i=0: (int(ANGEL_SLOTS[key][i] * W), int(ANGEL_SLOTS[key][i + 1] * H))
    frame = darken_rect(frame, ANGEL_SLOTS["chat"], s)

    # tagline, title (two lines, the second pushed right), filename tag
    tag = cfg.get("tagline", "SERVE / PROTECT / SILENT")
    tw = sum(hud_text_size(c, FONT, 0.5 * s, th)[0][0] + 5 * s for c in tag)
    spaced_text(layer, tag, (P("tagline")[0] - tw / 2, P("tagline")[1]), 0.5 * s, ANGEL_WHITE, tk(0.5 * s, 1.0), 5 * s)
    words = cfg["title"].split()
    lines = [" ".join(words[:max(1, len(words) // 2)]), " ".join(words[max(1, len(words) // 2):])]
    tx, ty = P("title")
    for i, line in enumerate(l for l in lines if l):
        put_text(layer, line, (tx + i * int(0.2 * W), ty + i * int(56 * s * TITLE_SCALE)), cv2.FONT_HERSHEY_COMPLEX, 1.7 * s, ANGEL_WHITE, tk(1.7 * s, 1.5), A)
    fx, fy = P("filename")
    cv2.rectangle(layer, (fx, fy - int(20 * s)), (fx + int(26 * s), fy + int(6 * s)), ANGEL_WHITE, th, A)
    cv2.line(layer, (fx + int(7 * s), fy - int(13 * s)), (fx + int(19 * s), fy), ANGEL_WHITE, th, A)
    spaced_text(layer, cfg.get("tag", "ASCENSION.exe"), (fx + int(40 * s), fy), 0.8 * s, ANGEL_WHITE, tk(0.8 * s, 1.0), 1 * s)

    # vertical text on the right edge, and sigils
    vx, vy = P("vertical")
    vertical_text(layer, cfg.get("vertical", "SYSTEM ANGEL // ONLINE"), vx, vy, 0.62 * s, ANGEL_WHITE, tk(0.62 * s, 1.0), 7 * s)
    if cfg.get("sigils", True):
        for (sx, sy, ssz), seed_, col in ((ANGEL_SLOTS["sigil_a"], 3, ANGEL_WHITE), (ANGEL_SLOTS["sigil_b"], 11, ANGEL_DIM)):
            draw_sigil(layer, int(sx * W), int(sy * H), ssz * s, col, th, seed_, t)

    # viewers (top right)
    vxp, vyp = P("viewers")
    spaced_text(layer, f"{(1200 + 250 * t) / 1000:.1f}K WATCHING", (vxp, vyp), 0.45 * s, ANGEL_WHITE, tk(0.45 * s, 1.0), 3 * s)
    cv2.circle(layer, (vxp - int(14 * s), vyp - int(5 * s)), int(4 * s), ANGEL_RED, -1, A)

    # chat window: thin frame with notched corners, header, fading messages
    cx0, cy0 = P("chat", 0)
    cx1, cy1 = P("chat", 2)
    n = int(14 * s)
    pts = [(cx0 + n, cy0), (cx1, cy0), (cx1, cy1 - n), (cx1 - n, cy1), (cx0, cy1), (cx0, cy0 + n)]
    if cfg.get("chat", True):                                 # frame and header hidden with --no-chat
        cv2.polylines(layer, [np.array(pts, np.int32)], True, ANGEL_WHITE, th, A)
        spaced_text(layer, "// CHAT_LINK", (cx0 + int(16 * s), cy0 + int(24 * s)), 0.42 * s, ANGEL_DIM, tk(0.42 * s, 1.0), 3 * s)
    line_h = int(46 * s)
    for k, user, msg, alpha, slot in visible_comments(t, cfg["comments"]):
        y = cy1 - int(24 * s) - slot * line_h
        if y < cy0 + int(46 * s):
            continue
        tmp = np.zeros_like(frame)
        if cfg.get("sigils", True):
            draw_sigil(tmp, cx0 + int(26 * s), y - int(8 * s), 7 * s, ANGEL_WHITE, th, k, t)
        else:
            cv2.circle(tmp, (cx0 + int(26 * s), y - int(8 * s)), max(2, int(3 * s)), ANGEL_WHITE, -1, A)
        put_text(tmp, user.upper(), (cx0 + int(46 * s), y - int(6 * s)), FONT, 0.46 * s, ANGEL_WHITE, tk(0.46 * s, 1.25), A)
        msg = fit_text(msg, FONT, 0.42 * s, th, cx1 - cx0 - int(70 * s))
        put_text(tmp, msg, (cx0 + int(46 * s), y + int(16 * s)), FONT, 0.42 * s, ANGEL_DIM, tk(0.42 * s, 1.0), A)
        layer = cv2.add(layer, cv2.convertScaleAbs(tmp, alpha=alpha))

    # input line with blinking cursor, date, timecode
    ix0, iy = P("input", 0)
    ix1 = P("input", 2)[0]
    cv2.line(layer, (ix0, iy + int(34 * s)), (ix1, iy + int(34 * s)), ANGEL_WHITE, th, A)
    end = spaced_text(layer, cfg["prompt"].replace("...", ""), (ix0 + int(6 * s), iy + int(22 * s)), 0.45 * s, ANGEL_DIM, tk(0.45 * s, 1.0), 3 * s)
    if int(t * 2) % 2 == 0:
        cv2.rectangle(layer, (int(end) + int(4 * s), iy + int(8 * s)), (int(end) + int(11 * s), iy + int(26 * s)), ANGEL_WHITE, -1)
    if cfg.get("hearts", True):
        draw_heart(layer, ix1 + int(34 * s), iy + int(20 * s), int(26 * s), ANGEL_RED, max(1, th))
        floating_hearts(layer, t, s, th, ix1 + int(8 * s), ix1 + int(0.07 * W), int(0.855 * H), int(0.33 * H), [ANGEL_RED, ANGEL_WHITE])
    spaced_text(layer, cfg.get("date", time.strftime("%d/%m/%Y")), P("date"), 0.5 * s, ANGEL_WHITE, tk(0.5 * s, 1.0), 2 * s)
    spaced_text(layer, timecode(t, fps), P("timecode"), 0.55 * s, ANGEL_WHITE, tk(0.55 * s, 1.0), 2 * s)
    barY = int(0.94 * H)
    cv2.line(layer, (0, barY), (W, barY), ANGEL_DIM, 1, A)
    cv2.line(layer, (0, barY), (int(W * frac), barY), ANGEL_WHITE, th + 1, A)

    glitch_bars(layer, W, H, t, ANGEL_DIM)
    if cfg.get("lockon", True):
        lock = np.zeros_like(frame)
        lock_on_boxes(lock, cfg.get("boxes", []), s, t, ANGEL_RED, th, cfg.get("body_box"))
        layer = cv2.add(layer, lock)
    glow = cv2.GaussianBlur(layer, (0, 0), 5 * s)
    layer = cv2.add(layer, cv2.convertScaleAbs(glow, alpha=0.9))
    return screen_blend(frame, layer)


def fake_code_line(rng):
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return f"{int(rng.integers(0, 65536)):04X}: " + " ".join(f"{int(b):02X}" for b in rng.integers(0, 256, 6))
    if kind == 1:
        return f"{CODE_TOKENS[int(rng.integers(0, 10))]} {CODE_TOKENS[int(rng.integers(10, 14))]}, {CODE_TOKENS[int(rng.integers(10, 14))]}"
    return "".join(rng.choice(list("01"), 28))


def scrolling_panel(layer, rect, t, s, th, seed, speed=5.0, lines_rate=1.0):
    """A bordered panel of scrolling machine-code text, clipped to the panel."""
    x0, y0, x1, y1 = rect
    cv2.rectangle(layer, (x0, y0), (x1, y1), TERM_ORANGE, th, cv2.LINE_AA)
    pw, ph = x1 - x0 - 6, y1 - y0 - 6
    if pw < 20 or ph < 20:
        return
    tmp = np.zeros((ph, pw, 3), np.uint8)
    lh = int(15 * s)
    first = int(t * speed * lines_rate)
    off = int((t * speed * lines_rate - first) * lh)
    for i in range(ph // lh + 2):
        rng = np.random.default_rng(seed * 100003 + first + i)
        col = TERM_TEXT[int(rng.integers(0, len(TERM_TEXT)))]
        put_text(tmp, fake_code_line(rng), (4, ph - i * lh + off), FONT, 0.36 * s, col, tk(0.36 * s, 0.8), cv2.LINE_AA)
    layer[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = np.maximum(layer[y0 + 3:y1 - 3, x0 + 3:x1 - 3], (tmp * 0.8).astype(np.uint8))


def draw_hud_terminal(frame, t, frac, fps, cfg):
    """Dense stack of bordered panels with scrolling machine code, a ticker, and a chat panel."""
    H, W = frame.shape[:2]
    s = H / 720.0
    th = max(1, int(round(s)))
    A = cv2.LINE_AA
    layer = np.zeros_like(frame)
    R = lambda a, b, c, d: (int(a * W), int(b * H), int(c * W), int(d * H))
    frame = darken_rect(frame, (0.02, 0.62, 0.48, 0.86), s)
    scrolling_panel(layer, R(0.0, 0.0, 1.0, 0.026), t, s, th, 1, speed=0, lines_rate=0)       # ticker frame
    tick = " ".join(fake_code_line(np.random.default_rng(i)) for i in range(40))
    off = int(t * 120 * s) % (len(tick) * int(6 * s))
    put_text(layer, tick[off // int(6 * s):off // int(6 * s) + 220], (6, int(0.018 * H)), FONT, 0.34 * s, TERM_TEXT[0], tk(0.34 * s, 0.8), A)
    for i, (y0, y1) in enumerate(((0.045, 0.20), (0.215, 0.37), (0.385, 0.54))):
        scrolling_panel(layer, R(0.70, y0, 0.985, y1), t, s, th, 10 + i, speed=4 + 2 * i)
    scrolling_panel(layer, R(0.50, 0.62, 0.985, 0.86), t, s, th, 20, speed=7)

    # chat panel (bottom left)
    cx0, cy0, cx1, cy1 = R(0.02, 0.62, 0.48, 0.86)
    if cfg.get("chat", True):                                 # frame and header hidden with --no-chat
        cv2.rectangle(layer, (cx0, cy0), (cx1, cy1), TERM_ORANGE, th, A)
        put_text(layer, "CHAT_LINK // LIVE", (cx0 + int(10 * s), cy0 + int(20 * s)), FONT, 0.42 * s, TERM_TEXT[0], tk(0.42 * s, 1.0), A)
    for k, user, msg, alpha, slot in visible_comments(t, cfg["comments"]):
        y = cy1 - int(18 * s) - slot * int(40 * s)
        if y < cy0 + int(40 * s):
            continue
        tmp = np.zeros_like(frame)
        put_text(tmp, f"<{user}>", (cx0 + int(10 * s), y), FONT, 0.46 * s, TERM_TEXT[1], tk(0.46 * s, 1.25), A)
        msg = fit_text(msg, FONT, 0.42 * s, th, cx1 - cx0 - int(24 * s))
        put_text(tmp, msg, (cx0 + int(10 * s), y + int(17 * s)), FONT, 0.42 * s, TERM_TEXT[2], tk(0.42 * s, 1.0), A)
        layer = cv2.add(layer, cv2.convertScaleAbs(tmp, alpha=alpha))

    if cfg.get("hearts", True):
        floating_hearts(layer, t, s, th, int(0.36 * W), int(0.47 * W), int(0.855 * H), int(0.22 * H), [TERM_ORANGE, TERM_TEXT[2]])
    # title, input line, status, progress
    put_text(layer, cfg["title"], (int(0.02 * W), int(0.075 * H)), cv2.FONT_HERSHEY_DUPLEX, 0.9 * s, TERM_TEXT[2], tk(0.9 * s, 1.25), A)
    iy = int(0.895 * H)
    cv2.rectangle(layer, (int(0.02 * W), iy), (int(0.70 * W), iy + int(34 * s)), TERM_ORANGE, th, A)
    cursor = "_" if int(t * 2) % 2 == 0 else " "
    put_text(layer, f"> {cfg['prompt']}{cursor}", (int(0.03 * W), iy + int(23 * s)), FONT, 0.46 * s, TERM_TEXT[0], tk(0.46 * s, 1.0), A)
    put_text(layer, f"{timecode(t, fps)}   {fps:.0f}FPS   SIGNAL {96 + int(3 * abs(math.sin(t * 1.7)))}%",
                (int(0.02 * W), int(0.975 * H)), FONT, 0.4 * s, TERM_TEXT[0], tk(0.4 * s, 1.0), A)
    barY = int(0.945 * H)
    cv2.line(layer, (0, barY), (W, barY), TERM_ORANGE, 1, A)
    cv2.line(layer, (0, barY), (int(W * frac), barY), TERM_TEXT[2], th + 1, A)
    if cfg.get("lockon", True):
        lock = np.zeros_like(frame)
        lock_on_boxes(lock, cfg.get("boxes", []), s, t, TERM_ORANGE, th, cfg.get("body_box"))
        layer = cv2.add(layer, lock)
    layer = cv2.add(layer, cv2.convertScaleAbs(cv2.GaussianBlur(layer, (0, 0), 3 * s), alpha=0.6))
    return screen_blend(frame, layer)


def composite_overlay(frame, overlay, opacity, blend):
    """Your own artwork on top. overlay is BGRA at the frame's size; blend = normal (alpha) or screen."""
    if overlay is None:
        return frame
    a = overlay[..., 3:4].astype(np.float32) / 255.0 * opacity
    rgb = overlay[..., :3].astype(np.float32)
    f = frame.astype(np.float32)
    if blend == "screen":
        mixed = 255 - (255 - f) * (255 - rgb) / 255.0
    else:
        mixed = rgb
    return np.clip(f * (1 - a) + mixed * a, 0, 255).astype(np.uint8)


def load_overlay(path, W, H):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read --overlay {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        print("Note: the overlay has no transparency channel, so it will cover the video. Export it as a PNG with alpha.")
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    if img.shape[1] != W or img.shape[0] != H:
        print(f"Note: resizing the overlay from {img.shape[1]}x{img.shape[0]} to {W}x{H}.")
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


HUD_THEMES = {"eve": None, "angelcore": draw_hud_angelcore, "terminal": draw_hud_terminal}


def ascii_safe(text):
    """OpenCV's built-in fonts only draw plain ASCII, so strip accents and drop emoji instead of showing '???'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


HANDLES = ["neon_fox", "pixelqueen", "static_dream", "moonbeam", "glitchgarden", "nightowl", "violetstorm", "angelrose",
           "lowfi_ghost", "cyber_lamb", "halo_404", "sigil_kid"]
COMMENT_LINE = re.compile(r"^\s*([\w.\-]{1,24})\s*[:>|]\s+(\S.*?)\s*$")


def load_comments(path):
    """One chat message per line. 'name: message' (also 'name> message' or 'name | message') gives that name;
    any other line becomes a message from an auto-named user, so log files work too."""
    if not path:
        return DEFAULT_COMMENTS
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise SystemExit(f"Could not open --comments {path}: {e}")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "cp1252")
    for enc in encodings:
        try:
            lines = raw.decode(enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"Could not decode {path}. Save it as UTF-8 and try again.")
    out = []
    for line in lines:
        if not line.strip():
            continue
        m = COMMENT_LINE.match(line)
        if m:
            out.append((clean_text(m.group(1)), clean_text(m.group(2))))
        else:
            out.append((HANDLES[len(out) % len(HANDLES)], clean_text(line.strip())))
    out = [(u, m) for u, m in out if m]
    if not out:
        print(f"WARNING: {path} had no usable lines (OpenCV text is ASCII only), so the default comments are used.")
        return DEFAULT_COMMENTS
    print(f"Loaded {len(out)} comments from {path}.")
    return out


def fit_text(text, font, scale, thick, max_px):
    """Shorten text with '...' until it fits max_px wide."""
    if hud_text_size(text, font, scale, thick)[0][0] <= max_px:
        return text
    while len(text) > 4 and hud_text_size(text + "...", font, scale, thick)[0][0] > max_px:
        text = text[:-1]
    return text + "..."


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# ---- Main -----------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- extra effects (all off by default)

CHAT_BACKDROP = 0.0        # --chat-backdrop
_BACKDROP_CACHE = {}


def darken_rect(frame, rect, s):
    """Soft dark panel behind the chat so the text stays readable on bright footage. rect = (x0, y0, x1, y1) as fractions."""
    amount = CHAT_BACKDROP
    if amount <= 0:
        return frame
    H, W = frame.shape[:2]
    key = (W, H, rect)
    m = _BACKDROP_CACHE.get(key)
    if m is None:
        sw, sh = max(8, W // 4), max(8, H // 4)
        small = np.zeros((sh, sw), np.float32)
        x0, y0, x1, y1 = (int(rect[0] * sw), int(rect[1] * sh), int(rect[2] * sw), int(rect[3] * sh))
        small[y0:y1, x0:x1] = 1.0
        small = cv2.GaussianBlur(small, (0, 0), max(2.0, 14 * s / 4))
        m = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        _BACKDROP_CACHE.clear()
        _BACKDROP_CACHE[key] = m
    return (frame.astype(np.float32) * (1 - amount * m)).astype(np.uint8)


def apply_echo(frame, mask, state, strength, decay, color, drift):
    """Trails: the person leaves fading ghost copies where they have been. Best with --background black/dim."""
    h, w = frame.shape[:2]
    f = min(1.0, BODY_WORK_WIDTH / w)
    sw, sh = max(8, int(w * f)), max(8, int(h * f))
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_LINEAR)
    trail = state.get("trail")
    if trail is None or trail.shape[:2] != (sh, sw):
        trail = np.zeros_like(small)
    trail = trail * decay
    if drift:
        trail = cv2.warpAffine(trail, np.float32([[1, 0, 0], [0, 1, -drift * f]]), (sw, sh))
    trail = np.maximum(trail, small * m[..., None])
    state["trail"] = trail
    ghost = trail
    if color is not None:
        lum = cv2.cvtColor(np.clip(trail, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        ghost = lum[..., None] * np.array(color, np.float32)[None, None, :] * 1.3
    ghost = cv2.resize(ghost, (w, h), interpolation=cv2.INTER_LINEAR)
    mf = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    g = np.clip(ghost * (1 - mf) * strength, 0, 255)
    out = 255.0 - (255.0 - frame.astype(np.float32)) * (255.0 - g) / 255.0       # screen blend
    return np.clip(out, 0, 255).astype(np.uint8)


def screen_fx(frame, t, split, wobble, scan):
    """Whole-frame signal damage: RGB split (chromatic aberration), wavy/tearing scanlines, CRT scanline darkening.
    split and wobble are in pixels at 720p; they pulse and spike in time."""
    H, W = frame.shape[:2]
    s = H / 720.0
    tick = int(t * 12)
    rng = np.random.default_rng(tick * 31 + 7)
    spike = 1.0 + (2.0 if rng.random() < 0.18 else 0.0)
    out = frame
    if wobble > 0:
        rows = np.arange(H, dtype=np.float32)
        dx = wobble * s * spike * (0.6 * np.sin(rows * 0.045 + t * 6.0) + 0.4 * np.sin(rows * 0.013 - t * 3.1))
        if rng.random() < 0.3:
            y0, hh = int(rng.integers(0, H)), int(rng.integers(max(2, H // 50), max(3, H // 12)))
            dx[y0:y0 + hh] += float(rng.normal()) * wobble * s * 6.0
        map_x = (np.arange(W, dtype=np.float32)[None, :] - dx[:, None]).astype(np.float32)
        map_y = np.repeat(rows[:, None], W, axis=1)
        out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if split > 0:
        d = split * s * (0.5 + 0.5 * abs(math.sin(t * 2.3))) * spike
        if d >= 0.5:
            b, g, r = cv2.split(out)
            r = cv2.warpAffine(r, np.float32([[1, 0, d], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            b = cv2.warpAffine(b, np.float32([[1, 0, -d], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            out = cv2.merge([b, g, r])
    if scan > 0:
        period = max(2, int(3 * s))
        lines = (np.arange(H) + int(t * 20)) % period == 0
        out = out.copy()
        out[lines] = (out[lines].astype(np.float32) * (1 - min(0.9, scan))).astype(np.uint8)
    return out


def draw_halo(layer, box, s, t, color, th, style):
    """Angelic crown (a tilted ring of ticks and orbiting beads above the head) and/or a broken mandala ring behind it."""
    cx, cy, bw, bh = box
    A = cv2.LINE_AA
    if style in ("crown", "both"):
        ccx, ccy = int(cx), int(cy - bh * 0.66)
        a, b = bw * 0.62, bw * 0.17
        cv2.ellipse(layer, (ccx, ccy), (int(a), int(b)), 0, 0, 360, color, th, A)
        cv2.ellipse(layer, (ccx, ccy), (int(a * 0.84), int(b * 0.84)), 0, 0, 360, tuple(int(c * 0.5) for c in color), th, A)
        n = 9
        for i in range(n):                                              # rotating dashes on the outer edge
            a0 = i * 360.0 / n + t * 28
            cv2.ellipse(layer, (ccx, ccy), (int(a * 1.14), int(b * 1.14)), 0, a0, a0 + 360.0 / n * 0.55, color, th + 1, A)
        for i in range(36):                                             # ticks the other way
            ang = math.radians(i * 10 - t * 40)
            p1 = (ccx + a * 0.93 * math.cos(ang), ccy + b * 0.93 * math.sin(ang))
            p2 = (ccx + a * 1.0 * math.cos(ang), ccy + b * 1.0 * math.sin(ang))
            cv2.line(layer, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1, A)
        for k in range(3):                                              # orbiting beads: brighter in front, dimmer behind
            ang = t * 1.6 + k * 2 * math.pi / 3
            bx, by = ccx + a * 1.14 * math.cos(ang), ccy + b * 1.14 * math.sin(ang)
            depth = 0.55 + 0.45 * math.sin(ang)
            cv2.circle(layer, (int(bx), int(by)), max(2, int((2 + 2 * depth) * s)), tuple(int(c * depth) for c in color), -1, A)
        for side in (-1, 1):                                            # a small sigil at each end
            draw_sigil(layer, int(ccx + side * a * 1.42), ccy, 7 * s, color, th, 5 + side, t)
    if style in ("ring", "both"):
        r = max(bw, bh) * 0.95
        rot = t * 22
        for j, (rr, n, gap) in enumerate(((1.0, 7, 0.55), (1.12, 11, 0.35))):
            for i in range(n):
                a0 = i * 360.0 / n + rot * (1 if j == 0 else -1.4)
                cv2.ellipse(layer, (int(cx), int(cy)), (int(r * rr), int(r * rr)), 0, a0, a0 + 360.0 / n * gap, color, th + (1 - j), A)
        for i in range(48):
            ang = math.radians(i * 7.5 + rot)
            ln = 0.07 if i % 4 == 0 else 0.035
            p1 = (cx + r * 1.2 * math.cos(ang), cy + r * 1.2 * math.sin(ang))
            p2 = (cx + r * (1.2 + ln) * math.cos(ang), cy + r * (1.2 + ln) * math.sin(ang))
            cv2.line(layer, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1, A)


def render_halos(frame, boxes, t, style):
    if not boxes:
        return frame
    H, W = frame.shape[:2]
    s = H / 720.0
    th = max(1, int(round(s)))
    layer = np.zeros_like(frame)
    for box in boxes:
        draw_halo(layer, box, s, t, ANGEL_WHITE, th, style)
    glow = cv2.GaussianBlur(layer, (0, 0), 5 * s)
    layer = cv2.add(layer, cv2.convertScaleAbs(glow, alpha=0.9))
    return screen_blend(frame, layer)


def load_sigils(images, font_path, chars):
    """Custom sigil art: image files (white shape on transparent or black, or dark on white) and/or characters from a font."""
    masks = []
    for path in [p.strip() for p in (images or "").split(",") if p.strip()]:
        if not os.path.isfile(path):
            raise SystemExit(f"--sigil-image: file not found: {path} (run from the folder that holds it, or give the full path)")
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"--sigil-image: could not read {path}")
        if im.dtype != np.uint8:                                   # 16-bit PNGs from Photoshop
            im = (im.astype(np.float32) * (255.0 / (65535.0 if im.dtype == np.uint16 else max(1.0, float(im.max()))))).astype(np.uint8)
        if im.ndim == 3 and im.shape[2] == 2:                      # grayscale + alpha
            im = cv2.merge([im[..., 0], im[..., 0], im[..., 0], im[..., 1]])
        if im.ndim == 3 and im.shape[2] == 4 and im[..., 3].min() < 250:
            m = im[..., 3].astype(np.float32)
        else:
            g = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            m = g.astype(np.float32)
            if m.mean() > 127:
                m = 255 - m
        masks.append(m.astype(np.uint8))
    if font_path and chars:
        if not os.path.isfile(font_path):
            raise SystemExit(f"--sigil-font: file not found: {font_path}")
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, 256)
        for ch in chars:
            if ch.isspace():
                continue
            l, t_, r, b = font.getbbox(ch)
            im = Image.new("L", (max(1, r - l + 8), max(1, b - t_ + 8)), 0)
            ImageDraw.Draw(im).text((4 - l, 4 - t_), ch, font=font, fill=255)
            masks.append(np.array(im))
    out = []
    for m in masks:
        ys, xs = np.where(m > 16)
        if len(ys) == 0:
            continue
        out.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out


SIGIL_MASKS = []
SIGIL_SCALE = 1.0   # --sigil-scale (custom sigil art only)
_SIGIL_SCALED = {}


def draw_custom_sigil(img, cx, cy, size, color, th, seed, t):
    n = len(SIGIL_MASKS)
    idx = (seed * 5 + seed // 2) % n
    base = SIGIL_MASKS[idx]
    sz = size * SIGIL_SCALE * (1 + 0.05 * math.sin(0.9 * t + seed))
    h = max(6, int(sz * 2.2))
    key = (idx, h)
    m = _SIGIL_SCALED.get(key)
    if m is None:
        w = max(2, int(base.shape[1] * h / base.shape[0]))
        m = cv2.resize(base, (w, h), interpolation=cv2.INTER_AREA)
        if len(_SIGIL_SCALED) > 200:
            _SIGIL_SCALED.clear()
        _SIGIL_SCALED[key] = m
    pulse = 0.8 + 0.2 * math.sin(1.7 * t + seed)

    def one(ox, col):
        blit_mask(img, m, int(cx + ox - m.shape[1] / 2), int(cy - h / 2), tuple(int(c * pulse) for c in col))

    tick = int(t * 10)
    if np.random.default_rng(tick * 7 + seed).random() < 0.07:
        off = size * 0.25 * (1 if tick % 2 else -1)
        one(off, tuple(int(c * 0.35) for c in color))
        one(-off * 0.3, color)
    else:
        one(0, color)


# ---------------------------------------------------------------- entry points: parse, set up, render
# main() = build_parser() -> setup() -> render_still() or render_video(). A GUI can call the same pieces.

# Module settings that setup() changes. Saved here so every setup() starts from the same defaults,
# even when it runs many times in one process (the GUI preview).
_RESETTABLE = ("GLYPH_SIZE", "SIGIL_SCALE", "TITLE_SCALE", "CHAT_BACKDROP", "SIGIL_MASKS", "HUD_TEXT_SCALE", "cascade", "yunet",
               "HUD_PINK", "HUD_CYAN", "ANGEL_WHITE", "ANGEL_DIM", "ANGEL_RED", "TERM_ORANGE", "TERM_TEXT", "UNICODE_OK",
               "HUD_FONT", "ENOCH")
_DEFAULTS = {name: copy.deepcopy(globals()[name]) for name in _RESETTABLE}


def build_parser():
    """Every command-line option, with its default and help text."""
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--style", choices=["default", "eve"], default="default",
                    help="'eve' = ice-blue look plus digits, smear and particles")
    ap.add_argument("--mode", choices=["blocks", "halftone", "blur", "none"], default="blocks",
                    help="base dissolve: square blocks, a hex lattice of dots, a growing blur, or nothing (use with --ripple or --moire)")
    ap.add_argument("--ripple", action="store_true", help="rings of distortion travel outward from the head's center")
    ap.add_argument("--moire", action="store_true", help="drifting interference bands (a periodic structure) across the head")
    ap.add_argument("--iridescent", action="store_true",
                    help="oil-slick color (blue, violet, cyan, magenta, white) on the head, driven by the ripple's wave phase; added after the color grade")
    ap.add_argument("--irid-palette", choices=sorted(IRID_PALETTES), default="cool",
                    help="cool = blue, violet, white, teal; oil = adds pink and green like an oil slick")
    ap.add_argument("--irid-glitch", type=float, default=0.7, help="0..1: warps the rings with noise and shear, and adds blocks where colors jump (default 0.7)")
    ap.add_argument("--irid-tear", type=float, default=-1, help="0..3: horizontal damage, independent of the rings: bands slide sideways, flash bright or dark, shift color, smear into streaks, split RGB. Default: same as --irid-glitch")
    ap.add_argument("--irid-rings", type=float, default=1.0, help="0..1: how much ring/contour structure (0 = none, so with --irid-tear you get pure glitch with no rings)")
    ap.add_argument("--irid-ring-size", type=float, default=1.0, help="ring spacing multiplier: 0.4 = tight, 1 = default, 2.5 = wide bands")
    ap.add_argument("--irid-strength", type=float, default=0.85, help="how much iridescence replaces the head's own color, 0..1")
    ap.add_argument("--digits", action="store_true", help="character overlay on the head (see --glyphs)")
    ap.add_argument("--glyphs", default="binary",
                    help="what --digits draws. One or more of binary, hex, code, enochian, comma-separated to mix them "
                         "(e.g. code,enochian), or 'all'. hex = bytes like 7F A3; code = MOV EAX 0xFF")
    ap.add_argument("--glyph-size", type=float, default=1.0, help="size of the --digits characters (1 = default, 0.5 = half, 2 = double)")
    ap.add_argument("--hud-text-scale", type=float, default=1.0, help="size of all HUD text (1 = default, 1.3 = 30%% bigger)")
    ap.add_argument("--hud-title-scale", type=float, default=1.0, help="extra size for the big title only (1 = default, 1.5 = 50%% bigger); stacks with --hud-text-scale")
    ap.add_argument("--hud-font", default="", help="path to a .ttf/.otf for the HUD text (default: OpenCV's built-in font)")
    ap.add_argument("--hud-title-font", default="", help="a separate .ttf/.otf just for the big title (default: same as --hud-font)")
    ap.add_argument("--emoji-font", default="", help="colour emoji font for comments (default: Segoe UI Emoji / Noto Color Emoji / Apple Color Emoji if found)")
    ap.add_argument("--chat-backdrop", type=float, default=0.0, help="dark soft panel behind the chat, 0 = off, 0.5 = medium, 0.8 = strong")
    ap.add_argument("--echo", type=float, default=0.0, help="ghost trails behind the moving person (needs a cutout), 0 = off, 0.8 = strong")
    ap.add_argument("--echo-decay", type=float, default=0.88, help="how long trails last: 0.8 short, 0.95 long")
    ap.add_argument("--echo-color", default="", help="tint the trails, e.g. #ff3fa4 (default: the person's own colors)")
    ap.add_argument("--echo-drift", type=float, default=0.0, help="trails float upward this many pixels per frame")
    ap.add_argument("--halo", choices=["off", "crown", "ring", "both"], default="off", help="animated halo above the head (crown) and/or a broken ring behind it")
    ap.add_argument("--rgb-split", type=float, default=0.0, help="whole-frame RGB split in pixels at 720p (try 4-10)")
    ap.add_argument("--wobble", type=float, default=0.0, help="wavy/tearing scanlines in pixels at 720p (try 2-8)")
    ap.add_argument("--scanlines", type=float, default=0.0, help="CRT scanline darkening 0-0.9 (try 0.25)")
    ap.add_argument("--fx-over-hud", action="store_true", help="apply RGB split / wobble / scanlines to the HUD as well")
    ap.add_argument("--sigil-image", default="", help="your own sigil art: one or more PNGs, comma-separated (white shape on transparent or black)")
    ap.add_argument("--sigil-scale", type=float, default=1.0, help="size of your custom sigil art (1 = default, 1.5 = 50%% bigger)")
    ap.add_argument("--sigil-font", default="", help="a font to draw sigils from (use with --sigil-text)")
    ap.add_argument("--sigil-text", default="", help="characters to use as sigils, e.g. 'ᚠᛉ†✝'")
    ap.add_argument("--enochian-font", default="", help="path to an Enochian .ttf/.otf (default: Enochian*.ttf/otf next to the script or in the current folder)")
    ap.add_argument("--target", choices=["head", "body", "both"], default="head",
                    help="head = face tracking (default). body = the whole person, found by background subtraction (fixed camera only). both = body plus head")
    ap.add_argument("--cutout", choices=["bgsub", "neural", "hybrid"], default="bgsub",
                    help="how the person is found for --target body / --background: bgsub = difference from the empty scene (fixed camera); "
                         "neural = a small network finds people (any camera, needs a 6 MB model file); hybrid = both, usually the cleanest")
    ap.add_argument("--seg-model", default="", help="path to the person-segmentation .onnx (default: next to the script)")
    ap.add_argument("--mask-smooth", type=float, default=0.2, help="0..0.9: blend with the previous frame's mask (0 = none, less flicker but more trailing)")
    ap.add_argument("--crop", default="", help="keep only x0,y0,x1,y1 (fractions of the frame, e.g. 0.1,0,0.9,1 trims 10%% off each side); the output is that size")
    ap.add_argument("--bg-image", default="", help="photo of the empty scene, same shape as the video (default: estimated from the video itself)")
    ap.add_argument("--bg-threshold", type=int, default=28, help="how different from the background a pixel must be to count as the person (0..255; lower catches more, including shadows)")
    ap.add_argument("--background", choices=["keep", "black", "color", "dim", "blur", "image"], default="keep",
                    help="subtract the background: keep the person and turn the rest black, a solid color, dimmed, blurred, or a picture "
                         "(uses the same background estimate as --target body, so same fixed-camera rule)")
    ap.add_argument("--bg-color", default="#000000", help="(--background color) #RRGGBB or R,G,B")
    ap.add_argument("--bg-replace", default="", help="(--background image) picture to put behind the person")
    ap.add_argument("--mask-video", default="", help="also write the person mask as a black-and-white video (use it as a matte/alpha in an editor)")
    ap.add_argument("--sigils", dest="sigils", action="store_true", help="(angelcore) show the sigils (the default; only needed to undo --no-sigils in a preset)")
    ap.add_argument("--no-sigils", dest="sigils", action="store_false", help="(angelcore) turn off the animated sigils; chat messages get a plain dot instead")
    ap.set_defaults(sigils=True)
    ap.add_argument("--no-hearts", dest="hearts", action="store_false", help="(angelcore, terminal) turn off the floating hearts and the heart by the input line")
    ap.add_argument("--accent", default="", help="accent color of the HUD (hearts, lock-on boxes, LIVE badge, live dot): #RRGGBB or R,G,B. Defaults: eve pink, angelcore red, terminal orange")
    ap.add_argument("--hud-color", default="", help="main HUD color (text and lines): #RRGGBB or R,G,B. Defaults: white (eve's secondary is ice blue)")
    ap.add_argument("--no-lockon", dest="lockon", action="store_false", help="hide the red lock-on boxes the angelcore and terminal themes draw on the head and body")
    ap.add_argument("--body-style", choices=["fill", "outline"], default="fill",
                    help="fill = effects over the whole silhouette; outline = only a band along its edge")
    ap.add_argument("--theme", choices=sorted(HUD_THEMES), default="eve", help="HUD look: eve (the original), angelcore, terminal")
    ap.add_argument("--hud-tag", default="ASCENSION.exe", help="(angelcore) small tag under the title")
    ap.add_argument("--overlay", default="", help="your own artwork: a PNG with transparency, drawn over the video (resized to fit)")
    ap.add_argument("--overlay-opacity", type=float, default=1.0, help="0..1")
    ap.add_argument("--overlay-blend", choices=["normal", "screen"], default="normal",
                    help="normal = alpha blend; screen = black becomes see-through, good for glowing art")
    ap.add_argument("--smear", action="store_true", help="horizontal pixel-stretch glitch")
    ap.add_argument("--particles", action="store_true", help="particle burst from the head")
    ap.add_argument("--period", type=float, default=4.0, help="seconds per dissolve-and-reform cycle")
    ap.add_argument("--blocks-across", type=int, default=12, help="coarse blocks across the head")
    ap.add_argument("--refresh", type=int, default=3, help="frames between block reshuffles (shimmer)")
    ap.add_argument("--hold", type=int, default=15, help="frames to keep the last head box if detection drops")
    ap.add_argument("--max-seconds", type=float, default=0, help="only process the first N seconds (0 = all)")
    ap.add_argument("--progress", type=float, default=0.7, help="(still images) 0 = clean, 1 = fully broken up")
    ap.add_argument("--no-grade", action="store_true", help="skip the color look")
    ap.add_argument("--min-face", type=float, default=1 / 14, help="smallest face to look for, as a fraction of frame width (default 0.07; try 0.04 for small faces, slower)")
    ap.add_argument("--detect-width", type=int, default=480, help="width the frame is shrunk to for face detection (default 480; 960 or 1280 finds smaller faces but is slower)")
    ap.add_argument("--max-faces", type=int, default=1, help="track and pixelate up to N heads at once")
    ap.add_argument("--stagger", type=float, default=0.0, help="phase offset between faces as a fraction of --period (0.25 = quarter cycle apart)")
    ap.add_argument("--confirm", type=int, default=1, help="detections needed before a new face gets the effect (2 hides one-off false hits)")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar",
                    help="haar = built in, no download. yunet = neural, needs a 230 KB model file (see error message)")
    ap.add_argument("--yunet-model", default="", help="path to the YuNet .onnx file (default: next to this script)")
    ap.add_argument("--yunet-score", type=float, default=0.5, help="YuNet confidence cutoff, 0..1 (lower finds more, 0.4 to 0.6 is typical)")
    ap.add_argument("--region", default="", help="only accept faces centered inside x0,y0,x1,y1 (fractions of the frame, e.g. 0,0,1,0.45 = top 45%%)")
    ap.add_argument("--skin-min", type=float, default=0.0, help="reject faces whose center is less than this fraction skin-colored (0.2 rejects printed or monochrome faces; see --debug for the values)")
    ap.add_argument("--rank", choices=["score", "size"], default="score", help="when more faces are found than --max-faces, keep the most confident or the largest")
    ap.add_argument("--debug", action="store_true", help="draw tracked boxes (green = detected, yellow = held) and every raw detection (white = kept, red = rejected, with the reason and skin value)")
    ap.add_argument("--min-neighbors", type=int, default=5, help="higher = fewer false detections but more misses (try 7 to 9)")
    ap.add_argument("--try-rotations", action="store_true", help="also look for sideways and upside-down heads (slower)")
    ap.add_argument("--scale", type=float, default=1.0, help="process and write at this fraction of the size (0.5 = quick draft)")
    ap.add_argument("--detect-every", type=int, default=1, help="run face detection every N frames (3 is faster; the box is smoothed between)")
    ap.add_argument("--hud", action=argparse.BooleanOptionalAction, default=None,
                    help="live-stream style overlay (on by default with --style eve; --no-hud turns it off)")
    ap.add_argument("--hud-title", default="LIVE STREAM", help="title text at the top left of the HUD")
    ap.add_argument("--comments", default="", help="text file of 'username: message' lines for the comment feed")
    ap.add_argument("--no-chat", dest="chat", action="store_false", help="hide the scrolling comment feed and its backdrop; the rest of the HUD stays")
    ap.add_argument("--hud-time", type=float, default=6.0, help="(still images) which second of the HUD animation to show")
    return ap


def setup(args):
    """Turn parsed options into the settings the renderers use. Also sets this module's look globals
    (colors, fonts, scales, detectors), starting from the defaults each time so repeated calls don't leak."""
    global GLYPH_SIZE, SIGIL_SCALE, TITLE_SCALE, CHAT_BACKDROP, SIGIL_MASKS, HUD_TEXT_SCALE, cascade, yunet, HUD_PINK, HUD_CYAN, ANGEL_WHITE, ANGEL_DIM, ANGEL_RED, TERM_ORANGE
    for name, value in _DEFAULTS.items():
        globals()[name] = copy.deepcopy(value)
    GLYPH_SIZE = max(0.2, args.glyph_size)
    SIGIL_SCALE = max(0.2, args.sigil_scale)
    TITLE_SCALE = max(0.3, args.hud_title_scale)
    CHAT_BACKDROP = min(0.95, max(0.0, args.chat_backdrop)) if args.chat else 0.0
    if args.sigil_image or (args.sigil_font and args.sigil_text):
        SIGIL_MASKS = load_sigils(args.sigil_image, args.sigil_font, args.sigil_text)
        if not SIGIL_MASKS:
            sys.exit("No usable sigil shapes found in --sigil-image / --sigil-text.")
        print(f"Using {len(SIGIL_MASKS)} custom sigil(s)." + ("" if args.theme == "angelcore" or args.halo != "off"
              else " NOTE: sigils only appear in the angelcore theme (--theme angelcore) or on --halo crown/both."))
        if not args.sigils:
            print("NOTE: --no-sigils is on, so your sigil art is hidden. Remove --no-sigils (or add --sigils after it).")
    echo_color = parse_color(args.echo_color) if args.echo_color else None
    HUD_TEXT_SCALE = max(0.3, args.hud_text_scale)
    global UNICODE_OK
    for key, flag in (("body", args.hud_font), ("title", args.hud_title_font), ("emoji", args.emoji_font)):
        if flag:
            if not os.path.isfile(flag):
                sys.exit(f"font file not found: {flag}")
            HUD_FONT[key] = flag
    try:
        import PIL  # noqa: F401
        have_pil = True
    except ImportError:
        have_pil = False
    if (HUD_FONT["body"] or HUD_FONT["title"] or args.comments or args.emoji_font) and not have_pil:
        print("NOTE: Pillow is not installed, so custom fonts, Unicode and emoji are off (pip install pillow).")
    if have_pil:
        HUD_FONT["fallback"] = HUD_FONT["body"] or first_existing(FALLBACK_FONTS)
        HUD_FONT["emoji"] = HUD_FONT["emoji"] or first_existing(EMOJI_FONTS)
        UNICODE_OK = bool(HUD_FONT["fallback"])

    region = None
    if args.region:
        try:
            region = tuple(float(v) for v in args.region.split(","))
            assert len(region) == 4 and region[0] < region[2] and region[1] < region[3]
        except Exception:
            raise SystemExit("--region needs four numbers x0,y0,x1,y1 with x0<x1 and y0<y1, e.g. --region 0,0,1,0.45")
    if args.detector == "yunet":
        yunet = load_yunet(args.yunet_model, args.yunet_score)
    else:
        cascade = load_cascade()
    det_kw = dict(detector=args.detector, region=region, skin_min=args.skin_min, rank=args.rank)
    look = LOOK_EVE if args.style == "eve" else LOOK_DEFAULT
    if args.style == "eve":
        args.digits = args.smear = args.particles = True
    if args.accent:
        HUD_PINK = ANGEL_RED = TERM_ORANGE = parse_color(args.accent)
    if args.hud_color:
        HUD_CYAN = ANGEL_WHITE = parse_color(args.hud_color)
        ANGEL_DIM = tuple(int(c * 0.7) for c in ANGEL_WHITE)
        TERM_TEXT[0] = ANGEL_WHITE
    glyph_styles = parse_glyphs(args.glyphs)
    if "enochian" in glyph_styles:
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise SystemExit("Enochian glyphs need Pillow:  pip install pillow")
        ENOCH["path"] = find_enochian_font(args.enochian_font)
        if not ENOCH["path"]:
            raise SystemExit("No Enochian font found. Put Enochian.otf (or EnochianPlain.ttf) next to this script, or pass --enochian-font path.")
        print("Enochian font:", ENOCH["path"])
    opts = {"blocks_across": args.blocks_across, "digits": args.digits,
            "smear": args.smear, "particles": args.particles,
            "mode": args.mode, "ripple": args.ripple, "moire": args.moire, "t": args.hud_time,
            "glyphs": glyph_styles}
    if args.hud is None:
        args.hud = args.style == "eve" or args.theme != "eve"
    hud_cfg = {"title": clean_text(args.hud_title), "comments": load_comments(args.comments) if args.chat else [], "chat": args.chat, "prompt": "TRANSMIT MESSAGE...",
               "tag": clean_text(args.hud_tag), "boxes": [], "lockon": args.lockon, "sigils": args.sigils, "hearts": args.hearts}
    hud_fn = HUD_THEMES[args.theme] or draw_hud
    irid_kw = dict(strength=args.irid_strength, palette=args.irid_palette, glitch=args.irid_glitch,
                   tear=None if args.irid_tear < 0 else args.irid_tear, rings=args.irid_rings, ring_size=args.irid_ring_size)
    use_head = args.target in ("head", "both")
    use_body = args.target in ("body", "both")
    need_mask = use_body or args.background != "keep" or args.echo > 0
    need_bg = need_mask and args.cutout != "neural"
    if need_mask and args.cutout != "bgsub":
        load_seg_model(args.seg_model)
    crop = None
    if args.crop:
        try:
            crop = tuple(float(v) for v in args.crop.split(","))
            assert len(crop) == 4 and 0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1
        except Exception:
            raise SystemExit("--crop needs x0,y0,x1,y1 as fractions with x0<x1 and y0<y1, e.g. --crop 0.1,0,0.9,1")
    bg_color = None
    if args.background == "color":
        bg_color = parse_color(args.bg_color)
    return SimpleNamespace(look=look, opts=opts, hud_cfg=hud_cfg, hud_fn=hud_fn, irid_kw=irid_kw, det_kw=det_kw,
                           use_head=use_head, use_body=use_body, need_mask=need_mask, need_bg=need_bg,
                           crop=crop, bg_color=bg_color, echo_color=echo_color)


def render_still(frame, args, ctx, bg=None):
    """One finished frame through the still-image pipeline. frame = a BGR image before cropping.
    bg = an optional small background estimate; without it, a bgsub/hybrid cutout reads --bg-image."""
    look, opts, hud_cfg, hud_fn, irid_kw, det_kw = ctx.look, ctx.opts, ctx.hud_cfg, ctx.hud_fn, ctx.irid_kw, ctx.det_kw
    use_head, use_body, need_mask, need_bg, crop, bg_color = ctx.use_head, ctx.use_body, ctx.need_mask, ctx.need_bg, ctx.crop, ctx.bg_color
    bg_fill = None
    oh, ow = frame.shape[:2]
    cx0, cy0, cx1, cy1 = crop_rect(ow, oh, crop)
    frame = frame[cy0:cy1, cx0:cx1]
    binfo = None
    if need_mask:
        if need_bg and bg is None:
            if not args.bg_image:
                raise SystemExit("For a still image, --cutout bgsub/hybrid needs --bg-image (a photo of the empty scene). --cutout neural does not.")
            bgi = cv2.imread(args.bg_image)
            if bgi is None:
                raise SystemExit(f"Could not read --bg-image {args.bg_image}")
            bgi = cv2.resize(bgi, (ow, oh), interpolation=cv2.INTER_AREA)[cy0:cy1, cx0:cx1]
            bg = cv2.resize(bgi, small_size(frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        mask = compute_mask(frame, bg, args.bg_threshold, {}, args.cutout, 0.0)
        if args.background != "keep":
            if args.background == "image":
                bg_fill = cv2.imread(args.bg_replace)
                if bg_fill is None:
                    raise SystemExit("--background image needs --bg-replace picture.jpg")
                bg_fill = cv2.resize(bg_fill, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            frame = replace_background(frame, mask, args.background, bg_color, bg_fill)
    if use_body:
        frame, binfo = apply_body_effect(frame, mask, args.progress, opts, args.hud_time, 1, args.body_style == "outline")
        print("Body found." if binfo else "No person found against the background; try a lower --bg-threshold.")
    boxes = detect_heads(frame, args.min_neighbors, args.try_rotations, args.max_faces, args.min_face, args.detect_width, **det_kw) if use_head else []
    if use_head:
        print(f"Found {len(boxes)} face(s)." if boxes else "No face detected.")
    for k, box in enumerate(boxes):
        frame = apply_head_effect(frame, box, args.progress, opts, seed=1 + 1000 * k, cycle=k)
    if not args.no_grade:
        frame = grade(frame, look)
    if args.iridescent:
        if binfo:
            frame = iridescence_body(frame, binfo, args.progress, args.hud_time, **irid_kw)
        for box in boxes:
            frame = iridescence(frame, box, args.progress, args.hud_time, **irid_kw)
    if args.halo != "off":
        frame = render_halos(frame, boxes, args.hud_time, args.halo)
    fx_on = args.rgb_split > 0 or args.wobble > 0 or args.scanlines > 0
    if fx_on and not args.fx_over_hud:
        frame = screen_fx(frame, args.hud_time, args.rgb_split, args.wobble, args.scanlines)
    hud_cfg["boxes"] = boxes
    hud_cfg["body_box"] = binfo["box"] if binfo else None
    if args.hud:
        frame = hud_fn(frame, args.hud_time, args.progress, 24.0, hud_cfg)
    if args.overlay:
        frame = composite_overlay(frame, load_overlay(args.overlay, frame.shape[1], frame.shape[0]), args.overlay_opacity, args.overlay_blend)
    if fx_on and args.fx_over_hud:
        frame = screen_fx(frame, args.hud_time, args.rgb_split, args.wobble, args.scanlines)
    return frame


def render_video(args, ctx):
    """The whole video, frame by frame, written to args.output."""
    look, opts, hud_cfg, hud_fn, irid_kw, det_kw = ctx.look, ctx.opts, ctx.hud_cfg, ctx.hud_fn, ctx.irid_kw, ctx.det_kw
    use_head, use_body, need_mask, need_bg, crop, bg_color = ctx.use_head, ctx.use_body, ctx.need_mask, ctx.need_bg, ctx.crop, ctx.bg_color
    echo_color = ctx.echo_color
    bg_fill = overlay = None
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    FW, FH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0, cx1, cy1 = crop_rect(FW, FH, crop)
    W0, H0 = cx1 - cx0, cy1 - cy0                                      # size after cropping
    scale = min(max(args.scale, 0.1), 1.0)
    W, H = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2         # mp4 wants even dimensions
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"Could not open {args.output} for writing (is it open in another program, "
                         "or is the folder read-only?). Nothing was processed.")
    limit = int(args.max_seconds * fps) if args.max_seconds else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if limit:
        total = min(total, limit)
    print(f"Input {W0}x{H0} at {fps:.1f} fps, {total} frames; writing {W}x{H} to {args.output}")

    bg_small, mask_state, mask_writer = None, {}, None
    echo_state = {}
    if need_bg:
        print("Estimating the background (needs a fixed camera)...")
        bg_small = build_background(args.input, limit, args.bg_image, crop, args.bg_threshold)
    if args.background == "image":
        bg_fill = cv2.imread(args.bg_replace)
        if bg_fill is None:
            raise SystemExit("--background image needs --bg-replace picture.jpg")
        bg_fill = cv2.resize(bg_fill, (W, H), interpolation=cv2.INTER_AREA)
    if args.mask_video:
        if not need_mask:
            raise SystemExit("--mask-video needs --target body (or both) or a --background mode, which compute the mask.")
        mask_writer = cv2.VideoWriter(args.mask_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if args.overlay:
        overlay = load_overlay(args.overlay, W, H)
    names = ["detect", "head effect", "color grade", "HUD", "read+write"]
    spent = dict.fromkeys(names, 0.0)
    tracker = FaceTracker(args.max_faces, args.hold, args.confirm)
    detected, idx = 0, 0
    t_start = last_log = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or (limit and idx >= limit):
            break
        if crop is not None:
            frame = frame[cy0:cy1, cx0:cx1]
        if (W, H) != (W0, H0):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        mask = None
        if need_mask:
            mask = compute_mask(frame, bg_small, args.bg_threshold, mask_state, args.cutout, args.mask_smooth)
            if mask_writer is not None:
                mask_writer.write(cv2.cvtColor((np.clip(mask, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            if args.background != "keep":
                frame = replace_background(frame, mask, args.background, bg_color, bg_fill)
        spent["read+write"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if use_head and idx % max(1, args.detect_every) == 0:
            dets = detect_heads(frame, args.min_neighbors, args.try_rotations, args.max_faces, args.min_face, args.detect_width, **det_kw)
            detected += bool(dets)
            tracker.update(dets)
        spent["detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        t = idx / fps
        opts["t"] = t
        progs, binfo = [], None
        if use_body:
            bprog = 0.5 - 0.5 * math.cos(2 * math.pi * (t / args.period))
            frame, binfo = apply_body_effect(frame, mask, bprog, opts, t, idx // args.refresh, args.body_style == "outline")
        for track in (tracker.active() if use_head else []):
            progress = 0.5 - 0.5 * math.cos(2 * math.pi * (t / args.period + track["id"] * args.stagger))
            progs.append((track["box"], progress))
            frame = apply_head_effect(frame, track["box"], progress, opts,
                                      idx // args.refresh + 1000 * track["id"], int(t // args.period) + track["id"])
        if args.echo > 0 and mask is not None:
            frame = apply_echo(frame, mask, echo_state, args.echo, args.echo_decay, echo_color, args.echo_drift)
        spent["head effect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if not args.no_grade:
            frame = grade(frame, look)
        if args.iridescent:
            if binfo:
                frame = iridescence_body(frame, binfo, bprog, t, **irid_kw)
            for box, progress in progs:
                frame = iridescence(frame, box, progress, t, **irid_kw)
        spent["color grade"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if args.halo != "off":
            frame = render_halos(frame, [pb for pb, _ in progs], t, args.halo)
        fx_on = args.rgb_split > 0 or args.wobble > 0 or args.scanlines > 0
        if fx_on and not args.fx_over_hud:
            frame = screen_fx(frame, t, args.rgb_split, args.wobble, args.scanlines)
        hud_cfg["boxes"] = [pb for pb, _ in progs]
        hud_cfg["body_box"] = binfo["box"] if binfo else None
        if args.hud:
            frame = hud_fn(frame, idx / fps, idx / total, fps, hud_cfg)
        if overlay is not None:
            frame = composite_overlay(frame, overlay, args.overlay_opacity, args.overlay_blend)
        if fx_on and args.fx_over_hud:
            frame = screen_fx(frame, t, args.rgb_split, args.wobble, args.scanlines)
        spent["HUD"] += time.perf_counter() - t0
        if args.debug:
            for (cx, cy, bw, bh), kept, note in last_candidates:     # thin boxes: every raw detection and why it was kept or not
                color = (255, 255, 255) if kept else (0, 0, 255)
                cv2.rectangle(frame, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), color, 1)
                cv2.putText(frame, note, (int(cx - bw / 2), min(H - 6, int(cy + bh / 2) + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            for track in tracker.tracks:
                cx, cy, bw, bh = track["box"]
                shown = track in tracker.active()
                color = (0, 255, 0) if track["fresh"] else (0, 220, 255)
                if not shown:
                    color = (160, 160, 160)           # gray = not confirmed yet
                cv2.rectangle(frame, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), color, 2)
                cv2.putText(frame, f"#{track['id']} hits={track['hits']} missed={track['missed']}",
                            (int(cx - bw / 2), max(12, int(cy - bh / 2) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        t0 = time.perf_counter()
        writer.write(frame)
        spent["read+write"] += time.perf_counter() - t0
        idx += 1

        now = time.perf_counter()
        if idx == 1 or now - last_log >= 2.0:
            elapsed = now - t_start
            rate = idx / elapsed
            print(f"\rframe {idx}/{total} ({100 * idx / total:.0f}%) | {rate:.1f} fps | "
                  f"elapsed {fmt_time(elapsed)} | ETA {fmt_time((total - idx) / rate)}   ", end="", flush=True)
            last_log = now

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    elapsed = time.perf_counter() - t_start
    size_mb = os.path.getsize(args.output) / 1e6 if os.path.exists(args.output) else 0
    print(f"\nDone: {idx} frames in {fmt_time(elapsed)}; wrote {args.output} ({size_mb:.1f} MB); "
          f"a head was found in {detected} detection runs.")
    if idx:
        print("Average ms per frame: " + ", ".join(f"{n} {1000 * spent[n] / idx:.0f}" for n in names))
    if size_mb == 0:
        print("WARNING: the output file is empty. The video writer failed; tell me your OpenCV version.")
    if idx and detected == 0 and use_head:
        print("No head was detected. Try frontal, well-lit footage, or swap in a better detector.")


def main():
    args = build_parser().parse_args()
    ctx = setup(args)
    if os.path.splitext(args.input)[1].lower() in IMAGE_EXTS:
        frame = cv2.imread(args.input)
        if frame is None:
            raise SystemExit(f"Could not read {args.input}")
        cv2.imwrite(args.output, render_still(frame, args, ctx))
        print("Wrote", args.output)
        return
    render_video(args, ctx)


if __name__ == "__main__":
    main()
