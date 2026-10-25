@echo off
rem ======================================================================
rem  Preset: angelcore. 
rem  Same layout as presets_template.bat: every option is listed, with
rem  this preset's values. Change a value or paste in a listed switch.
rem      render.bat angelcore input.mp4 output.mp4 [draft]
rem ======================================================================

set SCRIPT=pixel_effect.py

rem ---- LOOK: overall style, the base dissolve, and the color look
rem   --style   default | eve   (eve = ice-blue look plus digits, smear, particles)
rem   --mode    blocks | halftone | blur | none
rem   --look    auto | default | eve | rose | gold | teal | ember | vapor | noir   (auto = eve with --style eve)
rem             camera looks: cctv | vhs | nightshot | thermal | thermal-iron
rem             graphic looks: tactical | chrome | lilac | acid      modern looks: neon | editorial | redshift
rem   --tone-shadows / --tone-mids / --tone-highlights   your own colors, dark to light, #RRGGBB ("" = the look's)
rem   --tone-strength  0..1 how much the tone replaces the original colors (-1 = the look's own)
rem   --wash           a flat color over everything, #RRGGBB ("" = none)
rem   --wash-opacity   0..1     --wash-blend  normal | multiply | screen | overlay | softlight
rem   optional, add to the line to override the look:  --hue-shift 20  --saturation 1.2  --bloom 0.3  --chroma-shift 2
rem   switches: --no-grade   (skip the color look)
set LOOK=--style eve --mode blocks --look auto --tone-shadows "" --tone-mids "" --tone-highlights "" --tone-strength -1 --wash "" --wash-opacity 0.3 --wash-blend softlight

rem ---- HUD: the live-stream overlay
rem   --theme       eve | angelcore | terminal | security | vhs | thermal | military | scifi | y2k | modern
rem   --accent      hearts, lock-on boxes, LIVE badge: #RRGGBB or R,G,B ("" = theme default)
rem   --hud-color   main HUD text and lines: #RRGGBB or R,G,B ("" = theme default)
rem   --hud-tag     (angelcore) small tag under the title
rem   --hud-tagline (angelcore) small spaced caps across the top
rem   --hud-vertical (angelcore) text running up the right edge
rem   --hud-time    (still images) which second of the HUD animation to show
rem   HUD on/off:   left out = automatic (on with --style eve or a non-eve theme)
rem   --layout      a layout file (JSON) that moves or hides HUD elements ("" = the theme's own layout)
rem   --hud-clock   date and time the security/vhs clocks start at, e.g. "1988-09-17 17:14" ("" = now)
rem   switches: --hud  --no-hud  --no-lockon  --no-hearts  --no-sigils
set HUD=--theme angelcore --accent "#ff3fa4" --hud-color "#cfe8ff" --hud-title "LIVE STREAM" --hud-tag "ASCENSION.exe" --hud-tagline "SERVE / PROTECT / SILENT" --hud-vertical "SYSTEM ANGEL // ONLINE" --hud-time 6.0 --hud-clock "" --layout "" --no-lockon

rem ---- FONTS: HUD text ("" = OpenCV's built-in font; title "" = same as --hud-font)
rem   any font option takes an installed font's name ("Segoe UI Bold") or a file path
rem   --hud-text-scale   all HUD text (1.3 = 30% bigger)
rem   --hud-title-scale  extra size for the big title only; stacks with the above
set FONTS=--hud-font "C:\Windows\Fonts\segoeui.ttf" --hud-title-font "C:\Windows\Fonts\MATURASC.ttf" --emoji-font "" --hud-text-scale 1.0 --hud-title-scale 1.0

rem ---- CHAT: comment feed
rem   --comments        text file of "username: message" lines ("" = none)
rem   --chat-backdrop   dark panel behind the chat: 0 off, 0.5 medium, 0.8 strong
rem   switches: --no-chat   (hide the comment feed and its backdrop; the rest of the HUD stays)
set CHAT=--comments "%~dp0comments.txt" --chat-backdrop 0.6

rem ---- PIXEL: the dissolve-and-reform cycle on the head
rem   --period         seconds per cycle
rem   --blocks-across  coarse blocks across the head
rem   --refresh        frames between block reshuffles (shimmer)
rem   --progress       (still images) 0 clean .. 1 fully broken up
rem   switches: --ripple  --moire  --smear  --particles
set PIXEL=--period 4.0 --blocks-across 12 --refresh 3 --progress 0.7

rem ---- IRID: oil-slick iridescence (only active with --iridescent)
rem   --irid-palette    cool | oil
rem   --irid-strength   0..1 how much it replaces the head's color
rem   --irid-glitch     0..1 warps rings, adds blocks
rem   --irid-tear       0..3 horizontal damage (-1 = same as --irid-glitch)
rem   --irid-rings      0..1 ring structure (0 = none)
rem   --irid-ring-size  0.4 tight, 1 default, 2.5 wide
rem   switches: --iridescent   (turns it on)
set IRID=--irid-palette cool --irid-strength 0.85 --irid-glitch 0 --irid-tear 2.5 --irid-rings 0 --irid-ring-size 1.0 --iridescent

rem ---- GLYPHS: characters over the head (only active with --digits)
rem   --glyphs   binary, hex, code, enochian, comma-separated (e.g. code,enochian), or all
rem   --enochian-font  "" = Enochian*.ttf/otf next to the script
rem   switches: --digits   (turns it on)
set GLYPHS=--glyphs enochian --glyph-size 0.5 --enochian-font "" --digits

rem ---- ECHO: ghost trails behind the person (needs a cutout, see BODY)
rem   --echo        0 off, 0.8 strong
rem   --echo-decay  0.8 short .. 0.95 long
rem   --echo-color  #RRGGBB tint ("" = the person's own colors)
rem   --echo-drift  pixels per frame the trails float upward
set ECHO=--echo 0.9 --echo-decay 0.92 --echo-color "#ff3fa4" --echo-drift 2

rem ---- HALO
rem   --halo   off | crown | ring | both
set HALO=--halo crown

rem ---- DAMAGE: whole-frame glitch (pixels at 720p)
rem   --rgb-split  try 4-10    --wobble  try 2-8    --scanlines  0-0.9, try 0.25
rem   --tracking   VHS tracking band and tear 0..1    --bleed  VHS color smear, try 4-8
rem   --dropout    lost-signal flashes 0..1 (0.3 = now and then), showing --dropout-text
rem   switches: --fx-over-hud   (damage the HUD too)
set DAMAGE=--rgb-split 3 --wobble 1.5 --scanlines 0.25 --tracking 0.0 --bleed 0.0 --dropout 0.0 --dropout-text "NO SIGNAL"

rem ---- CAMERA: what the camera itself adds (pixels at 720p)
rem   --fps-stutter  choppy frame rate in frames per second, like CCTV (try 8-12; 0 = smooth)
rem   --shake        handheld shake, try 3-8      --fisheye   wide-lens bulge 0..1, try 0.3
rem   --vignette     darker corners 0..1          --noise     sensor/tape grain 0..1
set CAMERA=--fps-stutter 0.0 --shake 0.0 --fisheye 0.0 --vignette 0.0 --noise 0.0

rem ---- SIGIL: your own sigil art for angelcore
rem   --sigil-image  one or more PNGs, comma-separated
rem   --sigil-font + --sigil-text   draw sigils from font characters
set SIGIL=--sigil-image "" --sigil-scale 1.0 --sigil-font "" --sigil-text ""

rem ---- BODY: what gets the effect, and how the person is cut out
rem   --target      head | body | both
rem   --body-style  fill | outline
rem   --cutout      bgsub (fixed camera) | neural (any camera) | hybrid (cleanest)
rem   --mask-smooth 0..0.9 blend with the previous mask (less flicker, more trailing)
rem   --mask-video  also write the person mask as a matte video ("" = no)
set BODY=--target both --body-style fill --cutout hybrid --seg-model "" --mask-smooth 0.2 --mask-video ""

rem ---- BG: what happens behind the person
rem   --background    keep | black | color | dim | blur | image
rem   --bg-color      for "color": #RRGGBB or R,G,B
rem   --bg-replace    for "image": picture to put behind the person
rem   --bg-image      photo of the empty scene ("" = estimated from the video)
rem   --bg-threshold  0..255, lower catches more (including shadows)
set BG=--background black --bg-color "#000000" --bg-replace "" --bg-image "" --bg-threshold 28

rem ---- OVERLAY: your own artwork over the video: a PNG, or a video that plays in sync and loops
rem   --overlay-blend  normal | screen (black becomes see-through) | add (brighter, for light leaks)
rem   video overlays have no transparency: use screen/add for art on black, or --overlay-key for one color
rem   --overlay-start  seconds into the overlay video to begin
rem   --overlay-key    #RRGGBB that becomes see-through (e.g. #00ff00 green screen; "" = none)
rem   --overlay-key-tolerance  0..255 how close to the key counts
set OVERLAY=--overlay "" --overlay-opacity 1.0 --overlay-blend normal --overlay-start 0.0 --overlay-key "" --overlay-key-tolerance 60

rem ---- CROP: keep only x0,y0,x1,y1 as fractions ("" = whole frame)
set CROP=--crop 0.05,0.02,0.95,0.98

rem ---- FACE: finding and tracking heads
rem   --detector      haar (built in) | yunet (needs the .onnx model)
rem   --rank          score | size   (which faces win when there are too many)
rem   --region        only faces centered in x0,y0,x1,y1 ("" = anywhere)
rem   --skin-min      0.2 rejects printed or monochrome faces
rem   --min-face      smallest face as a fraction of frame width (0.0714286 = 1/14)
rem   --detect-every  run detection every N frames (3 is faster)
rem   --confirm       detections needed before a new face gets the effect
rem   --hold          frames to keep the last box if detection drops
rem   --stagger       phase offset between faces, fraction of --period
rem   --min-neighbors (haar) higher = fewer false hits, more misses
rem   switches: --try-rotations  --debug
set FACE=--detector yunet --yunet-model "" --yunet-score 0.4 --min-neighbors 5 --min-face 0.0714286 --detect-width 480 --detect-every 1 --max-faces 1 --rank size --confirm 1 --hold 15 --stagger 0.0 --region 0,0,1,0.45 --skin-min 0.2

rem ---- SPEED: FULL is used normally; DRAFT replaces it when you add "draft"
rem   --scale        fraction of full size (0.25 = quick draft)
rem   --max-seconds  only the first N seconds (0 = all)
set FULL=--scale 1.0 --max-seconds 0
set DRAFT=--scale 0.25 --max-seconds 4

echo preset angelcore loaded.
