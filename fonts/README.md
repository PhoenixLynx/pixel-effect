# fonts

Put font files here (`.ttf`, `.otf`, `.ttc`) that you want to use without installing them in Windows.

They show up by name in the GUI's font dropdowns, and work by name on the command line too:

```bash
python pixel_effect.py in.mp4 out.mp4 --hud-font "My Downloaded Font"
```

Installed fonts work the same way: `--hud-font "Segoe UI Bold"`. A full file path still works too.

Font files in this folder are not committed to git (see `.gitignore`), because most fonts' licenses
don't allow redistributing them.
