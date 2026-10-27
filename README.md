# Vriksha Studios — scan → CNC pipeline

Temple relief scan (photogrammetry mesh) → 3-axis G-code. Python 3.10+.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin on Linux/Mac
.venv/Scripts/python vriksha.py scan.ply --width 600 --max-depth 20 -o panel.nc
```

Open `preview.png` first: it shows what the finish cutter will leave. If the relief is upside down, add `--flip`.

What it does: keeps the biggest mesh piece → auto-orients (thinnest axis = Z) → scales to `--width` mm →
top-down height map → rough pass (flat end mill, `--stepdown` layers, `--allowance` left) → finish pass (ball nose, gouge-free).
G-code frame: X0 Y0 = lower-left, Z0 = top of relief. `--dialect generic|mach3|grbl` (tool-change lines only for now).

**Not for real cutting yet.** Test in air, then foam, then scrap wood. Set feeds/rpm for your machine and tool.

Known limits: 3-axis only (no undercuts); rough pass retracts every run; hole fill is 3 mm; no machine-specific post-processor.
Test: `pytest` (synthetic dome, random orientation).

## From a single photo (approximate)
```bash
pip install torch transformers          # optional, ~1 GB; downloads a ~100 MB depth model on first run
python photo_to_depth.py figure.jpg figure_depth.png
python vriksha.py figure_depth.png --width 150 --max-depth 12 --floor 0.3
```
A model guesses the depth: relative, front face only, not true to scale. Raise `--floor` to flatten more of the background.
For accurate geometry use a photogrammetry mesh instead.

## Licence

Proprietary software. Copyright (c) 2026 Nakama Nova, all rights reserved. It may not be copied, modified or shared; see `LICENSE`.
