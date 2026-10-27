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

## Workshop worker (photos or video -> 3D model -> G-code, automatically)

`workshop_worker.py` runs on a workshop PC (any PC; a GPU is only needed for full 3D). It signs in to the Vriksha server as the owner or an
artisan account, takes the oldest uploaded scan, builds a 3D model, exports `.glb .stl .obj .ply`,
makes `.nc` G-code and a preview when the scan asked for it, uploads everything and finishes the job.
Restoration scans end up as private files on the restoration project (owner and artisans only).

```bash
pip install -r requirements.txt
python workshop_worker.py --url https://your-server --user workshop-pc          # keeps running, checks every 30 s
python workshop_worker.py --url https://your-server --user workshop-pc --once   # one pass, then stops
```

First create the sign-in in the desktop app: Staff, Add staff, role **Artisan (restoration)** (or use the
owner). A manager cannot run restoration scans. The worker asks for the password when it starts; just type
it there (nothing shows as you type). Do not paste it into chat or a file. If you prefer an environment
variable, the syntax depends on the shell: `set VRIKSHA_PASSWORD=...` in cmd, `$env:VRIKSHA_PASSWORD='...'` in
PowerShell, `export VRIKSHA_PASSWORD='...'` in Git Bash (single quotes: a `!` in double quotes or unquoted
is expanded by bash history).

The worker looks at the PC it runs on and builds what that PC can. It checks again on every pass, so
installing a tool or driver while it runs is noticed. A scan it cannot build is **left waiting** (never
failed) for a PC that can.

| Scan shape | What it is | Needs on the PC |
|---|---|---|
| **Flat carving** (`relief`) | A flat or shallow carving read from **one straight-on photo**; if several are sent the sharpest is used. A depth model works out the relief; you get `.glb .stl .obj .ply`, a depth picture and, if asked, G-code. | Any PC, no GPU: `pip install -r requirements-relief.txt` (torch and transformers, about 1 GB, plus a ~100 MB model on the first run). A minute or two on a normal CPU. |
| **Full 3D object** (`full`) | Photos (8 or more) or a video walked all the way around. COLMAP builds a true 3D mesh. | **COLMAP with CUDA** and an **NVIDIA GPU**; for video also **ffmpeg**. Pass `--colmap` / `--ffmpeg` if they are not on PATH. |

When it starts it prints which of the two this PC can do. `--mode relief` or `--mode full` forces one.
The full route calls
`colmap automatic_reconstructor --data_type individual --quality low|medium|high --dense 1 --mesher poisson`
and reads `dense/0/meshed-poisson.ply`; if your COLMAP build words an option differently, run
`colmap automatic_reconstructor -h` and adjust `colmap_command()`.

The flat-carving depth is a model's guess: relative, front face only, no undercuts, and its size comes from the
real width entered on the scan (without one the model is marked "size not set"). Use the full route when the
true shape matters.

- The widest side of the model is set to the real width entered on the scan (mm). G-code needs that width.
- A model larger than the server accepts is simplified, the same way for every format, until each file fits.
- A failed job reports why on the scan (for example "too few overlapping photos") so it can be retaken.
- The worker renews its claim while COLMAP runs; if the PC dies, the server gives the job to another try (3 in all).
- `pytest test_worker.py` runs the whole job with a fake server and a synthetic mesh (no GPU needed).

## Licence

Proprietary software. Copyright (c) 2026 Nakama Nova, all rights reserved. It may not be copied, modified or shared; see `LICENSE`.
