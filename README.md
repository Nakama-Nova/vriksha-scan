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

`workshop_worker.py` runs on the GPU PC. It signs in to the Vriksha server as the owner or an
artisan account, takes the oldest uploaded scan, builds a 3D model, exports `.glb .stl .obj .ply`,
makes `.nc` G-code and a preview when the scan asked for it, uploads everything and finishes the job.
Restoration scans end up as private files on the restoration project (owner and artisans only).

```bash
pip install -r requirements.txt
set VRIKSHA_PASSWORD=...            # or it asks; never put it in a file
python workshop_worker.py --url https://your-server --user workshop-pc          # keeps running, checks every 30 s
python workshop_worker.py --url https://your-server --user workshop-pc --once   # one pass, then stops
```

Needs on the PC: **COLMAP** with CUDA (the dense step needs an NVIDIA GPU) and, for video scans,
**ffmpeg**, on PATH or passed with `--colmap` / `--ffmpeg`. The worker calls
`colmap automatic_reconstructor --data_type individual --quality low|medium|high --dense 1 --mesher poisson`
and reads `dense/0/meshed-poisson.ply`; if your COLMAP build words an option differently, run
`colmap automatic_reconstructor -h` and adjust `colmap_command()`.

- The widest side of the model is set to the real width entered on the scan (mm). G-code needs that width.
- A model larger than the server accepts is simplified, the same way for every format, until each file fits.
- A failed job reports why on the scan (for example "too few overlapping photos") so it can be retaken.
- The worker renews its claim while COLMAP runs; if the PC dies, the server gives the job to another try (3 in all).
- `pytest test_worker.py` runs the whole job with a fake server and a synthetic mesh (no GPU needed).

## Licence

Proprietary software. Copyright (c) 2026 Nakama Nova, all rights reserved. It may not be copied, modified or shared; see `LICENSE`.
