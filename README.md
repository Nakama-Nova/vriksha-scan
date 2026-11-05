# Vriksha Studios — scan → CNC pipeline

Temple relief scan (photogrammetry mesh) → 3-axis G-code. Python 3.10+ (CI runs 3.12, on Linux and Windows).

This repository has two tools:

- **`vriksha.py`**: the converter, mesh or depth image to G-code. Runs on its own.
- **`workshop_worker.py`**: the workshop-PC worker. It takes scans that staff upload from the phone app ([vriksha-mobile](https://github.com/Nakama-Nova/vriksha-mobile)) to the shop server ([nakamadesk-backend](https://github.com/Nakama-Nova/nakamadesk-backend)), builds the 3D model and G-code, and uploads the result.

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

Other options: `--worker NAME` (a name for this PC, default `workshop-pc`), `--work-dir DIR` (scratch space, default a `vriksha-worker` folder in the system temp), `--keep` (keep each job's working files), `--colmap` / `--ffmpeg` (tool paths), `--mode auto|full|relief`. `--url` and `--user` can also come from the `VRIKSHA_URL` and `VRIKSHA_USER` environment variables.

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

## Run it by itself (Windows)

The worker has to run on a workshop PC: the full 3D route needs that PC's NVIDIA GPU and COLMAP, and flat carvings need a ~1 GB depth model, so it cannot run on Vercel (short-lived functions, no GPU). Instead make the PC start it for you:

```powershell
# once, in this folder
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-relief.txt   # drop the second file if flat carvings are not wanted
.venv\Scripts\python workshop_worker.py --url https://api.vrikshastudios.com --user workshop-pc --save-password
.\install-autostart.ps1 -Url https://api.vrikshastudios.com -User workshop-pc
```

- `--save-password` asks for the password (nothing shows as you type), signs in to check it, and keeps it in the **Windows Credential Manager** under your Windows user. It is never written to a file, a script or the scheduled task. A wrong password is not saved. `--forget-password` removes it.
- `install-autostart.ps1` registers a scheduled task "Vriksha Scan Worker": it starts when this Windows user signs in, runs hidden, restarts a minute after a crash, and never runs two copies. `.\install-autostart.ps1 -Remove` undoes it; `-DryRun` shows what it would do.
- **Which account decides what the worker may process** (the server checks this, not the worker): an **owner** account can process everything; a **manager** account only product scans (furniture and other products); an **artisan** account only restoration and sculpture scans. So a PC that should do furniture carvings needs an owner or manager account, and one for temple restorations an owner or artisan account. Give the worker its own login (for example `workshop-pc`), not a person's.
- The log is `%LOCALAPPDATA%\VrikshaScan\worker.log` (about 1 MB, three older files kept). Status: `Get-ScheduledTask -TaskName 'Vriksha Scan Worker' | Get-ScheduledTaskInfo`.
- Set the PC to sign in by itself after a restart and not to sleep while plugged in; while it is asleep or signed out, scans just wait.
- `--unattended` (what the task uses) never asks for a password, keeps trying to sign in if the network is not up yet, and keeps going after an unexpected error. If the password was changed or the account is locked or switched off, it tries again only once an hour: the server locks an account after 5 failed sign-ins in 15 minutes, and that lock would also stop the same person using the desktop app. After changing the account's password, run `--save-password` again.

## Tests and CI

`python -m pytest` runs `test_vriksha.py` (the converter, on a synthetic dome) and `test_worker.py`. CI (`.github/workflows/ci.yml`) does the same on Linux and Windows, plus flake8 for syntax errors and undefined names only. COLMAP itself is not tested in CI.

## Licence

Proprietary software. Copyright (c) 2026 Nakama Nova, all rights reserved. It may not be copied, modified or shared; see `LICENSE`.
