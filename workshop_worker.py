"""Workshop PC worker: scan queue -> photos/video -> 3D model -> CNC G-code -> back to the server.

Runs on the GPU PC. It signs in to the Vriksha server (as the owner or an artisan account),
takes the oldest uploaded scan, builds a 3D model with COLMAP, exports .glb .stl .obj .ply,
makes G-code with vriksha.py when the scan asked for it, uploads everything and finishes
the job. The server only hands out signed links, so this PC holds no cloud keys.

    python workshop_worker.py --url https://your-server --user workshop-pc          # keeps running
    python workshop_worker.py --url https://your-server --user workshop-pc --once   # one pass, then stop

The password comes from VRIKSHA_PASSWORD or a prompt. Needs COLMAP (with CUDA for the dense
step) and, for video scans, ffmpeg, both on PATH or given with --colmap / --ffmpeg.
"""

import argparse
import collections
import getpass
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import trimesh

import relief
import vriksha

http = requests  # replaced in tests

POLL_SECONDS = 30
HEARTBEAT_SECONDS = 60  # the server gives a job 30 minutes; any progress call renews it
MAX_FRAMES = 300
MB = 1024 * 1024
# kind -> largest file the server accepts (media_service.PRIVATE_KINDS)
LIMITS = {"glb": 100 * MB, "stl": 300 * MB, "obj": 300 * MB, "ply": 300 * MB, "nc": 50 * MB, "render": 15 * MB}
CONTENT_TYPES = {".glb": "model/gltf-binary", ".stl": "model/stl", ".obj": "model/obj", ".ply": "application/octet-stream", ".nc": "text/plain", ".png": "image/png"}
COLMAP_QUALITY = {"fast": "low", "normal": "medium", "high": "high"}


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(f"{message} ({status})")
        self.status = status


def _detail(r):
    """The server's own reason for an error, in words."""
    try:
        d = r.json()["detail"]
        return d if isinstance(d, str) else d.get("message", str(d))
    except (ValueError, KeyError, AttributeError, TypeError):
        return (getattr(r, "text", "") or "")[:200]


class Api:
    """The few server calls the worker needs. Signs in again by itself when the token expires."""

    def __init__(self, base, worker, user=None, password=None):
        self.base, self.worker, self.user, self.password = base.rstrip("/"), worker, user, password
        self.token = None
        self.role = None

    def login(self):
        r = http.post(f"{self.base}/auth/login", data={"username": self.user, "password": self.password}, timeout=30)
        if r.status_code != 200:
            reason = _detail(r)
            if r.status_code == 401 and reason == "Invalid credentials":
                reason += ". The account must exist in the desktop app under Staff (role Artisan or Owner) and the password must match"
            raise ApiError(r.status_code, f"Sign-in failed: {reason or 'the server refused it'}")
        self.token = r.json()["access_token"]
        self.role = r.json().get("role")

    def _call(self, method, path, retry=True, **kw):
        if not self.token:
            self.login()
        r = http.request(method, f"{self.base}{path}", headers={"Authorization": f"Bearer {self.token}"}, timeout=60, **kw)
        if r.status_code == 401 and retry:  # token expired: sign in once more
            self.token = None
            return self._call(method, path, retry=False, **kw)
        if r.status_code >= 400:
            raise ApiError(r.status_code, _detail(r))
        return r.json()

    def queue(self):
        return self._call("GET", "/scans", params={"status": "uploaded", "limit": 50})

    def claim(self, sid):
        return self._call("POST", f"/scans/{sid}/claim", json={"worker": self.worker})

    def progress(self, sid, percent, note):
        return self._call("POST", f"/scans/{sid}/progress", json={"worker": self.worker, "percent": int(percent), "note": note[:200]})

    def urls(self, sid):
        return self._call("GET", f"/scans/{sid}/download-urls", params={"worker": self.worker})

    def presign(self, sid, kind, filename, content_type, size):
        body = {"worker": self.worker, "kind": kind, "filename": filename, "content_type": content_type, "size": size}
        return self._call("POST", f"/scans/{sid}/results/presign", json=body)

    def finish(self, sid, assets):
        return self._call("POST", f"/scans/{sid}/finish", json={"worker": self.worker, "assets": assets})

    def fail(self, sid, error):
        return self._call("POST", f"/scans/{sid}/fail", json={"worker": self.worker, "error": error[:500]})


class Progress:
    """Tells the server how far along we are. tick() renews the lease during long silent steps."""

    def __init__(self, api, sid, clock=time.monotonic):
        self.api, self.sid, self.clock = api, sid, clock
        self.percent, self.note, self.last = 0, "Starting", 0.0

    def set(self, percent, note):
        self.percent, self.note = percent, note
        self._send()

    def tick(self):
        if self.clock() - self.last >= HEARTBEAT_SECONDS:
            self._send()

    def _send(self):
        self.last = self.clock()
        try:
            self.api.progress(self.sid, self.percent, self.note)
        except ApiError:
            pass  # a missed update is harmless; losing the lease shows up when we finish


def run(cmd, progress=None):
    """Run a tool, calling progress.tick() as it prints. Returns its last output lines on failure."""
    tail = collections.deque(maxlen=15)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except FileNotFoundError:
        raise RuntimeError(f"'{cmd[0]}' was not found. Install it or pass its full path.") from None
    for line in proc.stdout:
        tail.append(line.rstrip())
        if progress:
            progress.tick()
    if proc.wait() != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n" + "\n".join(tail))


def download(api, sid, dest, progress):
    """Fetch every capture file into dest. Returns the local paths."""
    dest.mkdir(parents=True, exist_ok=True)
    items = api.urls(sid)
    paths = []
    for i, item in enumerate(items):
        path = dest / f"{i:04d}{Path(item['key']).suffix.lower()}"
        with http.get(item["url"], stream=True, timeout=(10, 300)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Could not download capture file {i + 1} ({r.status_code})")
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        paths.append(path)
        progress.set(2 + 8 * (i + 1) / len(items), f"Downloaded {i + 1} of {len(items)} files")
    return paths


def thin(files, limit=MAX_FRAMES):
    """Keep at most `limit` evenly spaced files (video frames are very similar to each other)."""
    if len(files) <= limit:
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]


def frames_from_video(video, out, ffmpeg="ffmpeg", fps=2, progress=None, min_frames=8):
    out.mkdir(parents=True, exist_ok=True)
    run([ffmpeg, "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2", str(out / "%05d.jpg")], progress)
    frames = sorted(out.glob("*.jpg"))
    keep = set(thin(frames))
    for f in frames:
        if f not in keep:
            f.unlink()
    if len(keep) < min_frames:
        raise RuntimeError("The video gave too few frames. Film slowly all the way around the piece.")
    return out


def colmap_command(colmap, images, workspace, quality):
    return [
        colmap, "automatic_reconstructor",
        "--workspace_path", str(workspace),
        "--image_path", str(images),
        "--data_type", "individual",
        "--quality", COLMAP_QUALITY.get(quality, "medium"),
        "--dense", "1",
        "--mesher", "poisson",
        "--use_gpu", "1",
    ]  # fmt: skip


def colmap_reconstruct(images, workspace, quality, progress, colmap="colmap"):
    """Photos -> sparse -> dense -> Poisson mesh. Returns the mesh (.ply)."""
    workspace.mkdir(parents=True, exist_ok=True)
    progress.set(15, "Reconstructing in 3D (this takes a while)")
    run(colmap_command(colmap, images, workspace, quality), progress)
    mesh = workspace / "dense" / "0" / "meshed-poisson.ply"
    if not mesh.exists():
        raise RuntimeError(
            "COLMAP finished but made no 3D model. Usually the photos do not overlap enough: "
            "take more, from many angles around the piece, in even light."
        )
    return mesh


def export_models(mesh_or_path, out, stem, width_mm=None, limits=None, mm=False):
    """Write .glb .stl .obj .ply. The widest side becomes width_mm when known (mm=True says the mesh
    is already in millimetres). A model too big for the server is simplified, the same way for
    every format, until each fits."""
    limits = limits or LIMITS
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path.copy()
    else:
        mesh = trimesh.load(mesh_or_path, force="mesh")
    if width_mm and not mm:
        mesh.apply_scale(width_mm / max(mesh.extents))
    formats = [(".glb", "glb"), (".stl", "stl"), (".obj", "obj"), (".ply", "ply")]
    while True:
        files = []
        fits = True
        for ext, kind in formats:
            path = out / f"{stem}{ext}"
            m = mesh.copy()
            if kind == "glb" and (width_mm or mm):
                m.apply_scale(0.001)  # glTF is in metres; the other formats stay in mm
            m.export(path)
            files.append((path, kind))
            fits = fits and path.stat().st_size <= limits[kind]
        if fits:
            return files
        if len(mesh.faces) < 20_000:
            raise RuntimeError("The 3D model is too large for the server even when simplified.")
        mesh = mesh.simplify_quadric_decimation(face_count=int(len(mesh.faces) * 0.6))


def make_gcode(model_path, out, stem, width_mm):
    """Relief G-code and a hill-shade preview from the exported mesh, using vriksha.py."""
    if not width_mm:
        raise RuntimeError("G-code needs the real width of the piece. Enter it on the scan and try again.")
    nc, png = out / f"{stem}.nc", out / f"{stem}-preview.png"
    c = vriksha.parse([str(model_path), "--width", str(width_mm), "-o", str(nc), "--preview", str(png)])
    code, tgt, _stats = vriksha.build(c)
    nc.write_text(code)
    vriksha.preview(tgt, c.px, png)
    return [(nc, "nc"), (png, "render")]


def make_gcode_depth(depth_png, out, stem, width_mm):
    """Relief G-code and a hill-shade preview straight from the depth map (vriksha.py's picture input)."""
    if not width_mm:
        raise RuntimeError("G-code needs the real width of the piece. Enter it on the scan and try again.")
    nc, png = out / f"{stem}.nc", out / f"{stem}-preview.png"
    argv = [str(depth_png), "--width", str(width_mm), "--max-depth", str(relief.RELIEF_DEPTH_MM), "--floor", str(relief.FLOOR), "-o", str(nc), "--preview", str(png)]
    c = vriksha.parse(argv)
    code, tgt, _stats = vriksha.build(c)
    nc.write_text(code)
    vriksha.preview(tgt, c.px, png)
    return [(nc, "nc"), (png, "render")]


def relief_files(images, folder, params, progress, estimate=None):
    """Flat carving: the sharpest photo -> depth map -> relief mesh (+ G-code). Returns the files to upload."""
    out = folder / "out"
    out.mkdir(parents=True, exist_ok=True)
    photo = relief.pick_front(sorted(Path(images).glob("*")))
    progress.set(15, "Reading the carving's depth (a minute or two on a normal PC)")
    depth = relief.shrink((estimate or relief.estimate_depth)(photo), relief.DEPTH_SIDE)
    width = params.get("real_width_mm")
    progress.set(70, "Making the 3D files")
    files = export_models(relief.relief_mesh(depth, width or 100.0), out, "relief", mm=True)
    depth_png = relief.save_depth_png(depth, folder / "depth.png")
    files.append((relief.save_preview_png(depth, out / "relief-depth.png"), "render"))
    if params.get("gcode"):
        progress.set(85, "Making the CNC G-code")
        files += make_gcode_depth(depth_png, out, "relief", width)
    return files


def upload(api, sid, files, label, progress):
    """Send each result straight to the store with a signed link. Returns the finish() assets."""
    assets = []
    for i, (path, kind) in enumerate(files):
        ctype = CONTENT_TYPES[path.suffix.lower()]
        signed = api.presign(sid, kind, path.name, ctype, path.stat().st_size)
        with open(path, "rb") as f:
            r = http.request(signed["method"], signed["url"], data=f, headers=signed["headers"], timeout=(10, 600))
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"Upload of {path.name} failed ({r.status_code})")
        assets.append({"key": signed["key"], "kind": kind, "visibility": "staff", "content_type": ctype, "alt": f"{label} ({path.suffix[1:].upper()})"})
        progress.set(92 + 7 * (i + 1) / len(files), f"Uploaded {path.name}")
    return assets


def process(api, scan, work, colmap="colmap", ffmpeg="ffmpeg", reconstruct=None, estimate=None, keep=False):
    """One whole job. Raises ApiError if the claim is lost; any other failure is reported to the server."""
    sid, params = scan["id"], scan.get("params") or {}
    shape = params.get("shape", "full")
    job = api.claim(sid)
    progress = Progress(api, sid)
    label = job.get("item_name") or "Restoration scan"
    folder = Path(work) / sid
    reconstruct = reconstruct or (lambda images, ws, q, p: colmap_reconstruct(images, ws, q, p, colmap))
    try:
        sources = download(api, sid, folder / "src", progress)
        if job["source_kind"] == "video":
            progress.set(10, "Cutting the video into frames")
            images = frames_from_video(sources[0], folder / "frames", ffmpeg, progress=progress, min_frames=1 if shape == "relief" else 8)
        else:
            images = folder / "src"
        if shape == "relief":
            files = relief_files(images, folder, params, progress, estimate)
        else:
            mesh = reconstruct(images, folder / "colmap", params.get("quality", "normal"), progress)
            progress.set(80, "Making the 3D files")
            width = params.get("real_width_mm")
            files = export_models(mesh, folder / "out", "model", width)
            if params.get("gcode"):
                progress.set(88, "Making the CNC G-code")
                stl = next(p for p, k in files if k == "stl")
                files += make_gcode(stl, folder / "out", "model", width)
        label = label + ("" if params.get("real_width_mm") else " (size not set)")
        assets = upload(api, sid, files, label, progress)
        api.finish(sid, assets)
        return "done"
    except Exception as exc:  # the job is ours: say why it failed so the artisan can retake photos
        try:
            api.fail(sid, str(exc))
        except ApiError:
            pass  # the lease was lost meanwhile; the server already gave the job back
        return "failed"
    finally:
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)


def gpu_available():
    """True when an NVIDIA GPU answers nvidia-smi (COLMAP's dense step needs one)."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def capabilities(colmap="colmap", ffmpeg="ffmpeg", mode="auto", gpu=None):
    """What this PC can build, as {what: None if it can, else why not}. Looked at again on every pass,
    so installing a tool or driver while the worker runs is noticed."""
    caps = {"relief": None, "full": None, "video": None}
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        caps["relief"] = "flat carvings need torch and transformers: pip install -r requirements-relief.txt"
    if not shutil.which(colmap):
        caps["full"] = f"full 3D needs COLMAP, which was not found ('{colmap}')"
    elif not (gpu_available() if gpu is None else gpu):
        caps["full"] = "full 3D needs an NVIDIA GPU, and none was found"
    if not shutil.which(ffmpeg):
        caps["video"] = f"video scans need ffmpeg, which was not found ('{ffmpeg}')"
    if mode == "relief":
        caps["full"] = "full 3D is turned off with --mode relief"
    elif mode == "full":
        caps["relief"] = "flat carvings are turned off with --mode full"
    return caps


def cannot_run(scan, caps):
    """Why this PC cannot do the scan, or None. Checked BEFORE claiming: a scan that fails is finished
    for good and the artisan would have to upload again, so one this PC cannot build stays waiting
    for a PC that can (a GPU PC, for full 3D)."""
    shape = (scan.get("params") or {}).get("shape", "full")
    if caps.get(shape):
        return caps[shape]
    return caps["video"] if scan.get("source_kind") == "video" else None


def run_once(api, work, log=print, caps=None, mode="auto", **kw):
    """Process every waiting scan this PC can build, oldest first. Returns how many were handled."""
    caps = caps or capabilities(kw.get("colmap", "colmap"), kw.get("ffmpeg", "ffmpeg"), mode)
    handled = 0
    for scan in api.queue():
        if not scan.get("restoration_id") and not scan.get("item_id"):
            log(f"skip {scan['id'][:8]}: not attached to a product or a restoration")
            continue
        problem = cannot_run(scan, caps)
        if problem:
            log(f"left {scan['id'][:8]} waiting: {problem}")
            continue
        try:
            outcome = process(api, scan, work, **kw)
        except ApiError as exc:
            log(f"skip {scan['id'][:8]}: {exc}")  # another PC took it, or it was cancelled
            continue
        log(f"{scan['id'][:8]}: {outcome}")
        handled += 1
    return handled


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("VRIKSHA_URL"), help="the server address")
    p.add_argument("--user", default=os.environ.get("VRIKSHA_USER"), help="an owner or artisan account")
    p.add_argument("--worker", default="workshop-pc", help="a name for this PC")
    p.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "vriksha-worker"))
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--mode", choices=["auto", "full", "relief"], default="auto", help="auto: build what this PC can; or force full 3D or flat carvings only")
    p.add_argument("--once", action="store_true", help="one pass over the queue, then stop")
    p.add_argument("--keep", action="store_true", help="keep each job's working files")
    a = p.parse_args(argv)
    if not a.url or not a.user:
        p.error("--url and --user are needed (or VRIKSHA_URL and VRIKSHA_USER)")
    api = Api(a.url, a.worker, a.user, os.environ.get("VRIKSHA_PASSWORD") or getpass.getpass("Password: "))
    try:
        api.login()
    except (ApiError, requests.RequestException) as exc:
        print(f"Could not sign in to {a.url}: {exc}")
        return 1
    print(f"signed in to {a.url} as {a.user} ({api.role}); worker '{a.worker}'")
    if api.role == "manager":
        print("note: a manager can only process product scans. Temple restoration scans need an owner or artisan account.")
    caps = capabilities(a.colmap, a.ffmpeg, a.mode)
    for what, label in (("relief", "flat carvings (one photo, any PC)"), ("full", "full 3D objects (photos all around)")):
        print(f"  {label}: " + ("can build" if not caps[what] else f"waiting for a PC that can ({caps[what]})"))
    if caps["video"]:
        print(f"  note: {caps['video']}")
    print(f"waiting for scans, checking every {POLL_SECONDS} s. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                handled = run_once(api, a.work_dir, mode=a.mode, colmap=a.colmap, ffmpeg=a.ffmpeg, keep=a.keep)
                if handled:
                    print("waiting for scans again")
            except (requests.RequestException, ApiError) as exc:
                print(f"server not reachable or refused: {exc}")  # keep going: it may be a blip
            if a.once:
                return 0
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
