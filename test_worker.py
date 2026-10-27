"""Workshop worker: the whole job with a fake server and a synthetic mesh (no COLMAP, no GPU)."""

import io
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

import relief
import workshop_worker as w

MB = 1024 * 1024
OK = {"relief": None, "full": None, "video": None}  # a PC that can build everything


def make_ply(path, faces_sub=3):
    m = trimesh.creation.icosphere(subdivisions=faces_sub)
    m.apply_scale([20, 15, 4])  # a flattened relief-like shape, in arbitrary units
    m.export(path)
    return path


def _jpeg():
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


JPEG = _jpeg()


class FakeResp:
    def __init__(self, status=200, data=b"", body=None):
        self.status_code, self._data, self._body = status, data, {} if body is None else body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        yield self._data

    def json(self):
        return self._body

    text = ""


class FakeHttp:
    """Stands in for `requests`: store downloads, signed uploads, and the server's own calls."""

    def __init__(self):
        self.uploads = {}

    def get(self, url, **kw):
        return FakeResp(200, JPEG)

    def request(self, method, url, data=None, **kw):
        if hasattr(data, "read"):
            self.uploads[url] = len(data.read())
            return FakeResp(200)
        return FakeResp(200, body={})


class FakeApi:
    worker = "workshop-pc"

    def __init__(self, scans=(), photos=10, kind="photos", lose_claim=()):
        self.scans, self.photos, self.kind, self.lose_claim = list(scans), photos, kind, set(lose_claim)
        self.calls, self.assets, self.failed = [], None, None

    def queue(self):
        return self.scans

    def claim(self, sid):
        self.calls.append("claim")
        if sid in self.lose_claim:
            raise w.ApiError(409, "This scan is not available to claim")
        return {"id": sid, "source_kind": self.kind, "item_name": "Carved Chair"}

    def progress(self, sid, percent, note):
        self.calls.append(f"progress:{percent}")

    def urls(self, sid):
        return [{"key": f"scans/{sid}/src/{i}.jpg", "url": f"https://store.invalid/{i}"} for i in range(self.photos)]

    def presign(self, sid, kind, filename, content_type, size):
        self.calls.append(f"presign:{kind}")
        return {"key": f"scans/{sid}/out/{filename}", "url": f"https://store.invalid/up/{filename}", "method": "PUT", "headers": {}}

    def finish(self, sid, assets):
        self.calls.append("finish")
        self.assets = assets

    def fail(self, sid, error):
        self.calls.append("fail")
        self.failed = error


@pytest.fixture
def fake_http(monkeypatch):
    h = FakeHttp()
    monkeypatch.setattr(w, "http", h)
    return h


def fake_reconstruct(images, workspace, quality, progress):
    workspace.mkdir(parents=True, exist_ok=True)
    return make_ply(workspace / "mesh.ply")


# ------------------------------------------------------------------ pieces
def test_thinning_keeps_even_spread_and_a_cap():
    files = list(range(1000))
    kept = w.thin(files, 300)
    assert len(kept) == 300 and kept[0] == 0 and kept[-1] > 950
    assert w.thin(files[:50], 300) == files[:50]


def test_colmap_command_uses_the_quality_the_scan_asked_for():
    cmd = w.colmap_command("colmap", "imgs", "ws", "high")
    assert cmd[:2] == ["colmap", "automatic_reconstructor"]
    assert cmd[cmd.index("--quality") + 1] == "high"
    assert w.colmap_command("c", "i", "w", "fast")[w.colmap_command("c", "i", "w", "fast").index("--quality") + 1] == "low"
    assert w.colmap_command("c", "i", "w", "weird").count("medium") == 1  # unknown -> normal


def test_run_reports_a_missing_tool_and_the_tail_of_a_failure():
    with pytest.raises(RuntimeError, match="was not found"):
        w.run(["definitely-not-installed-tool"])
    with pytest.raises(RuntimeError, match="boom"):
        w.run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])


def test_export_scales_to_the_real_width_and_writes_every_format(tmp_path):
    ply = make_ply(tmp_path / "m.ply")
    files = w.export_models(ply, tmp_path / "out", "model", width_mm=300)
    assert [k for _, k in files] == ["glb", "stl", "obj", "ply"]
    assert all(p.exists() and p.stat().st_size > 0 for p, _ in files)
    stl = trimesh.load(tmp_path / "out" / "model.stl", force="mesh")
    assert max(stl.extents) == pytest.approx(300, rel=1e-3)  # millimetres
    glb = trimesh.load(tmp_path / "out" / "model.glb", force="mesh")
    assert max(glb.extents) == pytest.approx(0.3, rel=1e-2)  # glTF is in metres


def test_a_model_too_big_for_the_server_is_simplified_until_it_fits(tmp_path):
    ply = make_ply(tmp_path / "big.ply", faces_sub=6)  # about 82,000 faces
    limit = 1_500_000
    files = w.export_models(ply, tmp_path / "out", "model", limits={k: limit for k in ("glb", "stl", "obj", "ply")})
    assert all(p.stat().st_size <= limit for p, _ in files)
    assert len(trimesh.load(tmp_path / "out" / "model.stl", force="mesh").faces) < 82_000
    small = make_ply(tmp_path / "small.ply")
    with pytest.raises(RuntimeError, match="too large"):
        w.export_models(small, tmp_path / "o2", "m", limits={k: 100 for k in ("glb", "stl", "obj", "ply")})


def test_gcode_needs_the_real_width_and_makes_a_cutting_file_and_preview(tmp_path):
    stl = w.export_models(make_ply(tmp_path / "m.ply"), tmp_path / "out", "model", 40)[1][0]
    with pytest.raises(RuntimeError, match="real width"):
        w.make_gcode(stl, tmp_path / "out", "model", None)
    files = w.make_gcode(stl, tmp_path / "out", "model", 40)
    assert [k for _, k in files] == ["nc", "render"]
    assert files[0][0].read_text().startswith("G21")
    assert files[1][0].suffix == ".png" and files[1][0].stat().st_size > 0


def test_progress_only_renews_the_lease_when_a_minute_has_passed():
    now = [1000.0]
    api = FakeApi()
    p = w.Progress(api, "s1", clock=lambda: now[0])
    p.set(15, "Reconstructing")
    sent = len(api.calls)
    now[0] += 10
    p.tick()
    assert len(api.calls) == sent  # too soon
    now[0] += w.HEARTBEAT_SECONDS
    p.tick()
    assert len(api.calls) == sent + 1 and api.calls[-1] == "progress:15"


def test_the_api_signs_in_again_when_the_token_expires(monkeypatch):
    seen = []

    class H:
        def post(self, url, **kw):
            seen.append("login")
            return FakeResp(200, body={"access_token": f"t{len(seen)}"})

        def request(self, method, url, headers=None, **kw):
            seen.append(headers["Authorization"])
            return FakeResp(401 if headers["Authorization"] == "Bearer t1" else 200, body=[])

    monkeypatch.setattr(w, "http", H())
    api = w.Api("https://x", "pc", "u", "p")
    assert api.queue() == []
    assert seen == ["login", "Bearer t1", "login", "Bearer t3"]


@pytest.mark.parametrize(
    "status,body,expect",
    [
        (401, {"detail": "Invalid credentials"}, "must exist in the desktop app under Staff"),
        (401, {"detail": "This account is switched off. Ask the owner."}, "switched off"),
        (429, {"detail": "Too many failed sign-in attempts. Try again in 15 minutes."}, "Too many failed"),
    ],
)
def test_a_failed_sign_in_says_why_and_never_prints_a_traceback(monkeypatch, capsys, status, body, expect):
    class H:
        def post(self, url, **kw):
            return FakeResp(status, body=body)

    monkeypatch.setattr(w, "http", H())
    monkeypatch.setenv("VRIKSHA_PASSWORD", "x")
    assert w.main(["--url", "https://x", "--user", "workshop-pc", "--once"]) == 1
    out = capsys.readouterr().out
    assert "Could not sign in" in out and expect in out and "Traceback" not in out


def signed_in_http(role, queue=()):
    class H:
        def post(self, url, **kw):
            return FakeResp(200, body={"access_token": "t", "role": role})

        def request(self, method, url, **kw):
            return FakeResp(200, body=list(queue))

    return H()


def test_ctrl_c_stops_quietly_and_a_manager_is_told_about_restorations(monkeypatch, capsys):
    monkeypatch.setattr(w, "http", signed_in_http("manager"))
    monkeypatch.setenv("VRIKSHA_PASSWORD", "x")

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(w.time, "sleep", interrupt)
    assert w.main(["--url", "https://x", "--user", "supriya"]) == 0
    out = capsys.readouterr().out
    assert "waiting for scans" in out and "stopped" in out and "Traceback" not in out
    assert "(manager)" in out and "restoration scans need an owner or artisan" in out


def test_an_artisan_gets_no_manager_warning(monkeypatch, capsys):
    monkeypatch.setattr(w, "http", signed_in_http("artisan"))
    monkeypatch.setenv("VRIKSHA_PASSWORD", "x")
    assert w.main(["--url", "https://x", "--user", "carver", "--once"]) == 0
    assert "only process product scans" not in capsys.readouterr().out


def test_capabilities_follow_the_hardware(monkeypatch):
    monkeypatch.setattr(w.shutil, "which", lambda name: name)  # every tool "exists"
    monkeypatch.setattr(w.importlib.util, "find_spec", lambda name: object())  # torch and transformers too
    assert w.capabilities(gpu=True) == OK
    no_gpu = w.capabilities(gpu=False)
    assert no_gpu["relief"] is None and "NVIDIA GPU" in no_gpu["full"]  # a CPU-only PC still does flat carvings
    monkeypatch.setattr(w.shutil, "which", lambda name: None if name in ("colmap", "ffmpeg") else name)
    bare = w.capabilities(gpu=True)
    assert "COLMAP" in bare["full"] and "ffmpeg" in bare["video"] and bare["relief"] is None
    monkeypatch.setattr(w.importlib.util, "find_spec", lambda name: None)
    assert "torch" in w.capabilities(gpu=True)["relief"]


def test_the_mode_flag_can_turn_a_route_off(monkeypatch):
    monkeypatch.setattr(w.shutil, "which", lambda name: name)
    monkeypatch.setattr(w.importlib.util, "find_spec", lambda name: object())
    assert "turned off" in w.capabilities(mode="relief", gpu=True)["full"]
    assert "turned off" in w.capabilities(mode="full", gpu=True)["relief"]
    assert w.capabilities(mode="auto", gpu=True) == OK


def test_gpu_detection_needs_a_working_nvidia_smi(monkeypatch):
    monkeypatch.setattr(w.shutil, "which", lambda name: None)
    assert w.gpu_available() is False
    monkeypatch.setattr(w.shutil, "which", lambda name: name)
    ran = type("R", (), {"returncode": 0, "stdout": "GPU 0: NVIDIA GeForce RTX 3060 (UUID: x)"})()
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: ran)
    assert w.gpu_available() is True
    monkeypatch.setattr(w.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 9, "stdout": ""})())
    assert w.gpu_available() is False


def test_what_this_pc_cannot_build_stays_waiting_and_the_rest_is_built(tmp_path, fake_http):
    cpu_only = {"relief": None, "full": "full 3D needs an NVIDIA GPU, and none was found", "video": None}
    flat = {**scan("flat0001"), "params": {"shape": "relief"}}
    solid = scan("full0001")  # no shape = full
    api = FakeApi(scans=[solid, flat], photos=1)
    logs = []
    handled = w.run_once(api, tmp_path, log=logs.append, caps=cpu_only, estimate=gradient_depth)
    assert handled == 1 and api.calls.count("claim") == 1 and api.calls.count("finish") == 1
    assert any("left full0001 waiting" in line and "NVIDIA GPU" in line for line in logs)
    assert not any("left flat0001" in line for line in logs)


def test_a_video_scan_waits_for_ffmpeg_but_a_photo_scan_does_not():
    no_ffmpeg = {**OK, "video": "video scans need ffmpeg, which was not found ('ffmpeg')"}
    video = {**scan("v"), "source_kind": "video"}
    assert "ffmpeg" in w.cannot_run(video, no_ffmpeg)
    assert w.cannot_run({**video, "source_kind": "photos"}, no_ffmpeg) is None


# ------------------------------------------------------------------ flat carving (relief)
def gradient_depth(photo):
    """A stand-in depth estimate: a smooth bump, nearest in the middle."""
    y, x = np.mgrid[-1:1:120j, -1:1:160j]
    return np.clip(1 - np.hypot(x, y), 0, 1).astype(np.float32)


def write_photo(path, noise):
    rng = np.random.default_rng(1)
    base = np.full((200, 300), 120.0)
    if noise:
        base = base + rng.normal(0, noise, base.shape)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB").save(path)
    return path


def test_the_sharpest_photo_is_chosen(tmp_path):
    blurry, crisp = write_photo(tmp_path / "a.jpg", 0), write_photo(tmp_path / "b.jpg", 40)
    assert relief.pick_front([blurry, crisp]) == crisp
    with pytest.raises(RuntimeError, match="no photo"):
        relief.pick_front([tmp_path / "notes.txt"])


def test_the_relief_mesh_has_the_real_width_and_depth_and_faces_up():
    d = gradient_depth(None)
    mesh = relief.relief_mesh(d, width_mm=300, depth_mm=15)
    lo, hi = mesh.bounds
    assert hi[0] - lo[0] == pytest.approx(300) and lo[2] == pytest.approx(-15) and hi[2] == pytest.approx(0)
    assert hi[1] - lo[1] == pytest.approx(300 * (d.shape[0] - 1) / (d.shape[1] - 1))  # same shape as the photo
    assert mesh.faces.shape[0] == 2 * (d.shape[0] - 1) * (d.shape[1] - 1)
    assert mesh.face_normals[:, 2].min() >= -1e-9  # every face looks up at the viewer
    flat = relief.relief_mesh(np.full((20, 30), 0.1), 100)  # all below the floor: one flat background
    assert flat.extents[2] == pytest.approx(0)


def test_shrink_keeps_the_shape_and_never_enlarges():
    big = np.random.default_rng(0).random((400, 800)).astype(np.float32)
    assert relief.shrink(big, 200).shape == (100, 200)
    assert relief.shrink(big, 5000) is big


def test_a_flat_carving_job_needs_no_colmap_and_uploads_model_depth_and_gcode(tmp_path, fake_http):
    api = FakeApi(photos=1)
    used = []

    def estimate(photo):
        used.append("depth")
        return gradient_depth(photo)

    params = {"shape": "relief", "real_width_mm": 40, "gcode": True}
    out = w.process(api, {**scan(), "params": params}, tmp_path, reconstruct=lambda *a: used.append("colmap"), estimate=estimate)
    assert out == "done" and api.failed is None
    assert used == ["depth"]  # COLMAP is never touched for a flat carving
    assert [a["kind"] for a in api.assets] == ["glb", "stl", "obj", "ply", "render", "nc", "render"]
    assert all(a["visibility"] == "staff" for a in api.assets)
    assert "size not set" not in api.assets[0]["alt"]


def test_a_flat_carving_without_a_width_still_gets_a_model_but_no_gcode(tmp_path, fake_http):
    api = FakeApi(photos=1)
    w.process(api, {**scan(), "params": {"shape": "relief"}}, tmp_path, estimate=gradient_depth)
    assert [a["kind"] for a in api.assets] == ["glb", "stl", "obj", "ply", "render"]
    assert "size not set" in api.assets[0]["alt"]  # so nobody trusts its size
    bad = FakeApi(photos=1)
    assert w.process(bad, {**scan(), "params": {"shape": "relief", "gcode": True}}, tmp_path, estimate=gradient_depth) == "failed"
    assert "real width" in bad.failed


def test_the_flat_carving_model_is_in_millimetres(tmp_path, fake_http, monkeypatch):
    kept = {}
    real = w.export_models

    def spy(mesh, out, stem, **kw):
        files = real(mesh, out, stem, **kw)
        kept.update({k: p for p, k in files})
        return files

    monkeypatch.setattr(w, "export_models", spy)
    w.process(FakeApi(photos=1), {**scan(), "params": {"shape": "relief", "real_width_mm": 250}}, tmp_path, keep=True, estimate=gradient_depth)
    stl = trimesh.load(kept["stl"], force="mesh")
    glb = trimesh.load(kept["glb"], force="mesh")
    assert stl.extents[0] == pytest.approx(250, rel=1e-3) and glb.extents[0] == pytest.approx(0.25, rel=1e-2)


# ------------------------------------------------------------------ the job
def scan(sid="aaaa1111", **params):
    return {"id": sid, "restoration_id": "r1", "item_id": None, "params": params}


def test_a_whole_job_photos_to_files_uploaded_and_finished(tmp_path, fake_http):
    api = FakeApi()
    out = w.process(api, scan(real_width_mm=40, gcode=True, quality="normal"), tmp_path, reconstruct=fake_reconstruct)
    assert out == "done" and api.failed is None
    kinds = [c.split(":")[1] for c in api.calls if c.startswith("presign")]
    assert kinds == ["glb", "stl", "obj", "ply", "nc", "render"]
    assert api.calls[0] == "claim" and api.calls[-1] == "finish"
    assert [a["kind"] for a in api.assets] == kinds
    assert all(a["visibility"] == "staff" for a in api.assets)  # never published by the PC
    assert len(fake_http.uploads) == 6 and all(n > 0 for n in fake_http.uploads.values())
    assert not (tmp_path / "aaaa1111").exists()  # working files are cleared


def test_without_gcode_only_the_four_model_files_go_up(tmp_path, fake_http):
    api = FakeApi()
    w.process(api, scan(), tmp_path, reconstruct=fake_reconstruct)
    assert [a["kind"] for a in api.assets] == ["glb", "stl", "obj", "ply"]


def test_a_failure_is_reported_with_its_reason_and_nothing_is_finished(tmp_path, fake_http):
    def no_overlap(*a):
        raise RuntimeError("COLMAP made no 3D model: take more photos")

    api = FakeApi()
    assert w.process(api, scan(), tmp_path, reconstruct=no_overlap) == "failed"
    assert "finish" not in api.calls and api.calls[-1] == "fail"
    assert "take more photos" in api.failed
    assert not (tmp_path / "aaaa1111").exists()


def test_gcode_without_a_width_fails_the_job_clearly(tmp_path, fake_http):
    api = FakeApi()
    assert w.process(api, scan(gcode=True), tmp_path, reconstruct=fake_reconstruct) == "failed"
    assert "real width" in api.failed and "finish" not in api.calls


def test_a_lost_claim_touches_nothing_and_the_queue_moves_on(tmp_path, fake_http):
    api = FakeApi(scans=[scan("lost0001"), scan("mine0002")], lose_claim={"lost0001"})
    logs = []
    handled = w.run_once(api, tmp_path, log=logs.append, caps=OK, reconstruct=fake_reconstruct)
    assert handled == 1 and api.calls.count("finish") == 1
    assert any("skip lost0001" in line for line in logs)


def test_a_scan_with_no_product_or_restoration_is_left_alone(tmp_path, fake_http):
    api = FakeApi(scans=[{"id": "orphan01", "restoration_id": None, "item_id": None, "params": {}}])
    assert w.run_once(api, tmp_path, log=lambda m: None, caps=OK, reconstruct=fake_reconstruct) == 0
    assert api.calls == []


def test_a_video_scan_is_cut_into_frames_first(tmp_path, fake_http, monkeypatch):
    cut = []

    def fake_frames(video, out, ffmpeg, progress=None, min_frames=8):
        cut.append(video.name)
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(w, "frames_from_video", fake_frames)
    api = FakeApi(photos=1, kind="video")
    assert w.process(api, scan(), tmp_path, reconstruct=fake_reconstruct) == "done"
    assert cut == ["0000.jpg"]  # the first downloaded file is the video
