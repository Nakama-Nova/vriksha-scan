import re

import trimesh

import vriksha


def test_dome(tmp_path):
    # 120x80x20 mm ellipsoid in a random orientation: orient() must find the panel normal on its own
    m = trimesh.creation.uv_sphere(radius=1, count=[64, 64])
    m.apply_scale([60, 40, 10])
    m.apply_transform(trimesh.transformations.random_rotation_matrix(seed=1))
    scan = tmp_path / "dome.ply"
    m.export(scan)

    c = vriksha.parse([str(scan), "--width", "60", "--max-depth", "20"])
    code, tgt, stats = vriksha.build(c)

    rough, fin = code.split("(ROUGH)")[1].split("(FINISH)")
    z = lambda s: [float(v) for v in re.findall(r"^G1 .*Z(-?[\d.]+)", s, re.M)]
    zr, zf = z(rough), z(fin)
    assert zr and zf
    assert max(zr + zf) <= 1e-6, "cutter above stock top is fine, but cuts must not exceed Z0"
    assert abs(max(zf)) < 0.1, "finish pass should touch the top of the dome"
    assert min(zf) >= -stats["depth_mm"] - 1e-3
    assert min(zr) >= min(zf) - 1e-3, "rough pass must never cut deeper than finish"
    assert stats["depth_mm"] < 20 and 2.5 < stats["depth_mm"] < 8  # 60mm wide dome of 120mm ellipse -> ~5 mm
    vriksha.preview(tgt, c.px, tmp_path / "p.png")
    assert (tmp_path / "p.png").exists()
