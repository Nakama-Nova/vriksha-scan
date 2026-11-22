import numpy as np
import trimesh

import web_glb


def test_decimates_keeps_colour_and_metres(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=6, radius=0.3)  # 81,920 faces, 0.6 m wide
    m.visual.vertex_colors = np.tile([200, 40, 30, 255], (len(m.vertices), 1)).astype(np.uint8)
    src, dst = tmp_path / "a.glb", tmp_path / "b.glb"
    m.export(src)
    before, after, size = web_glb.web_ready(src, dst, faces=10_000)
    out = trimesh.load(dst, force="mesh")
    assert before == 81_920 and after <= 10_000 and size == dst.stat().st_size
    assert abs(out.extents.max() - 0.6) < 0.02          # size in metres is untouched
    assert (out.visual.vertex_colors[:, 0] > 150).all()  # red survived the decimation


def test_shrinks_until_it_fits(tmp_path):
    m = trimesh.creation.icosphere(subdivisions=5)
    src, dst = tmp_path / "a.glb", tmp_path / "b.glb"
    m.export(src)
    _, after, size = web_glb.web_ready(src, dst, faces=20_000, max_bytes=300_000)
    assert size <= 300_000 and after < 20_000
