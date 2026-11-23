"""The CAD folder: outline from the depth map, DXF/SVG, and the zip."""

import zipfile

import numpy as np
import pytest

import cad_pack
import relief


def disk(h=300, w=400):
    """A carving shaped like an ellipse on background (0), as an outlined area leaves it."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2) / (w * 0.4)) ** 2 + ((yy - h / 2) / (h * 0.4)) ** 2 <= 1
    d = np.where(inside, 0.3 + 0.7 * np.random.default_rng(1).random((h, w)), 0).astype(np.float32)
    return d, inside


def area(points):
    p = np.array(points)
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def test_a_full_map_is_a_plain_rectangle_of_the_real_size():
    depth = np.full((300, 400), 0.5, dtype=np.float32)
    pts, w, h = cad_pack.outline_mm(depth, 1000)
    assert (w, h) == (1000, 750) and len(pts) == 4
    assert area(pts) == pytest.approx(1000 * 750)


def test_a_shaped_area_gets_its_own_outline_in_millimetres():
    depth, inside = disk()
    pts, w, h = cad_pack.outline_mm(depth, 400)  # 1 px = 1 mm
    assert 12 < len(pts) < 150  # a smooth curve, not every pixel
    assert area(pts) == pytest.approx(inside.sum(), rel=0.04)
    p = np.array(pts)
    assert p[:, 0].min() == pytest.approx(40, abs=3) and p[:, 0].max() == pytest.approx(360, abs=3)
    assert p[:, 1].min() == pytest.approx(30, abs=3) and p[:, 1].max() == pytest.approx(270, abs=3)  # y up


def test_a_hole_inside_the_carving_is_not_part_of_the_outline():
    depth, inside = disk()
    depth[140:160, 190:210] = 0  # a low spot clipped to zero
    pts, _, _ = cad_pack.outline_mm(depth, 400)
    assert area(pts) == pytest.approx(inside.sum(), rel=0.04)


def test_the_dxf_and_svg_say_what_they_hold():
    pts = [(0, 0), (100, 0), (100, 50), (0, 50)]
    text = cad_pack.dxf(pts)
    assert text.count("VERTEX") == 4 and text.startswith("0\nSECTION") and text.endswith("EOF\n")
    assert "10\n100.000\n20\n50.000" in text
    s = cad_pack.svg(pts, 100, 50)
    assert 'width="100.0mm"' in s and "M0.000,50.000 L100.000,50.000 L100.000,0.000 L0.000,0.000 Z" in s


def test_the_zip_holds_the_model_the_outline_the_depth_map_and_a_readme(tmp_path):
    depth, _ = disk()
    png = relief.save_depth_png(depth, tmp_path / "d.png")
    stl, obj = tmp_path / "a.stl", tmp_path / "a.obj"
    stl.write_bytes(b"solid x")
    obj.write_bytes(b"v 0 0 0")
    path = cad_pack.pack(depth, 400, [stl, obj], png, tmp_path / "out", "Queen bed: headboard border")
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["README.txt", "depth-map.png", "model.obj", "model.stl", "outline.dxf", "outline.svg"]
        text = z.read("README.txt").decode()
    assert "Queen bed: headboard border" in text and "400 x 300 mm" in text and "estimate" in text
