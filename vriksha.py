"""Temple scan -> 3-axis CNC relief G-code.

mesh -> clean/orient/scale -> height map -> rough pass (flat tool) + finish pass (ball nose).
G-code frame: X0 Y0 = lower-left of panel, Z0 = top of relief, cuts are negative Z.
"""
import argparse
import math
import os

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi

# ponytail: dialect = tool-change/header strings only; add per-controller quirks once the machine is bought
DIALECTS = {
    "generic": dict(head="G21 G90 G17 G94", tool="M6 T{n} (tool {n}: {name})", tail="M30"),
    "mach3": dict(head="G21 G90 G17 G94 G40 G49", tool="M6 T{n} (tool {n}: {name})", tail="M30"),
    "grbl": dict(head="G21 G90 G17 G94", tool="M0 (change to tool {n}: {name}, then resume)", tail="M2"),
}


def load_mesh(path):
    if not os.path.isfile(path):
        raise SystemExit(f"scan file not found: {path} (give the real path to your .ply/.obj/.stl mesh)")
    m = trimesh.load(path, force="mesh")
    m = trimesh.Trimesh(m.vertices, m.faces, process=False)  # geometry only: 4096px textures cost GBs when copied
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    # photogrammetry leaves floating junk: keep only the biggest connected piece
    # ponytail: no hole filling, the z-buffer + nearest fill in heightmap() covers small holes
    return max(m.split(only_watertight=False), key=lambda p: len(p.faces))


def orient(m, flip=False):
    """PCA: thinnest axis becomes Z (panel normal), widest X. `flip` turns the panel over if the relief faces down."""
    v = m.vertices - m.vertices.mean(0)
    R = np.linalg.eigh(np.cov(v.T))[1][:, ::-1].T  # rows: widest, middle, thinnest
    if np.linalg.det(R) < 0:
        R[2] *= -1
    if flip:
        R = np.diag([-1, 1, -1]) @ R
    return trimesh.Trimesh(vertices=v @ R.T, faces=m.faces, process=False)


def heightmap(m, width_mm, px, max_depth, smooth=1.5, hole_mm=3.0):
    """Top-down z-buffer. Returns (h, depth_mm): row 0 = Y min, top of relief = 0, floor = -depth."""
    v = m.vertices * (width_mm / np.ptp(m.vertices[:, 0]))
    m = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
    lo, nx, ny = v.min(0), math.ceil(np.ptp(v[:, 0]) / px), math.ceil(np.ptp(v[:, 1]) / px)
    h = np.full((ny, nx), -np.inf)
    total = int(m.area / px**2 * 12)  # ~12 samples per pixel: fewer leaves speckle that the cutter would trace
    for n in [2_000_000] * (total // 2_000_000) + [total % 2_000_000 + 1]:
        p = m.sample(n)
        np.maximum.at(h, (np.minimum(((p[:, 1] - lo[1]) / px).astype(int), ny - 1),
                          np.minimum(((p[:, 0] - lo[0]) / px).astype(int), nx - 1)), p[:, 2])
    seen = np.isfinite(h)
    dist, idx = ndi.distance_transform_edt(~seen, return_indices=True)
    filled = h[tuple(idx)]
    # small gaps = scan holes, copy nearest; big empty areas = outside the carving, leave at floor
    # ponytail: fixed 3 mm gap limit, expose as a flag if scans have bigger holes
    h = np.where(dist * px <= hole_mm, filled, h[seen].min())
    h = ndi.gaussian_filter(h, smooth)  # denoise: scan noise would become tool chatter
    h -= h.max()
    depth = -h.min()
    if depth > max_depth:
        print(f"warning: relief {depth:.1f} mm deep, compressing to {max_depth} mm")
        h *= max_depth / depth
        depth = max_depth
    return h, depth


def depth_image(path, width_mm, px, max_depth, floor, smooth=1.5):
    """Depth map image (white = nearest, see photo_to_depth.py) -> (h, depth_mm). Anything darker than `floor` (0-1) becomes flat background."""
    g = np.asarray(Image.open(path), dtype=float)
    g = g.mean(2) if g.ndim == 3 else g
    g = np.clip((g / g.max() - floor) / (1 - floor), 0, 1)[::-1]  # flip: row 0 is Y min
    nx = round(width_mm / px)
    ny = round(nx * g.shape[0] / g.shape[1])
    g = np.asarray(Image.fromarray(np.ascontiguousarray(g, dtype=np.float32)).resize((nx, ny), Image.BOX))
    return (ndi.gaussian_filter(g, smooth) - 1) * max_depth, max_depth


def tool_surface(h, px, radius, ball):
    """Lowest safe tool-tip Z at each pixel without gouging (flat or ball-nose cutter)."""
    r = int(math.ceil(radius / px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    d = np.hypot(xx, yy) * px
    fp = d <= radius
    rise = radius - np.sqrt(np.maximum(radius**2 - d**2, 0)) if ball else np.zeros_like(d)
    return ndi.grey_dilation(h, footprint=fp, structure=-rise, mode="nearest")


class GCode:
    def __init__(self, dialect, safe_z):
        self.d, self.safe, self.out, self.pos, self.f, self.minutes = DIALECTS[dialect], safe_z, [], (0, 0, safe_z), None, 0.0

    def rapid(self, x, y, z):
        self.out.append(f"G0 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        self.pos = (x, y, z)

    def cut(self, x, y, z, f):
        self.minutes += math.dist(self.pos, (x, y, z)) / f
        self.out.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f}" + ("" if f == self.f else f" F{f}"))
        self.pos, self.f = (x, y, z), f

    def tool(self, n, name, rpm):
        self.out += [f"M5", self.d["tool"].format(n=n, name=name), f"M3 S{rpm}"]
        self.rapid(*self.pos[:2], self.safe)


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def simplify(z, tol):
    """Greedy collinear-point removal: indices to keep so the polyline stays within tol of z."""
    if len(z) < 3:
        return list(range(len(z)))
    keep, a = [0], 0
    for i in range(2, len(z)):
        t = np.arange(a, i + 1)
        if np.abs(z[a : i + 1] - (z[a] + (z[i] - z[a]) * (t - a) / (i - a))).max() > tol:
            keep.append(i - 1)
            a = i - 1
    return keep + [len(z) - 1]


def rough(g, h, px, c):
    tgt = np.minimum(tool_surface(h, px, c.rough_dia / 2, False) + c.allowance, 0)
    low, z, layers = tgt.min(), 0.0, []
    while z > low:
        z = max(z - c.stepdown, low)
        layers.append(z)
    step = max(1, round(c.rough_dia * 0.5 / px))
    g.out.append("(ROUGH)")
    for z in layers:
        for k, r in enumerate(range(0, h.shape[0], step)):
            seg = runs(tgt[r] <= z + 1e-6)
            for a, b in seg[::-1] if k % 2 else seg:  # zigzag so travel is not wasted
                (a, b) = (b, a) if k % 2 else (a, b)
                g.rapid((a + 0.5) * px, (r + 0.5) * px, g.safe)
                g.cut((a + 0.5) * px, (r + 0.5) * px, z, c.plunge)
                g.cut((b + 0.5) * px, (r + 0.5) * px, z, c.feed)
                g.rapid((b + 0.5) * px, (r + 0.5) * px, g.safe)  # ponytail: retract every run, merge near runs if cycle time matters
    return len(layers)


def finish(g, h, px, c):
    tgt = tool_surface(h, px, c.ball_dia / 2, True)
    g.out.append("(FINISH)")
    for k, r in enumerate(range(0, h.shape[0], max(1, round(c.finish_stepover / px)))):
        z = tgt[r]
        idx = simplify(z, 0.01)
        if k % 2:
            idx = idx[::-1]
        y = (r + 0.5) * px
        g.rapid((idx[0] + 0.5) * px, y, g.safe)
        g.cut((idx[0] + 0.5) * px, y, z[idx[0]], c.plunge)
        for i in idx[1:]:
            g.cut((i + 0.5) * px, y, z[i], c.feed)
        g.rapid((idx[-1] + 0.5) * px, y, g.safe)  # ponytail: full retract per row, use local-max retract to save time
    return tgt


def preview(tgt, px, path):
    """Hillshade of what the finish cutter leaves: check this before cutting."""
    gy, gx = np.gradient(tgt, px)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    shade = np.clip((n / np.linalg.norm(n, axis=2, keepdims=True)) @ np.array([-0.4, 0.5, 0.77]), 0, 1)
    Image.fromarray((shade[::-1] * 255).astype(np.uint8)).save(path)  # flip: row 0 is Y min


def build(c):
    if c.scan.lower().endswith((".png", ".tif", ".tiff")):
        h, depth = depth_image(c.scan, c.width, c.px, c.max_depth, c.floor)
    else:
        h, depth = heightmap(orient(load_mesh(c.scan), c.flip), c.width, c.px, c.max_depth)
    g = GCode(c.dialect, c.safe_z)
    g.out += [DIALECTS[c.dialect]["head"], f"(Vriksha Studios {os.path.basename(c.scan)}: {c.width} mm wide, {depth:.1f} mm deep)"]
    g.tool(1, f"{c.rough_dia} mm flat", c.rpm)
    layers = rough(g, h, c.px, c)
    g.tool(2, f"{c.ball_dia} mm ball nose", c.rpm)
    tgt = finish(g, h, c.px, c)
    g.out += ["M5", f"G0 Z{g.safe:.3f}", "G0 X0 Y0", DIALECTS[c.dialect]["tail"]]
    stats = dict(size_mm=(h.shape[1] * c.px, h.shape[0] * c.px), depth_mm=depth, layers=layers, lines=len(g.out), minutes=g.minutes)
    return "\n".join(g.out) + "\n", tgt, stats


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scan", help="mesh (.ply .obj .stl .glb) or depth image (.png .tif)")
    p.add_argument("--width", type=float, required=True, help="real panel width in mm (X)")
    p.add_argument("--max-depth", type=float, default=20, help="deepest cut in mm; deeper reliefs are compressed")
    p.add_argument("--flip", action="store_true", help="turn the panel over if the relief faces down")
    p.add_argument("--px", type=float, default=0.5, help="height-map resolution in mm")
    p.add_argument("--floor", type=float, default=0.25, help="depth-image input only: brightness (0-1) below which is flat background")
    p.add_argument("--rough-dia", type=float, default=6)
    p.add_argument("--ball-dia", type=float, default=3)
    p.add_argument("--stepdown", type=float, default=3)
    p.add_argument("--allowance", type=float, default=0.5, help="stock left for the finish pass, mm")
    p.add_argument("--finish-stepover", type=float, default=0.4)
    p.add_argument("--feed", type=int, default=1500)
    p.add_argument("--plunge", type=int, default=400)
    p.add_argument("--rpm", type=int, default=18000)
    p.add_argument("--safe-z", type=float, default=5)
    p.add_argument("--dialect", choices=DIALECTS, default="generic")
    p.add_argument("-o", "--out", default="out.nc")
    p.add_argument("--preview", default="preview.png")
    return p.parse_args(argv)


if __name__ == "__main__":
    c = parse()
    code, tgt, s = build(c)
    open(c.out, "w").write(code)
    preview(tgt, c.px, c.preview)
    print(f"{c.out}: {s['size_mm'][0]:.0f}x{s['size_mm'][1]:.0f} mm, {s['depth_mm']:.1f} mm deep, "
          f"{s['layers']} rough layers, {s['lines']} lines, ~{s['minutes']:.0f} min cutting. Preview: {c.preview}")
