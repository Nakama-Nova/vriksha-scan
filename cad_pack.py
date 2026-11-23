"""The "CAD folder": everything a designer needs from one flat carving, as a single zip.

A 3D model (STL, OBJ), the 2D outline of the carved area (DXF and SVG, in millimetres), the depth
map (16-bit grey, for CAM programs that read a height map) and a plain read-me. The outline comes
from the depth map itself: a carving cut out of a photo has background (0) outside its outline.
"""

import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import relief

OUTLINE_SIDE = 800  # the mask is looked at this big (longest side), so a corner is found to about 1 px of it
FULL = 0.97  # a map this full of carving is a plain rectangle: no outline to trace


def _simplify(points, tol):
    """Douglas-Peucker on an open polyline; the loop is cut at its two farthest corners by the caller."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.hypot(*ab)
    d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n if n else np.hypot(*(p - a).T)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.array([a, b])
    return np.vstack([_simplify(p[: i + 1], tol)[:-1], _simplify(p[i:], tol)])


def outline_px(depth):
    """The outer outline of the carved area in a depth map: corner points (x, y) in pixels of the map
    at its own size, clockwise as seen on screen. A full map gives its own four corners."""
    h, w = depth.shape
    scale = min(1.0, OUTLINE_SIDE / max(h, w))
    small = relief.shrink(depth, OUTLINE_SIDE) if scale < 1 else depth
    sh, sw = small.shape
    mask = ndi.binary_fill_holes(ndi.binary_closing(small > 0, iterations=2))
    lab, n = ndi.label(mask)
    if n == 0 or mask.mean() >= FULL:
        return [(0, 0), (w, 0), (w, h), (0, h)]
    mask = lab == (1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1)))))
    # the pixel edges between carving and background, each pointing so the carving is on its right
    nxt = {}
    pad = np.pad(mask, 1)
    for y, x in zip(*np.nonzero(mask)):
        if not pad[y, x + 1]:
            nxt.setdefault((x, y), []).append((x + 1, y))
        if not pad[y + 1, x + 2]:
            nxt.setdefault((x + 1, y), []).append((x + 1, y + 1))
        if not pad[y + 2, x + 1]:
            nxt.setdefault((x + 1, y + 1), []).append((x, y + 1))
        if not pad[y + 1, x]:
            nxt.setdefault((x, y + 1), []).append((x, y))
    loops = []
    while nxt:  # follow the edges round, once per loop
        start = next(iter(nxt))
        loop, cur = [start], start
        while True:
            ends = nxt.get(cur)
            if not ends:
                break
            step = ends.pop()
            if not ends:
                del nxt[cur]
            if step == start:
                break
            loop.append(step)
            cur = step
        loops.append(loop)
    loop = np.array(max(loops, key=len), dtype=float)  # the outer boundary is the longest
    far = int(np.argmax(np.hypot(*(loop - loop[0]).T)))  # cut the loop where it is widest, simplify both halves
    a, b = _simplify(loop[: far + 1], 1.0), _simplify(np.vstack([loop[far:], loop[:1]]), 1.0)
    pts = np.vstack([a[:-1], b[:-1]]) * np.array([w / sw, h / sh])
    return [tuple(map(float, p)) for p in pts]


def outline_mm(depth, width_mm):
    """The outline in millimetres, y up (as CAD wants it), with (0, 0) at the area's lower-left corner.
    Returns (points, width, height)."""
    h, w = depth.shape
    k = width_mm / w
    pts = [(x * k, (h - y) * k) for x, y in outline_px(depth)]
    return pts, width_mm, h * k


def dxf(points):
    """A closed polyline on layer OUTLINE, as a plain R12 DXF that any CAD or CAM program opens."""
    rows = ["0", "SECTION", "2", "ENTITIES", "0", "POLYLINE", "8", "OUTLINE", "66", "1", "70", "1"]
    for x, y in points:
        rows += ["0", "VERTEX", "8", "OUTLINE", "10", f"{x:.3f}", "20", f"{y:.3f}", "30", "0.0"]
    rows += ["0", "SEQEND", "8", "OUTLINE", "0", "ENDSEC", "0", "EOF"]
    return "\n".join(rows) + "\n"


def svg(points, width, height):
    """The same outline as an SVG in millimetres (y flipped back, SVG counts down)."""
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.3f},{height - y:.3f}" for i, (x, y) in enumerate(points)) + " Z"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}mm" height="{height:.1f}mm" '
        f'viewBox="0 0 {width:.3f} {height:.3f}"><path d="{path}" fill="none" stroke="black" stroke-width="0.5"/></svg>\n'
    )


def readme(title, width, height, depth_mm=relief.RELIEF_DEPTH_MM, floor=relief.FLOOR):
    return f"""{title}
{'=' * len(title)}

All sizes are in millimetres. The carved area is {width:.0f} x {height:.0f} mm and the relief is
about {depth_mm:g} mm deep. The carving was read from a photo, so the depth is an estimate:
check it against the real piece before cutting anything.

Files
-----
model.stl, model.obj   The carving as a 3D surface. Open either in any CAD, CAM or mesh program
                       (FreeCAD, Fusion, Blender, MeshLab...). X is the width, Y is up, the top
                       of the relief is Z = 0 and the flat background is at Z = -{depth_mm:g}.
outline.dxf            The outline of the carved area as a closed 2D polyline, in millimetres, in the
                       same frame as the model: (0, 0) is the lower-left corner of the area. Use it
                       to cut the shape out or to draw round the carving.
outline.svg            The same outline as an SVG, for a drawing or laser program.
depth-map.png          The master height map (16-bit grey, white = highest). CAM programs that
                       read a height map can cut from it: set the width to {width:.0f} mm and the
                       depth to {depth_mm:g} mm. Everything darker than {floor * 100:.0f}% of the height is flat
                       background.

The model and the height map are the same carving, so use whichever your program prefers.
"""


def pack(depth, width_mm, meshes, depth_png, out, title):
    """Write cad-folder.zip in `out` with `meshes` (paths of the STL and OBJ), the outline files, the
    stored depth map and the read-me. Returns the path of the zip."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pts, width, height = outline_mm(depth, width_mm)
    path = out / "cad-folder.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", readme(title, width, height))
        z.writestr("outline.dxf", dxf(pts))
        z.writestr("outline.svg", svg(pts, width, height))
        for mesh in meshes:
            z.write(mesh, f"model{Path(mesh).suffix}")
        z.write(depth_png, "depth-map.png")
    return path
