"""Make a scan light enough for a product page: <= 100k triangles, textures <= 2048 px, <= 25 MB.

    python web_glb.py figure_m.glb figure_web.glb

The 25 MB cap is what the shop server accepts for a published .glb. Size in metres is kept as it is.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

MAX_FACES = 100_000
MAX_TEXTURE = 2048
MAX_BYTES = 25 * 1024 * 1024


def web_ready(src, dst, faces=MAX_FACES, tex=MAX_TEXTURE, max_bytes=MAX_BYTES):
    m = trimesh.load(src, force="mesh")
    colours = m.visual.vertex_colors if m.visual.kind == "vertex" else None
    tree = cKDTree(m.vertices) if colours is not None else None
    old = m.vertices
    while True:
        out = m.simplify_quadric_decimation(face_count=faces) if len(m.faces) > faces else m.copy()
        if colours is not None:  # decimation drops vertex colours: take them from the nearest original vertex
            out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=colours[tree.query(out.vertices)[1]])
        elif m.visual.kind == "texture":
            img = m.visual.material.baseColorTexture
            if img is not None and max(img.size) > tex:
                img = img.copy()
                img.thumbnail((tex, tex))
                m.visual.material.baseColorTexture = img
                out.visual = m.visual if len(out.faces) == len(m.faces) else out.visual
        Path(dst).write_bytes(out.export(file_type="glb"))
        size = Path(dst).stat().st_size
        if size <= max_bytes or faces < 5_000:
            break
        faces = int(faces * 0.7)  # still too big: thin it out and try again
    if size > max_bytes:
        sys.exit(f"Still {size / 2**20:.1f} MB at {len(out.faces)} triangles; the shop only accepts {max_bytes / 2**20:.0f} MB.")
    return len(m.faces), len(out.faces), size


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--faces", type=int, default=MAX_FACES)
    a = p.parse_args()
    before, after, size = web_ready(a.src, a.dst, a.faces)
    print(f"{before:,} -> {after:,} triangles, {size / 2**20:.1f} MB: {a.dst}")
