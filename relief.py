"""Flat-carving ("relief") route: one straight-on photo -> depth map -> relief mesh.

Runs on any PC, no GPU: the depth model works on the CPU. The depth is a model's guess (relative,
front face only), so sizes come from the real width entered on the scan. For an object that is
photographed all the way around, use the full COLMAP route in workshop_worker.py instead.
"""

import functools

import numpy as np
import trimesh
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
PHOTO_SIDE = 1024  # the depth model is run at this size; bigger only costs time
DEPTH_SIDE = 2048  # largest depth map kept (G-code resamples it to the cutter's resolution anyway)
MESH_SIDE = 256  # relief mesh grid: about 65,000 vertices, small enough for the server
RELIEF_DEPTH_MM = 15.0  # how deep the carving is cut, nearest to farthest
FLOOR = 0.25  # depth below this share is treated as flat background (same default as vriksha.py)
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}


def sharpness(path) -> float:
    """How much fine detail a photo has (blurry photos score low)."""
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((640, 640))
        g = np.asarray(im, dtype=float)
    gy, gx = np.gradient(g)
    return float(np.var(gx) + np.var(gy))


def pick_front(files):
    """The sharpest photo. A flat carving needs one straight-on photo; if several are sent, the
    upload order is not kept, so the sharpest is the best guess."""
    photos = [f for f in files if f.suffix.lower() in PHOTO_TYPES]
    if not photos:
        raise RuntimeError("There is no photo to read the carving from.")
    return max(photos, key=sharpness)


@functools.lru_cache(maxsize=1)
def _depth_pipeline():
    try:
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "Flat-carving mode needs torch and transformers on this PC: pip install -r requirements-relief.txt"
        ) from None
    return pipeline("depth-estimation", model=MODEL)


def estimate_depth(photo) -> np.ndarray:
    """One photo -> relative depth, 0 (far) to 1 (nearest), same height/width as the (shrunk) photo."""
    import torch

    img = Image.open(photo).convert("RGB")
    img.thumbnail((PHOTO_SIDE, PHOTO_SIDE))
    d = _depth_pipeline()(img)["predicted_depth"]
    d = torch.nn.functional.interpolate(d.reshape(1, 1, *d.shape[-2:]), size=img.size[::-1], mode="bicubic")[0, 0].numpy()
    span = float(np.ptp(d))
    return ((d - d.min()) / span if span > 0 else np.zeros_like(d)).astype(np.float32)


def shrink(depth, side) -> np.ndarray:
    """Resize so the longest side is at most `side` pixels (averaging, so detail is kept)."""
    h, w = depth.shape
    scale = side / max(h, w)
    if scale >= 1:
        return depth
    img = Image.fromarray(np.ascontiguousarray(depth, dtype=np.float32), mode="F")
    return np.asarray(img.resize((max(2, round(w * scale)), max(2, round(h * scale))), Image.BOX), dtype=np.float32)


def save_depth_png(depth, path):
    """16-bit grey image, white = nearest: the input vriksha.py reads for G-code."""
    Image.fromarray((np.clip(depth, 0, 1) * 65535).astype(np.uint16)).save(path)
    return path


def save_preview_png(depth, path):
    """8-bit version for looking at (small enough to upload as a picture)."""
    Image.fromarray((np.clip(depth, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def relief_mesh(depth, width_mm, depth_mm=RELIEF_DEPTH_MM, floor=FLOOR, side=MESH_SIDE) -> trimesh.Trimesh:
    """Depth map -> height-field mesh in millimetres. X is the width, Y is up, the top of the relief
    is Z=0 and the background is at Z=-depth_mm (the same frame vriksha.py cuts in)."""
    d = shrink(depth, side)
    d = d / d.max() if d.max() > 0 else d  # nearest = 1, as vriksha.py reads a depth image
    d = np.clip((d - floor) / (1 - floor), 0, 1)  # everything below the floor becomes flat background
    ny, nx = d.shape
    px = width_mm / (nx - 1)
    xs = np.arange(nx) * px
    ys = (ny - 1 - np.arange(ny)) * px  # photo row 0 is the top, so it gets the largest Y
    x, y = np.meshgrid(xs, ys)
    vertices = np.column_stack([x.ravel(), y.ravel(), ((d - 1.0) * depth_mm).ravel()])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, e = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, e])])  # normals face +Z
    return trimesh.Trimesh(vertices, faces, process=False)
