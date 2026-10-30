"""Flat-carving ("relief") route: one straight-on photo -> depth map -> relief mesh.

Runs on any PC, no GPU: the depth model works on the CPU. The depth is a model's guess (relative,
front face only), so sizes come from the real width entered on the scan. For an object that is
photographed all the way around, use the full COLMAP route in workshop_worker.py instead.
"""

import functools

import numpy as np
import trimesh
from PIL import Image, ImageOps
from scipy import ndimage as ndi

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
PHOTO_SIDE = 2048  # the photo is read at this size; the model only sees ~518 px at a time, so tiles carry the fine detail
TILE = 640  # side of one photo tile fed to the model (it is shown at about its native pixels, so nothing is blurred away)
TILE_OVERLAP = 0.5  # share of a tile shared with its neighbour; the overlap is blended so no seams show
DETAIL_GAIN = 0.15  # how strongly the photo's own fine shading (grooves, ridges, strands) is cut into the depth; 0 = model only
DETAIL_SIGMA = 3.0  # pixels: shading finer than this counts as detail, anything broader is left to the depth model
DEPTH_SIDE = 2048  # largest depth map kept (G-code resamples it to the cutter's resolution anyway)
MESH_SIDE = 1024  # relief mesh grid: about 780,000 vertices; export_models simplifies it if a file is too big for the server
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


def _model_depth(img) -> np.ndarray:
    """The model's raw depth for one image, resized back to the image's own size."""
    import torch

    d = _depth_pipeline()(img)["predicted_depth"]
    return torch.nn.functional.interpolate(d.reshape(1, 1, *d.shape[-2:]), size=img.size[::-1], mode="bicubic")[0, 0].numpy()


def _starts(n, size, step):
    """Tile start positions that cover 0..n with the last tile flush against the edge."""
    if n <= size:
        return [0]
    return sorted({*range(0, n - size, step), n - size})


def estimate_depth(photo) -> np.ndarray:
    """One photo -> relative depth, 0 (far) to 1 (nearest), same height/width as the (shrunk) photo.

    The model shows the whole photo at about 518 px, which smooths away carved detail. So the photo is
    also read tile by tile at its own resolution; each tile is fitted to the whole-photo depth (the
    model's scale differs per tile) and the overlaps are blended. The whole-photo depth keeps the
    overall shape, the tiles add the fine detail."""
    img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")  # phone photos are stored sideways
    img.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.LANCZOS)
    w, h = img.size
    whole = _model_depth(img)
    if max(w, h) <= TILE:
        return _unit(whole)
    step = max(1, int(TILE * (1 - TILE_OVERLAP)))
    fine, weight = np.zeros_like(whole), np.zeros_like(whole)
    ramp = np.hanning(TILE + 2)[1:-1].astype(np.float32)  # blend weight: 0 at a tile edge, 1 in its middle
    for y in _starts(h, TILE, step):
        for x in _starts(w, TILE, step):
            th, tw = min(TILE, h - y), min(TILE, w - x)
            tile = _model_depth(img.crop((x, y, x + tw, y + th)))
            ref = whole[y : y + th, x : x + tw]
            a, b = np.polyfit(tile.ravel()[::7], ref.ravel()[::7], 1)  # tile -> whole-photo scale
            k = np.outer(ramp[:th], ramp[:tw]) + 1e-6
            fine[y : y + th, x : x + tw] += (a * tile + b) * k
            weight[y : y + th, x : x + tw] += k
    fine /= weight
    # shape from the whole photo, detail (the fine ripples) from the tiles
    sigma = TILE / 8
    return _unit(shade_detail(ndi.gaussian_filter(whole, sigma) + fine - ndi.gaussian_filter(fine, sigma), img))


def shade_detail(depth, img, gain=DETAIL_GAIN) -> np.ndarray:
    """Add the photo's fine light/dark detail to the depth. A depth model smooths away carved strands and
    ridges, but a lit carving still shows them as shading: darker in the grooves, lighter on the ridges.
    Only fine detail is added (broad lighting is removed), scaled to the depth's own 0..1 range.
    Painted colour pattern on a flat surface would also show up, so keep the gain low for paintings."""
    lum = np.asarray(img.convert("L"), dtype=np.float32)
    fine = lum - ndi.gaussian_filter(lum, DETAIL_SIGMA)
    top = np.percentile(np.abs(fine), 99)
    return depth + gain * np.clip(fine / top, -1, 1) * float(np.ptp(depth)) if top > 0 else depth


def _unit(d) -> np.ndarray:
    """Stretch to 0..1. The top and bottom 0.1% are clipped first, so one stray speck cannot flatten the carving."""
    lo, hi = np.percentile(d, [0.1, 99.9])
    return (np.clip(d - lo, 0, None) / (hi - lo) if hi > lo else np.zeros_like(d)).clip(0, 1).astype(np.float32)


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
