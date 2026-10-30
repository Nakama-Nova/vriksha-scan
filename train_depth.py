"""Fine-tune the depth model on your own carvings.

Training data = folder of pairs: `name.jpg` (straight-on photo) + `name.depth.png` (the TRUE depth of
that same view, 16-bit grey, white = nearest: the format relief.py / vriksha.py already use).
The loss ignores the depth's scale and offset (the model's depth is relative) but punishes wrong
edges and wrong fine detail. The tuned model is only saved if it beats the original on held-out pairs.

    python train_depth.py pairs/ --out depth-tuned
    set VRIKSHA_DEPTH_MODEL=depth-tuned        (PowerShell: $env:VRIKSHA_DEPTH_MODEL='depth-tuned')
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import relief

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
CROP = 518  # 37 x 14: the size the model reads, so training crops look like what it sees at run time


def find_pairs(folder):
    pairs = []
    for p in sorted(Path(folder).iterdir()):
        d = p.with_name(p.stem + ".depth.png")
        if p.suffix.lower() in relief.PHOTO_TYPES and ".depth" not in p.stem and d.exists():
            pairs.append((p, d))
    return pairs


def load(photo, depth, crop, rng=None):
    """One pair -> (3,crop,crop) normalised photo and (crop,crop) depth in 0..1. Random crop and flip if rng."""
    img = Image.open(photo).convert("RGB")
    d = Image.fromarray(np.asarray(Image.open(depth), dtype=np.float32) / 65535, mode="F").resize(img.size, Image.BILINEAR)
    s = crop / min(img.size)  # short side -> crop, so a crop is what the model reads in one go
    size = (max(crop, round(img.width * s)), max(crop, round(img.height * s)))
    img, d = img.resize(size, Image.BICUBIC), d.resize(size, Image.BILINEAR)
    x, y = ((rng.randint(0, size[0] - crop), rng.randint(0, size[1] - crop)) if rng else ((size[0] - crop) // 2, (size[1] - crop) // 2))
    box = (x, y, x + crop, y + crop)
    a, b = (np.asarray(t.crop(box), np.float32) for t in (img, d))
    if rng and rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    return torch.from_numpy(((a / 255 - MEAN) / STD).transpose(2, 0, 1).copy()), torch.from_numpy(b.copy())


def loss_fn(pred, tgt):
    """Shift-and-scale-invariant L1 on depth plus L1 on its gradients (the fine detail)."""
    p, g = pred.flatten(1), tgt.flatten(1)
    pc, gm = p - p.mean(1, keepdim=True), g.mean(1, keepdim=True)
    s = ((pc * (g - gm)).mean(1, keepdim=True) / pc.pow(2).mean(1, keepdim=True).clamp_min(1e-6))  # unclamped so gradients always flow; the pretrained start keeps depth the right way up
    a = (s * pc + gm).view_as(tgt)
    grads = lambda t: (t[:, :, 1:] - t[:, :, :-1], t[:, 1:] - t[:, :-1])
    return (a - tgt).abs().mean() + sum((u - v).abs().mean() for u, v in zip(grads(a), grads(tgt)))


@torch.no_grad()
def evaluate(model, items, crop):
    model.eval()
    return float(np.mean([loss_fn(model(pixel_values=x[None]).predicted_depth, y[None]).item() for x, y in (load(*p, crop) for p in items)]))


def train(model, pairs, epochs=10, lr=5e-6, batch=2, crop=CROP, out=None, seed=0):
    """Returns (loss before, best loss after) on the held-out pairs; saves to `out` only on improvement."""
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 photo + depth pairs (one is held out to check the result).")
    val = pairs[::10]
    fit = [p for p in pairs if p not in val]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if n.startswith("backbone")], "lr": lr}, {"params": head, "lr": lr * 10}])
    first = best = evaluate(model, val, crop)
    print(f"{len(fit)} pairs to learn from, {len(val)} held out. Held-out loss before: {first:.4f}")
    for e in range(epochs):
        model.train()
        rng.shuffle(fit)
        for i in range(0, len(fit), batch):
            xs, ys = zip(*(load(*p, crop, rng) for p in fit[i:i + batch]))
            loss = loss_fn(model(pixel_values=torch.stack(xs)).predicted_depth, torch.stack(ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
        now = evaluate(model, val, crop)
        better = now < best
        print(f"epoch {e + 1}/{epochs}: held-out loss {now:.4f}" + ("  (best, saved)" if better and out else ""))
        if better:
            best = now
            if out:
                model.save_pretrained(out)
    return first, best


def main():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pairs", help="folder of name.jpg + name.depth.png pairs")
    p.add_argument("--out", default="depth-tuned", help="folder for the tuned model")
    p.add_argument("--base", default=relief.MODEL, help="model to start from (a hub name or an earlier tuned folder)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-6, help="for the image encoder; the depth head learns 10x faster")
    p.add_argument("--batch", type=int, default=2)
    a = p.parse_args()
    pairs = find_pairs(a.pairs)
    if not pairs:
        raise SystemExit(f"No pairs in {a.pairs}: each photo needs a matching name.depth.png next to it.")
    AutoImageProcessor.from_pretrained(a.base).save_pretrained(a.out)  # so the folder loads like a hub model
    first, best = train(AutoModelForDepthEstimation.from_pretrained(a.base), pairs, a.epochs, a.lr, a.batch, out=a.out)
    if best < first:
        print(f"Held-out loss {first:.4f} -> {best:.4f}. Use it: VRIKSHA_DEPTH_MODEL={a.out}")
    else:
        print("The tuned model did not beat the original on the held-out pairs, so it was not saved. Add more or cleaner pairs.")


if __name__ == "__main__":
    main()
