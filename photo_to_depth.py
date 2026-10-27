"""One photo -> relative depth map (16-bit PNG, white = nearest). Feed the PNG to vriksha.py.

Needs `pip install torch transformers`. Depth is a model's guess: relative, not metric, front face only.
"""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

photo, out = sys.argv[1], sys.argv[2]
img = Image.open(photo).convert("RGB")
d = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")(img)["predicted_depth"]
d = torch.nn.functional.interpolate(d.reshape(1, 1, *d.shape[-2:]), size=img.size[::-1], mode="bicubic")[0, 0].numpy()
d = (d - d.min()) / np.ptp(d)
Image.fromarray((d * 65535).astype(np.uint16)).save(out)
print(f"{out}: {img.size[0]}x{img.size[1]} depth map")
