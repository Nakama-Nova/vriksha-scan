"""One photo -> relative depth map (16-bit PNG, white = nearest). Feed the PNG to vriksha.py.

Needs `pip install torch transformers`. Depth is a model's guess: relative, not metric, front face only.
Same detail-preserving estimate as the workshop worker (see relief.py).
"""
import sys

import relief

photo, out = sys.argv[1], sys.argv[2]
d = relief.estimate_depth(photo)
relief.save_depth_png(d, out)
print(f"{out}: {d.shape[1]}x{d.shape[0]} depth map")
