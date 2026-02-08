import numpy as np
from PIL import Image
from pathlib import Path

p = next(Path("bdd_seg_ready/masks/train").glob("*.png"))
arr = np.array(Image.open(p))

print("shape:", arr.shape, "dtype:", arr.dtype)
print("min/max:", arr.min(), arr.max())
print("unique (first 30):", np.unique(arr)[:30])
