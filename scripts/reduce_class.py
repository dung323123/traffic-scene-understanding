from pathlib import Path
import numpy as np
from PIL import Image
import shutil

SRC = Path("bdd_seg_final")
DST = Path("bdd_seg_6cls_final")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IGNORE = 255

ROAD = 1
SIDEWALK = 2
PERSON = 12
RIDER = 13
VEH_IDS = {14, 15, 16, 17, 18, 19}

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def get_mask_uuid(stem: str) -> str:
    s = stem.lower()
    for suf in ["_train_id", "_val_id", "_test_id"]:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s.split("_")[0]

def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == IGNORE] = IGNORE
    out[mask == ROAD] = 1
    out[mask == SIDEWALK] = 2
    out[mask == PERSON] = 3
    out[mask == RIDER] = 4
    for vid in VEH_IDS:
        out[mask == vid] = 5
    return out

def main():
    for sp in SPLITS:
        (DST / "images" / sp).mkdir(parents=True, exist_ok=True)
        (DST / "masks" / sp).mkdir(parents=True, exist_ok=True)

    for sp in SPLITS:
        img_dir = SRC / "images" / sp
        msk_dir = SRC / "masks" / sp
        out_img_dir = DST / "images" / sp
        out_msk_dir = DST / "masks" / sp

        imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        masks = [p for p in msk_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        mask_idx = {get_mask_uuid(p.stem): p for p in masks}

        ok = 0
        miss = 0
        unique_classes = set()

        for img in imgs:
            uuid = img.stem.lower()
            m = mask_idx.get(uuid)
            if m is None:
                miss += 1
                continue

            safe_copy(img, out_img_dir / img.name)

            arr = np.array(Image.open(m))
            out = remap_mask(arr)
            unique_classes.update(np.unique(out).tolist())

            Image.fromarray(out).save(out_msk_dir / f"{uuid}.png")
            ok += 1

        print(f"[{sp}] paired_ok={ok} missing_mask={miss} classes={sorted(unique_classes)}")

    print("Output:", DST.resolve())

if __name__ == "__main__":
    main()
