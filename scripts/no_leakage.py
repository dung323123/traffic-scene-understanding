from pathlib import Path
import shutil

ROOT = Path("bdd_seg_ready")

TRAIN_IMG = ROOT / "images" / "train"
TRAIN_MSK = ROOT / "masks" / "train"
VAL_IMG = ROOT / "images" / "val"
VAL_MSK = ROOT / "masks" / "val"
TEST_IMG = ROOT / "images" / "test"
TEST_MSK = ROOT / "masks" / "test"

OUT = Path("bdd_seg_final")
OUT_TRAIN_IMG = OUT / "images" / "train"
OUT_TRAIN_MSK = OUT / "masks" / "train"
OUT_VAL_IMG = OUT / "images" / "val"
OUT_VAL_MSK = OUT / "masks" / "val"
OUT_TEST_IMG = OUT / "images" / "test"
OUT_TEST_MSK = OUT / "masks" / "test"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def norm_uuid_from_maskstem(s: str) -> str:
    s = s.lower()
    for suffix in ["_train_id", "_val_id", "_test_id"]:
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s.split("_")[0]

def count_files(p: Path, exts):
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts) if p.exists() else 0

def main():
    test_masks = [p for p in TEST_MSK.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    test_uuids = {norm_uuid_from_maskstem(p.stem) for p in test_masks}

    for img in VAL_IMG.iterdir():
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            safe_copy(img, OUT_VAL_IMG / img.name)
    for m in VAL_MSK.iterdir():
        if m.is_file() and m.suffix.lower() == ".png":
            safe_copy(m, OUT_VAL_MSK / m.name)

    for img in TEST_IMG.iterdir():
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            uuid = img.stem.lower()
            if uuid in test_uuids:
                safe_copy(img, OUT_TEST_IMG / img.name)
    
    for m in test_masks:
        safe_copy(m, OUT_TEST_MSK / m.name)

    for img in TRAIN_IMG.iterdir():
        if not (img.is_file() and img.suffix.lower() in IMG_EXTS):
            continue
        uuid = img.stem.lower()
        if uuid in test_uuids:
            continue
        
        safe_copy(img, OUT_TRAIN_IMG / img.name)
        
        mask_path = TRAIN_MSK / f"{uuid}_train_id.png"
        if not mask_path.exists():
            cand = list(TRAIN_MSK.glob(f"{uuid}_*.png"))
            if not cand:
                raise FileNotFoundError(f"Missing mask for train image {img.name}")
            mask_path = cand[0]
        
        safe_copy(mask_path, OUT_TRAIN_MSK / mask_path.name)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "test500_manifest.txt"
    manifest.write_text("\n".join(sorted(test_uuids)), encoding="utf-8")

    print("=== FINAL SPLIT (NO-LEAK) ===")
    print("Train images:", count_files(OUT_TRAIN_IMG, IMG_EXTS), "Train masks:", count_files(OUT_TRAIN_MSK, {'.png'}))
    print("Val   images:", count_files(OUT_VAL_IMG, IMG_EXTS),   "Val masks:",   count_files(OUT_VAL_MSK, {'.png'}))
    print("Test  images:", count_files(OUT_TEST_IMG, IMG_EXTS),  "Test masks:",  count_files(OUT_TEST_MSK, {'.png'}))
    print("Manifest:", manifest.resolve())
    print("Output:", OUT.resolve())

if __name__ == "__main__":
    main()
