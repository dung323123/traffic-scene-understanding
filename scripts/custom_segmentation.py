from __future__ import annotations
from pathlib import Path
import shutil, random, re
from dataclasses import dataclass

BASE_10K = Path("10k")
SEG_ROOT = Path("bdd100k_segmentation")
OUT = Path("bdd_seg_ready")

MASK_SOURCE = "labels"
SEED = 42

DO_SPLIT_TRAIN_TO_TEST = True
N_TEST_FROM_TRAIN = 500

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXT = ".png"
def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def list_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def count_files(dir_path: Path, exts: set[str]) -> int:
    return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts) if dir_path.exists() else 0

def print_seg_counts(seg_root: Path) -> None:
    print("=== SEGMENTATION FILE COUNTS ===")
    for top in ["labels", "color_labels"]:
        top_path = seg_root / top
        if not top_path.exists():
            print(f"[MISSING] {top_path}")
            continue
        print(f"\n[{top}] -> {top_path}")
        total = 0
        for split in ["train", "val", "test"]:
            sp = top_path / split
            if sp.exists():
                n = count_files(sp, {".png"})
                total += n
                print(f"  {split:<5}: {n}")
        print(f"  TOTAL : {total}")
    print()

TAIL_RE = re.compile(r"_(train|val|test)_id$", flags=re.IGNORECASE)


def norm_key(stem: str) -> str:
    s = stem.strip().lower()
    s = TAIL_RE.sub("", s)  # remove _train_id / _val_id / _test_id
    if "_" in s:
        s = s.split("_")[0]
    return s

def build_mask_index(seg_root: Path, mask_source: str) -> tuple[dict[str, Path], int, int]:
    src_root = seg_root / mask_source
    if not src_root.exists():
        raise FileNotFoundError(f"Mask source folder not found: {src_root}")

    pngs = list(src_root.rglob("*.png"))
    idx: dict[str, Path] = {}
    collisions = 0

    for p in pngs:
        key = norm_key(p.stem)
        if key in idx:
            collisions += 1
            # prefer shallower / more specific path (usually split folder)
            if len(p.parts) < len(idx[key].parts):
                idx[key] = p
        else:
            idx[key] = p

    return idx, len(pngs), collisions


@dataclass
class MapReport:
    images: int
    mask_ok: int
    mask_missing: int

def map_base10k_to_seg(base_10k: Path, seg_root: Path, out_root: Path, mask_source: str) -> dict[str, MapReport]:
    mask_idx, total_png, collisions = build_mask_index(seg_root, mask_source)

    print("=== MASK INDEX ===")
    print("Mask source       :", (seg_root / mask_source).resolve())
    print("PNG scanned       :", total_png)
    print("Unique mask keys  :", len(mask_idx))
    print("Key collisions    :", collisions)
    print()

    # Debug sample
    sample_masks = sorted(list((seg_root / mask_source).rglob("*.png")))[:5]
    sample_imgs = sorted(list_images(base_10k / "train"))[:5]
    print("=== DEBUG SAMPLES ===")
    print("Sample image stems:", [p.stem for p in sample_imgs])
    print("Sample image keys :", [norm_key(p.stem) for p in sample_imgs])
    print("Sample mask stems :", [p.stem for p in sample_masks])
    print("Sample mask keys  :", [norm_key(p.stem) for p in sample_masks])
    print()

    report: dict[str, MapReport] = {}

    for split in ["train", "val", "test"]:
        img_dir = base_10k / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing base split folder: {img_dir}")

        out_img = out_root / "images" / split
        out_msk = out_root / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        imgs = list_images(img_dir)
        ok = 0
        miss = 0
        miss_examples = []

        for img in imgs:
            safe_copy(img, out_img / img.name)

            key = norm_key(img.stem)
            m = mask_idx.get(key)

            if m is not None:
                safe_copy(m, out_msk / m.name)
                ok += 1
            else:
                miss += 1
                if len(miss_examples) < 10:
                    miss_examples.append((img.name, img.stem, key))

        report[split] = MapReport(images=len(imgs), mask_ok=ok, mask_missing=miss)

        if split in ["train", "val"] and ok == 0:
            print(f"!!! WARNING: split={split} mask_ok=0")
            print("Missing examples (img_name, img_stem, key):")
            for ex in miss_examples:
                print(" ", ex)
            print()

    print("=== MAP REPORT ===")
    for sp, r in report.items():
        print(f"{sp:<5} {{'images': {r.images}, 'mask_ok': {r.mask_ok}, 'mask_missing': {r.mask_missing}}}")
    print("Output:", out_root.resolve())
    print()
    return report

def split_train_to_test_with_masks(out_root: Path, n_test: int, seed: int) -> None:
    random.seed(seed)

    img_train = out_root / "images" / "train"
    msk_train = out_root / "masks" / "train"
    img_test = out_root / "images" / "test"
    msk_test = out_root / "masks" / "test"

    img_test.mkdir(parents=True, exist_ok=True)
    msk_test.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_train.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    train_masks = [p for p in msk_train.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    train_mask_idx = {norm_key(m.stem): m for m in train_masks}

    paired = [(img, train_mask_idx[k]) for img in imgs if (k := norm_key(img.stem)) in train_mask_idx]

    if len(paired) < n_test:
        raise ValueError(f"Not enough (image,mask) pairs to sample {n_test}. Available: {len(paired)}")

    random.shuffle(paired)
    for img, msk in paired[:n_test]:
        safe_copy(img, img_test / img.name)
        safe_copy(msk, msk_test / msk.name)

    print("=== TRAIN->TEST (COPY) REPORT ===")
    print("Requested test pairs:", n_test)
    print("Copied test pairs   :", len(paired[:n_test]))
    print("Test images dir     :", img_test.resolve())
    print("Test masks dir      :", msk_test.resolve())
    print()

def main():
    print_seg_counts(SEG_ROOT)

    rep = map_base10k_to_seg(
        base_10k=BASE_10K,
        seg_root=SEG_ROOT,
        out_root=OUT,
        mask_source=MASK_SOURCE,
    )

    if DO_SPLIT_TRAIN_TO_TEST:
        if rep["train"].mask_ok == 0:
            print("SKIP split_train_to_test_with_masks: train mask_ok=0 (chưa map được cặp image-mask).")
        else:
            split_train_to_test_with_masks(OUT, n_test=N_TEST_FROM_TRAIN, seed=SEED)

    print("=== SUMMARY ===")
    print("Mask source used:", MASK_SOURCE)
    print("Train masks :", count_files(OUT / "masks" / "train", {".png"}))
    print("Val masks   :", count_files(OUT / "masks" / "val", {".png"}))
    print("Test masks  :", count_files(OUT / "masks" / "test", {".png"}))
    print("OUT         :", OUT.resolve())
    print()

if __name__ == "__main__":
    main()