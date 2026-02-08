import random
import shutil
from pathlib import Path

BASE = Path("custom_bd100k")
YOLO = BASE / "bdd100k-yolo-format"

N_MOVE = 10000
SEED = 42

def list_images(d: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts])

def main():
    random.seed(SEED)

    test_img = YOLO / "images" / "test"
    test_lab = YOLO / "labels" / "test"
    train_img = YOLO / "images" / "train"
    train_lab = YOLO / "labels" / "train"

    for p in [test_img, test_lab, train_img, train_lab]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    imgs = list_images(test_img)
    if len(imgs) < N_MOVE:
        raise RuntimeError(f"YOLO test only has {len(imgs)} images, not enough to move {N_MOVE}")

    random.shuffle(imgs)
    chosen = imgs[:N_MOVE]

    miss_label = 0
    print(f"[Move] Moving {N_MOVE} samples from YOLO test -> YOLO train (seed={SEED})")

    for img_path in chosen:
        stem = img_path.stem
        lab_path = test_lab / f"{stem}.txt"

        shutil.move(str(img_path), str(train_img / img_path.name))

        if lab_path.exists():
            shutil.move(str(lab_path), str(train_lab / lab_path.name))
        else:
            miss_label += 1
            (train_lab / f"{stem}.txt").write_text("", encoding="utf-8")

    new_train = len(list_images(train_img))
    new_test = len(list_images(test_img))

    print(f"[Done] Moved samples: {N_MOVE}")
    print(f"Missing labels (created empty): {miss_label}")
    print(f"New YOLO train images: {new_train}")
    print(f"New YOLO test  images: {new_test}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
