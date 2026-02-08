from pathlib import Path
import json, random, shutil
from tqdm import tqdm

SRC_ROOT = Path("./custom_bd100k")
OUT_ROOT = Path("./custom_bd100k/bd100k-yolo-format")

TARGET = {"train": 8000, "val": 1000, "test": 1000}
SEED = 42

MAP = {
    "car": 0,
    "person": 1,
    "rider": 1,
    "truck": 2,
    "bus": 3,
    "traffic light": 4,
    "traffic sign": 4,
}

NAMES = ["car", "person", "truck", "bus", "traffic_control"]


def list_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def yolo_line_from_rect(exterior, W, H):
    (x1, y1), (x2, y2) = exterior
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    x_min = max(0.0, min(float(x_min), float(W)))
    x_max = max(0.0, min(float(x_max), float(W)))
    y_min = max(0.0, min(float(y_min), float(H)))
    y_max = max(0.0, min(float(y_max), float(H)))

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 1e-6 or bh <= 1e-6:
        return None

    xc = x_min + bw / 2.0
    yc = y_min + bh / 2.0
    return (xc / W, yc / H, bw / W, bh / H)


def convert_one(img_path: Path, ann_path: Path, out_img: Path, out_lbl: Path):
    shutil.copy2(img_path, out_img)

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    W = int(data["size"]["width"])
    H = int(data["size"]["height"])

    lines = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            continue
        
        title = obj.get("classTitle")
        if title not in MAP:
            continue

        exterior = obj.get("points", {}).get("exterior")
        if not exterior or len(exterior) != 2:
            continue

        bbox = yolo_line_from_rect(exterior, W, H)
        if bbox is None:
            continue

        cls = MAP[title]
        xc, yc, bw, bh = bbox
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(path: Path, d: dict):
    def dump(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.append(dump(v, indent + 1))
                else:
                    lines.append(f"{sp}{k}: {v}")
            return "\n".join(lines)
        if isinstance(obj, list):
            lines = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    lines.append(f"{sp}-")
                    lines.append(dump(v, indent + 1))
                else:
                    lines.append(f"{sp}- {v}")
            return "\n".join(lines)
        return f"{sp}{obj}"

    path.write_text(dump(d) + "\n", encoding="utf-8")


def main():
    rng = random.Random(SEED)

    # output structure
    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ["train", "val", "test"]:
        img_dir = SRC_ROOT / split / "img"
        ann_dir = SRC_ROOT / split / "ann"

        imgs = list_images(img_dir)
        pairs = []
        missing_ann = 0

        for img_path in imgs:
            ann_path = ann_dir / (img_path.name + ".json")
            if ann_path.exists():
                pairs.append((img_path, ann_path))
            else:
                missing_ann += 1

        if len(pairs) > TARGET[split]:
            pairs = rng.sample(pairs, TARGET[split])

        stats[split] = {"selected": len(pairs), "missing_ann": missing_ann}

        kept_boxes = 0
        empty_labels = 0

        for img_path, ann_path in tqdm(pairs, desc=f"convert {split}"):
            out_img = OUT_ROOT / "images" / split / img_path.name
            out_lbl = OUT_ROOT / "labels" / split / (img_path.stem + ".txt")
            convert_one(img_path, ann_path, out_img, out_lbl)

            lbl_content = out_lbl.read_text(encoding="utf-8").strip()
            if lbl_content:
                kept_boxes += lbl_content.count("\n") + 1
            else:
                empty_labels += 1

        stats[split]["kept_boxes"] = kept_boxes
        stats[split]["empty_labels"] = empty_labels

    data_yaml = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    write_yaml(OUT_ROOT / "data.yaml", data_yaml)

    print("DONE. Output:", OUT_ROOT)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
