from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge classroom_laying COCO exports into YOLO human detection data and extract posture crops."
    )
    parser.add_argument("--base-detection", default="training_yolo26_boxes")
    parser.add_argument("--laying-root", default="data/labeled")
    parser.add_argument("--detection-output", default="training_yolo26_boxes_with_laying")
    parser.add_argument("--base-posture", default="classification_splits/posture_binary")
    parser.add_argument("--posture-output", default="classification_splits/posture_with_laying")
    parser.add_argument("--source-posture-folder", default="PostureClasssification.folder/train/person_laying")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_clean_dir(path: Path, workspace: Path, overwrite: bool) -> None:
    path = path.resolve()
    if not str(path).lower().startswith(str(workspace).lower()):
        raise RuntimeError(f"Refusing to modify path outside workspace: {path}")
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists, pass --overwrite to replace: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ignore = shutil.ignore_patterns("*.cache", "__pycache__", ".ipynb_checkpoints")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def find_laying_exports(root: Path) -> list[Path]:
    return sorted(root.glob("*classroom_laying*/*/_annotations.coco.json"))


def original_stem(image: dict) -> str:
    extra = image.get("extra") or {}
    name = extra.get("name") or image.get("file_name", "")
    return Path(name).stem


def load_coco_records(coco_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen_files: set[Path] = set()
    for coco_path in coco_paths:
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {cat["id"]: str(cat.get("name", "")).lower() for cat in data.get("categories", [])}
        images = {img["id"]: img for img in data.get("images", [])}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            if categories.get(ann.get("category_id")) == "human":
                anns_by_image[ann["image_id"]].append(ann)

        image_dir = coco_path.parent
        for image_id, image in images.items():
            src = (image_dir / image["file_name"]).resolve()
            if src in seen_files:
                continue
            seen_files.add(src)
            records.append(
                {
                    "source_image": src,
                    "file_name": image["file_name"],
                    "original_stem": original_stem(image),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": anns_by_image.get(image_id, []),
                }
            )
    return records


def split_records(records: list[dict], seed: int, train_frac: float, val_frac: float) -> dict[str, list[dict]]:
    groups = sorted({record["original_stem"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_train = round(len(groups) * train_frac)
    n_val = round(len(groups) * val_frac)
    split_by_group = {}
    for group in groups[:n_train]:
        split_by_group[group] = "train"
    for group in groups[n_train : n_train + n_val]:
        split_by_group[group] = "val"
    for group in groups[n_train + n_val :]:
        split_by_group[group] = "test"

    out = {split: [] for split in SPLITS}
    for record in records:
        out[split_by_group[record["original_stem"]]].append(record)
    return out


def yolo_lines(record: dict) -> list[str]:
    width = record["width"]
    height = record["height"]
    lines = []
    for ann in record["annotations"]:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        x1 = max(0.0, min(width, x))
        y1 = max(0.0, min(height, y))
        x2 = max(0.0, min(width, x + w))
        y2 = max(0.0, min(height, y + h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
    return lines


def add_laying_to_detection(base: Path, output: Path, split_records_by_name: dict[str, list[dict]]) -> None:
    copy_tree(base, output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for record in split_records_by_name[split]:
            out_stem = f"classroom_laying_{record['original_stem']}"
            ext = record["source_image"].suffix.lower()
            image_dst = output / "images" / split / f"{out_stem}{ext}"
            label_dst = output / "labels" / split / f"{out_stem}.txt"
            shutil.copy2(record["source_image"], image_dst)
            lines = yolo_lines(record)
            label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- Human\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")


def extract_crops(
    split_records_by_name: dict[str, list[dict]],
    posture_output: Path,
    source_posture_folder: Path,
) -> list[dict]:
    manifest: list[dict] = []
    source_posture_folder.mkdir(parents=True, exist_ok=True)
    crop_counts_by_stem: dict[str, int] = defaultdict(int)

    for split in SPLITS:
        class_dir = posture_output / split / "person_laying"
        class_dir.mkdir(parents=True, exist_ok=True)
        for record in split_records_by_name[split]:
            with Image.open(record["source_image"]) as img:
                img = img.convert("RGB")
                width, height = img.size
                for ann in record["annotations"]:
                    x, y, w, h = [float(v) for v in ann["bbox"]]
                    left = max(0, int(x))
                    top = max(0, int(y))
                    right = min(width, int(round(x + w)))
                    bottom = min(height, int(round(y + h)))
                    if right <= left or bottom <= top:
                        continue
                    crop_counts_by_stem[record["original_stem"]] += 1
                    crop_idx = crop_counts_by_stem[record["original_stem"]]
                    crop_name = f"classroom_laying_{record['original_stem']}_crop_{crop_idx:02d}.jpg"
                    crop = img.crop((left, top, right, bottom))
                    crop.save(class_dir / crop_name, quality=95)
                    crop.save(source_posture_folder / crop_name, quality=95)
                    manifest.append(
                        {
                            "split": split,
                            "class_name": "person_laying",
                            "crop_file": crop_name,
                            "source_image": str(record["source_image"]),
                            "original_stem": record["original_stem"],
                            "bbox_x": f"{x:.2f}",
                            "bbox_y": f"{y:.2f}",
                            "bbox_w": f"{w:.2f}",
                            "bbox_h": f"{h:.2f}",
                        }
                    )
    return manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_images(path: Path) -> dict[str, int]:
    counts = {}
    for split in SPLITS:
        folder = path / "images" / split
        counts[split] = len([p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS]) if folder.exists() else 0
    return counts


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    base_detection = Path(args.base_detection)
    laying_root = Path(args.laying_root)
    detection_output = Path(args.detection_output)
    base_posture = Path(args.base_posture)
    posture_output = Path(args.posture_output)
    source_posture_folder = Path(args.source_posture_folder)

    coco_paths = find_laying_exports(laying_root)
    if not coco_paths:
        raise RuntimeError(f"No classroom_laying COCO exports found under {laying_root}")
    records = load_coco_records(coco_paths)
    if not records:
        raise RuntimeError("No laying image records found")
    missing = [str(record["source_image"]) for record in records if not record["source_image"].exists()]
    if missing:
        raise RuntimeError(f"Missing image files: {missing[:5]}")

    split_records_by_name = split_records(records, args.seed, args.train_frac, args.val_frac)

    safe_clean_dir(detection_output, workspace, args.overwrite)
    add_laying_to_detection(base_detection, detection_output, split_records_by_name)

    safe_clean_dir(posture_output, workspace, args.overwrite)
    copy_tree(base_posture, posture_output)
    manifest = extract_crops(split_records_by_name, posture_output, source_posture_folder)
    write_manifest(posture_output / "laying_crops_manifest.csv", manifest)

    summary = {
        "coco_exports": len(coco_paths),
        "laying_images": len(records),
        "laying_boxes": sum(len(record["annotations"]) for record in records),
        "laying_split_images": {split: len(items) for split, items in split_records_by_name.items()},
        "laying_split_boxes": {
            split: sum(len(record["annotations"]) for record in items) for split, items in split_records_by_name.items()
        },
        "merged_detection_images": count_images(detection_output),
        "posture_laying_crops": len(manifest),
        "detection_output": str(detection_output),
        "posture_output": str(posture_output),
        "source_posture_folder": str(source_posture_folder),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
