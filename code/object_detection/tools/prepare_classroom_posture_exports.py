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
        description="Convert labeled classroom COCO human boxes into YOLO detection data and posture classification crops."
    )
    parser.add_argument("--base-detection", default="training_yolo26_boxes")
    parser.add_argument("--base-posture-split", default="classification_splits/posture_binary")
    parser.add_argument("--labeled-root", default="data/labeled")
    parser.add_argument("--detection-output", default="training_yolo26_boxes_with_classroom_postures")
    parser.add_argument("--classification-output", default="PostureClasssification_classroom_roboflow.folder")
    parser.add_argument("--posture-split-output", default="classification_splits/posture_with_classroom_postures")
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


def infer_posture_class(path: Path) -> str | None:
    text = path.as_posix().lower()
    if "laying" in text or "lying" in text:
        return "person_laying"
    if "sitting" in text:
        return "person_sitting"
    if "standing" in text:
        return "person_standing"
    return None


def original_stem(image: dict) -> str:
    extra = image.get("extra") or {}
    name = extra.get("name") or image.get("file_name", "")
    return Path(name).stem


def find_coco_exports(root: Path) -> list[Path]:
    paths = sorted(root.glob("*classroom*/*/_annotations.coco.json"))
    return [path for path in paths if infer_posture_class(path)]


def load_records(coco_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen_files: set[Path] = set()
    for coco_path in coco_paths:
        posture_class = infer_posture_class(coco_path)
        if not posture_class:
            continue
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {cat["id"]: str(cat.get("name", "")).lower() for cat in data.get("categories", [])}
        images = {img["id"]: img for img in data.get("images", [])}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            if categories.get(ann.get("category_id")) == "human":
                anns_by_image[ann["image_id"]].append(ann)

        for image_id, image in images.items():
            source_image = (coco_path.parent / image["file_name"]).resolve()
            if source_image in seen_files:
                continue
            seen_files.add(source_image)
            records.append(
                {
                    "source_image": source_image,
                    "file_name": image["file_name"],
                    "original_stem": original_stem(image),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": anns_by_image.get(image_id, []),
                    "posture_class": posture_class,
                    "source_group": coco_path.parents[1].name,
                }
            )
    return records


def split_records(records: list[dict], seed: int, train_frac: float, val_frac: float) -> dict[str, list[dict]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for record in records:
        group = f"{record['posture_class']}::{record['source_group']}::{record['original_stem']}"
        if group not in by_class[record["posture_class"]]:
            by_class[record["posture_class"]].append(group)

    split_by_group: dict[str, str] = {}
    rng = random.Random(seed)
    for groups in by_class.values():
        groups = groups[:]
        rng.shuffle(groups)
        n_train = round(len(groups) * train_frac)
        n_val = round(len(groups) * val_frac)
        for group in groups[:n_train]:
            split_by_group[group] = "train"
        for group in groups[n_train : n_train + n_val]:
            split_by_group[group] = "val"
        for group in groups[n_train + n_val :]:
            split_by_group[group] = "test"

    out = {split: [] for split in SPLITS}
    for record in records:
        group = f"{record['posture_class']}::{record['source_group']}::{record['original_stem']}"
        out[split_by_group[group]].append(record)
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


def add_records_to_detection(base: Path, output: Path, records_by_split: dict[str, list[dict]]) -> None:
    copy_tree(base, output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for record in records_by_split[split]:
            prefix = record["posture_class"].replace("person_", "classroom_")
            out_stem = f"{prefix}_{record['source_group'].split('-')[-1].strip().replace(' ', '_')}_{record['original_stem']}"
            ext = record["source_image"].suffix.lower()
            shutil.copy2(record["source_image"], output / "images" / split / f"{out_stem}{ext}")
            lines = yolo_lines(record)
            label_text = "\n".join(lines) + ("\n" if lines else "")
            (output / "labels" / split / f"{out_stem}.txt").write_text(label_text, encoding="utf-8")

    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "- Human\n",
        encoding="utf-8",
    )


def crop_record(record: dict, crop_counts: dict[str, int]) -> list[tuple[str, Image.Image, dict]]:
    crops = []
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
            key = f"{record['posture_class']}::{record['source_group']}::{record['original_stem']}"
            crop_counts[key] += 1
            crop_name = (
                f"{record['posture_class'].replace('person_', 'classroom_')}_"
                f"{record['source_group'].split('-')[-1].strip().replace(' ', '_')}_"
                f"{record['original_stem']}_crop_{crop_counts[key]:02d}.jpg"
            )
            meta = {
                "class_name": record["posture_class"],
                "crop_file": crop_name,
                "source_image": str(record["source_image"]),
                "source_group": record["source_group"],
                "original_stem": record["original_stem"],
                "bbox_x": f"{x:.2f}",
                "bbox_y": f"{y:.2f}",
                "bbox_w": f"{w:.2f}",
                "bbox_h": f"{h:.2f}",
            }
            crops.append((crop_name, img.crop((left, top, right, bottom)), meta))
    return crops


def write_classification_outputs(
    records_by_split: dict[str, list[dict]],
    classification_output: Path,
    posture_split_output: Path,
    base_posture_split: Path,
) -> list[dict]:
    copy_tree(base_posture_split, posture_split_output)
    manifest = []
    crop_counts: dict[str, int] = defaultdict(int)
    upload_counts: dict[str, int] = defaultdict(int)

    for split in SPLITS:
        for record in records_by_split[split]:
            for crop_name, crop, meta in crop_record(record, crop_counts):
                class_name = record["posture_class"]
                split_class_dir = posture_split_output / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                crop.save(split_class_dir / crop_name, quality=95)

                upload_counts[class_name] += 1
                upload_name = f"{class_name}_{upload_counts[class_name]:06d}.jpg"
                upload_class_dir = classification_output / class_name
                upload_class_dir.mkdir(parents=True, exist_ok=True)
                crop.save(upload_class_dir / upload_name, quality=95)

                meta = {"split": split, **meta, "roboflow_file": upload_name}
                manifest.append(meta)
    return manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def image_counts(root: Path, split_layout: bool) -> dict:
    if split_layout:
        return {
            split: {
                cls.name: len([p for p in cls.glob("*") if p.suffix.lower() in IMAGE_EXTS])
                for cls in sorted((root / split).iterdir())
                if cls.is_dir()
            }
            for split in SPLITS
            if (root / split).exists()
        }
    return {
        cls.name: len([p for p in cls.glob("*") if p.suffix.lower() in IMAGE_EXTS])
        for cls in sorted(root.iterdir())
        if cls.is_dir()
    }


def main() -> None:
    args = parse_args()
    workspace = Path.cwd().resolve()
    base_detection = Path(args.base_detection)
    base_posture_split = Path(args.base_posture_split)
    labeled_root = Path(args.labeled_root)
    detection_output = Path(args.detection_output)
    classification_output = Path(args.classification_output)
    posture_split_output = Path(args.posture_split_output)

    coco_paths = find_coco_exports(labeled_root)
    if not coco_paths:
        raise RuntimeError(f"No classroom posture COCO exports found under {labeled_root}")
    records = load_records(coco_paths)
    missing = [str(record["source_image"]) for record in records if not record["source_image"].exists()]
    if missing:
        raise RuntimeError(f"Missing image files: {missing[:5]}")

    records_by_split = split_records(records, args.seed, args.train_frac, args.val_frac)

    safe_clean_dir(detection_output, workspace, args.overwrite)
    add_records_to_detection(base_detection, detection_output, records_by_split)

    safe_clean_dir(classification_output, workspace, args.overwrite)
    safe_clean_dir(posture_split_output, workspace, args.overwrite)
    manifest = write_classification_outputs(
        records_by_split=records_by_split,
        classification_output=classification_output,
        posture_split_output=posture_split_output,
        base_posture_split=base_posture_split,
    )
    write_manifest(posture_split_output / "classroom_posture_crops_manifest.csv", manifest)

    detection_counts = {
        split: len([p for p in (detection_output / "images" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS])
        for split in SPLITS
    }
    summary = {
        "coco_exports": len(coco_paths),
        "source_images": len(records),
        "source_boxes": sum(len(record["annotations"]) for record in records),
        "source_images_by_class": dict(
            sorted(
                (cls, sum(1 for record in records if record["posture_class"] == cls))
                for cls in {record["posture_class"] for record in records}
            )
        ),
        "source_boxes_by_class": dict(
            sorted(
                (
                    cls,
                    sum(len(record["annotations"]) for record in records if record["posture_class"] == cls),
                )
                for cls in {record["posture_class"] for record in records}
            )
        ),
        "detection_images": detection_counts,
        "classification_upload_counts": image_counts(classification_output, split_layout=False),
        "posture_split_counts": image_counts(posture_split_output, split_layout=True),
        "detection_output": str(detection_output),
        "classification_output": str(classification_output),
        "posture_split_output": str(posture_split_output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
