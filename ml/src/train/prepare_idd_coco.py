"""Prepare IDD (VOC XML) -> RF-DETR's Roboflow-COCO layout for fine-tuning.

RF-DETR's trainer (dataset_file="roboflow") expects:
    <out>/train/_annotations.coco.json + images
    <out>/valid/_annotations.coco.json + images
    <out>/test/_annotations.coco.json  + images   (optional)

IDD ships as VOC: JPEGImages/<sub>/<seq>/<id>.jpg + Annotations/.../<id>.xml + {train,val,test}.txt.
This script reads the split lists, parses every VOC XML, builds one COCO json per split, and
links each image into the split folder under a FLATTENED name (sub__seq__id.jpg) so basename
collisions across sequences can't clash. Symlinks by default (instant, no disk blow-up); use
--copy on filesystems without symlink support.

ALL IDD classes are kept (incl. autorickshaw / vehicle-fallback) — learning the India-specific
classes that COCO zero-shot cannot see is the whole point of the fine-tune.

Pure stdlib (runs anywhere, no torch needed). Run on the Lightning machine after upload:
    python -m src.train.prepare_idd_coco --idd-root datasets/idd-detection/IDD_Detection --out datasets/idd_coco
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

SPLsuper = {"train": "train", "val": "valid", "test": "test"}  # IDD name -> roboflow name


def resolve(idd_root: Path, entry: str):
    img = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = idd_root / "JPEGImages" / f"{entry}{ext}"
        if p.exists():
            img = p
            break
    xml = idd_root / "Annotations" / f"{entry}.xml"
    return img, (xml if xml.exists() else None)


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        b = o.find("bndbox")
        if b is None or not name:
            continue
        x1 = float(b.findtext("xmin", "0")); y1 = float(b.findtext("ymin", "0"))
        x2 = float(b.findtext("xmax", "0")); y2 = float(b.findtext("ymax", "0"))
        if x2 <= x1 or y2 <= y1:
            continue
        objs.append((name, x1, y1, x2, y2))
    return w, h, objs


def read_split(idd_root: Path, split: str):
    f = idd_root / f"{split}.txt"
    if not f.exists():
        return []
    entries = f.read_text(encoding="utf-8").split()
    pairs = []
    for e in entries:
        img, xml = resolve(idd_root, e)
        if img and xml:
            pairs.append((e, img, xml))
    return pairs


def build_categories(idd_root: Path, splits) -> dict:
    """Scan all splits' XMLs once to fix a stable class->id map (1-indexed)."""
    names = set()
    for sp in splits:
        for _e, _img, xml in read_split(idd_root, sp):
            try:
                _w, _h, objs = parse_voc(xml)
            except Exception:
                continue
            names.update(n for n, *_ in objs)
    return {name: i for i, name in enumerate(sorted(names), start=1)}


def link_image(src: Path, dst: Path, copy: bool):
    if dst.exists():
        return
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def convert_split(idd_root: Path, split: str, out_root: Path, cat_map: dict, copy: bool, limit):
    pairs = read_split(idd_root, split)
    if limit:
        pairs = pairs[:limit]
    rb = SPLsuper[split]
    if not pairs:
        # IDD's test set is unlabeled (no XMLs) -> skip; RF-DETR only needs train/valid.
        print(f"[{rb}] 0 labelled images -> skipped (no annotations on disk)", flush=True)
        return 0, 0
    split_dir = out_root / rb
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for img_id, (entry, img, xml) in enumerate(pairs, start=1):
        try:
            w, h, objs = parse_voc(xml)
        except Exception:
            continue
        flat = entry.replace("/", "__").replace("\\", "__") + img.suffix
        link_image(img, split_dir / flat, copy)
        images.append({"id": img_id, "file_name": flat, "width": w, "height": h})
        for name, x1, y1, x2, y2 in objs:
            cid = cat_map.get(name)
            if cid is None:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
        if img_id % 2000 == 0:
            print(f"   [{rb}] {img_id}/{len(pairs)}", flush=True)
    categories = [{"id": cid, "name": name, "supercategory": "none"}
                  for name, cid in sorted(cat_map.items(), key=lambda kv: kv[1])]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"[{rb}] {len(images)} images, {len(annotations)} boxes -> {split_dir}", flush=True)
    return len(images), len(annotations)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd-root", required=True, help="path to IDD_Detection")
    ap.add_argument("--out", required=True, help="output COCO dataset dir")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--limit", type=int, default=None, help="cap images per split (debug)")
    args = ap.parse_args()

    idd_root = Path(args.idd_root)
    out_root = Path(args.out)
    splits = [s for s in ("train", "val", "test") if (idd_root / f"{s}.txt").exists()]
    if not splits:
        print(f"ERROR: no split .txt files under {idd_root}")
        return 1
    print(f"[prepare] splits found: {splits}")
    cat_map = build_categories(idd_root, [s for s in splits if s != "test"])
    print(f"[prepare] {len(cat_map)} classes: {cat_map}")
    total = {}
    for sp in splits:
        total[sp] = convert_split(idd_root, sp, out_root, cat_map, args.copy, args.limit)
    # persist the class map for reference / eval
    (out_root / "classes.json").write_text(json.dumps(cat_map, indent=2), encoding="utf-8")
    print(f"[prepare] DONE. num_classes={len(cat_map)}. dataset_dir={out_root}")
    print("[prepare] next: python -m src.train.train_detection --dataset-dir", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
