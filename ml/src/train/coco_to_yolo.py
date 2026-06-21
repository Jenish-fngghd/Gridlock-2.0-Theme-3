"""Convert the IDD Roboflow-COCO dataset (already built) -> YOLO format for Ultralytics.

Reuses datasets/idd_coco_sub/{train,valid}/ (images + _annotations.coco.json). Writes YOLO
labels and links the images into the structure Ultralytics expects:

    <out>/images/train -> (junction to) idd_coco_sub/train
    <out>/images/val   -> (junction to) idd_coco_sub/valid
    <out>/labels/train/*.txt   (class cx cy w h, normalized; class = coco_cat_id - 1)
    <out>/labels/val/*.txt
    <out>/data.yaml

No image re-copy (uses a Windows directory junction; falls back to symlink/copy).

Run:
    python -m src.train.coco_to_yolo --coco datasets/idd_coco_sub --out datasets/idd_yolo
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

SPLITMAP = {"train": "train", "valid": "val"}  # coco dir name -> yolo split name


def link_dir(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
        return
    except OSError:
        pass
    # Windows junction (no admin needed)
    try:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src.resolve())],
                       check=True, capture_output=True)
        return
    except Exception:
        # last resort: copy
        import shutil
        shutil.copytree(src, dst)


def convert_split(coco_dir: Path, coco_split: str, out: Path) -> tuple[int, int, list]:
    jdir = coco_dir / coco_split
    jf = jdir / "_annotations.coco.json"
    if not jf.exists():
        return 0, 0, []
    coco = json.loads(jf.read_text(encoding="utf-8"))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    catid_to_idx = {c["id"]: i for i, c in enumerate(cats)}  # 1-indexed coco id -> 0-indexed yolo
    images = {im["id"]: im for im in coco["images"]}
    by_img: dict[int, list[str]] = {}
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if not im:
            continue
        w, h = im["width"], im["height"]
        if w <= 0 or h <= 0:
            continue
        x, y, bw, bh = a["bbox"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        nw, nh = bw / w, bh / h
        if nw <= 0 or nh <= 0:
            continue
        cls = catid_to_idx.get(a["category_id"])
        if cls is None:
            continue
        by_img.setdefault(a["image_id"], []).append(
            f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    # Write YOLO labels NEXT TO the images (in the same dir). Ultralytics' label resolver
    # looks for "<image_stem>.txt" beside the image when the path has no /images/ segment —
    # this is robust and avoids junctions (which broke label discovery).
    n_lbl = 0
    for img_id, lines in by_img.items():
        stem = Path(images[img_id]["file_name"]).stem
        (jdir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_lbl += 1
    for im in coco["images"]:           # empty label = background (valid for YOLO)
        lp = jdir / f"{Path(im['file_name']).stem}.txt"
        if not lp.exists():
            lp.write_text("", encoding="utf-8")
    return len(coco["images"]), n_lbl, names, jdir.resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Roboflow-COCO dataset dir")
    ap.add_argument("--out", required=True, help="output YOLO dataset dir")
    args = ap.parse_args()
    coco = Path(args.coco)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = None
    split_dirs = {}
    for csplit in ("train", "valid"):
        n_img, n_lbl, nm, jdir = convert_split(coco, csplit, out)
        if nm:
            names = nm
            split_dirs[csplit] = jdir
        print(f"[coco_to_yolo] {csplit}: {n_img} images, {n_lbl} with boxes  ({jdir})")
    if names is None:
        print("ERROR: no splits converted"); return 1

    data_yaml = out / "data.yaml"
    # point train/val directly at the image dirs (labels live beside the images)
    lines = [f"train: {split_dirs['train'].as_posix()}",
             f"val: {split_dirs.get('valid', split_dirs['train']).as_posix()}",
             f"nc: {len(names)}", "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[coco_to_yolo] {len(names)} classes: {names}")
    print(f"[coco_to_yolo] data.yaml -> {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
