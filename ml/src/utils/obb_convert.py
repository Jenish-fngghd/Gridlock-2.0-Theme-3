"""OBB polygon -> axis-aligned bounding box conversion (datasets #8 Seatbelt, #11 Wrong-Way).

Roboflow YOLOv8-OBB labels are 9 tokens per line:
    class_id  x1 y1 x2 y2 x3 y3 x4 y4        (all xy normalized 0..1)

The rest of the pipeline and the §7 mAP eval use axis-aligned boxes, so we collapse each
4-corner polygon to its min/max AABB. We ALSO compute the polygon's rotation angle and keep
it separately: it is a cheap proxy for vehicle heading direction and could supplement the
monocular-3D yaw the geometry engine uses for wrong-side reasoning (Phase 6 fallback).

Pure-stdlib (no numpy/cv2 needed). CLI converts a whole labels/ dir to axis-aligned YOLO
labels and reports angle stats.

Run:  python -m src.utils.obb_convert --labels "datasets/.../train/labels" --out out/aabb_labels
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def obb_to_aabb(coords: list[float]) -> tuple[float, float, float, float]:
    """8 normalized corner coords -> axis-aligned (xc, yc, w, h), normalized (YOLO style)."""
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin)


def obb_to_xyxy(coords: list[float]) -> tuple[float, float, float, float]:
    """8 corner coords -> (xmin, ymin, xmax, ymax)."""
    xs = coords[0::2]
    ys = coords[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def obb_angle_deg(coords: list[float]) -> float:
    """Rotation angle (degrees, -90..90) of the polygon's longest edge.

    Heading proxy: angle of the longer of the first two edges relative to the x-axis.
    """
    (x1, y1, x2, y2, x3, y3, x4, y4) = coords
    e1 = (x2 - x1, y2 - y1)
    e2 = (x3 - x2, y3 - y2)
    # pick the longer edge as the orientation reference
    edge = e1 if (e1[0] ** 2 + e1[1] ** 2) >= (e2[0] ** 2 + e2[1] ** 2) else e2
    ang = math.degrees(math.atan2(edge[1], edge[0]))
    # normalize to (-90, 90]
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180
    return ang


def parse_obb_line(line: str) -> dict | None:
    toks = line.split()
    if len(toks) != 9:
        return None
    cls = int(float(toks[0]))
    coords = [float(t) for t in toks[1:]]
    return {
        "class_id": cls,
        "polygon": coords,
        "aabb_xywh": obb_to_aabb(coords),
        "aabb_xyxy": obb_to_xyxy(coords),
        "angle_deg": obb_angle_deg(coords),
    }


def parse_obb_label_file(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = parse_obb_line(line)
        if rec:
            out.append(rec)
    return out


def convert_dir(labels_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = n_boxes = 0
    angles: list[float] = []
    for lf in sorted(labels_dir.glob("*.txt")):
        recs = parse_obb_label_file(lf)
        lines = []
        for r in recs:
            xc, yc, w, h = r["aabb_xywh"]
            lines.append(f"{r['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            angles.append(r["angle_deg"])
        (out_dir / lf.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_files += 1
        n_boxes += len(recs)
    stats = {
        "label_files": n_files,
        "boxes": n_boxes,
        "angle_min": round(min(angles), 2) if angles else None,
        "angle_max": round(max(angles), 2) if angles else None,
        "angle_mean": round(sum(angles) / len(angles), 2) if angles else None,
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="OBB polygon labels -> axis-aligned YOLO labels")
    ap.add_argument("--labels", required=True, help="input YOLOv8-OBB labels/ dir")
    ap.add_argument("--out", required=True, help="output axis-aligned labels/ dir")
    args = ap.parse_args()
    stats = convert_dir(Path(args.labels), Path(args.out))
    print(f"[obb_convert] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
