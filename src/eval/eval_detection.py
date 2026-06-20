"""Phase 3 — Detection evaluation (IDD, VOC) with COCO-style mAP.

Evaluates the zero-shot RF-DETR detector against IDD's VOC ground truth using pycocotools
(mAP@0.5, mAP@0.5:0.95) per 00_master_design.md §7.

Honesty about the domain gap (§6/§7): RF-DETR zero-shot only knows COCO classes, so we score
the classes that have a COCO equivalent (car, motorcycle, bus, truck, bicycle, person — IDD
`rider` folded into person, since a rider is a person on a 2-wheeler). IDD's India-specific
classes (auto-rickshaw, vehicle-fallback, animal, traffic-sign) have NO COCO equivalent and
are reported separately as a structural gap (expected ~0 recall zero-shot) — never silently
dropped, never counted as a free win.

Run:
  python -m src.eval.eval_detection --limit 200 --variant nano --threshold 0.3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from src.modules.detection import VehicleDetector
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

IDD_ROOT = REPO_ROOT / "datasets" / "idd-detection" / "IDD_Detection"

# eval class -> stable category id
EVAL_CLASSES = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4,
                "bus": 5, "truck": 6, "traffic light": 7}
# IDD GT name -> eval class
IDD_TO_EVAL = {
    "car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
    "bicycle": "bicycle", "person": "person", "rider": "person",
    "traffic light": "traffic light",
}
# IDD classes with NO COCO equivalent -> structural domain gap (reported, not scored)
GAP_CLASSES = {"autorickshaw", "vehicle fallback", "animal", "traffic sign",
               "caravan", "trailer", "train"}


def load_val_list(root: Path, limit: int | None) -> list[str]:
    vf = root / "val.txt"
    entries = vf.read_text(encoding="utf-8").split() if vf.exists() else []
    out = []
    for e in entries:
        img = None
        for ext in (".jpg", ".png"):
            p = root / "JPEGImages" / f"{e}{ext}"
            if p.exists():
                img = p
                break
        xml = root / "Annotations" / f"{e}.xml"
        if img and xml.exists():
            out.append((str(img), str(xml)))
        if limit and len(out) >= limit:
            break
    return out


def parse_voc(xml_path: str) -> tuple[int, int, list[tuple[str, list[float]]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        b = o.find("bndbox")
        if b is None:
            continue
        box = [float(b.findtext("xmin", "0")), float(b.findtext("ymin", "0")),
               float(b.findtext("xmax", "0")), float(b.findtext("ymax", "0"))]
        objs.append((name, box))
    return w, h, objs


def run(limit: int, variant: str, threshold: float) -> dict:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:  # noqa: BLE001
        return {"error": f"pycocotools unavailable: {e}"}

    samples = load_val_list(IDD_ROOT, limit)
    if not samples:
        return {"error": f"no IDD val samples found under {IDD_ROOT}"}
    log(f"[eval_detection] {len(samples)} IDD val images | RF-DETR-{variant} @ thr={threshold}")

    det = VehicleDetector(variant=variant, threshold=threshold)
    if det.model is None:
        return {"error": f"detector unavailable: {det._unavailable_reason}", "model_unavailable": True}

    images, gts, dts = [], [], []
    gap_gt_counts: dict[str, int] = {}
    ann_id = 1
    for img_id, (img_path, xml_path) in enumerate(samples, 1):
        w, h, objs = parse_voc(xml_path)
        images.append({"id": img_id, "width": w, "height": h, "file_name": os.path.basename(img_path)})
        for name, (x1, y1, x2, y2) in objs:
            if name in GAP_CLASSES:
                gap_gt_counts[name] = gap_gt_counts.get(name, 0) + 1
            ev = IDD_TO_EVAL.get(name)
            if ev is None:
                continue
            gts.append({"id": ann_id, "image_id": img_id, "category_id": EVAL_CLASSES[ev],
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0})
            ann_id += 1
        res = det.detect(img_path)
        for d in res.detections:
            if d.class_name not in EVAL_CLASSES:
                continue
            x1, y1, x2, y2 = d.xyxy
            dts.append({"image_id": img_id, "category_id": EVAL_CLASSES[d.class_name],
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d.confidence})
        if img_id % 50 == 0:
            log(f"   ...{img_id}/{len(samples)}")

    if not gts:
        return {"error": "no mappable GT objects in sampled images"}

    categories = [{"id": cid, "name": n} for n, cid in EVAL_CLASSES.items()]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gts, "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dts) if dts else None

    metrics = {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "per_class_AP@0.5": {}}
    if coco_dt is not None:
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        metrics["mAP@0.5:0.95"] = round(float(ev.stats[0]), 4)
        metrics["mAP@0.5"] = round(float(ev.stats[1]), 4)
        # per-class AP@0.5  (precision shape [T,R,K,A,M]; T idx 0 -> IoU 0.5)
        prec = ev.eval["precision"]
        for k, (name, cid) in enumerate(EVAL_CLASSES.items()):
            p = prec[0, :, k, 0, -1]
            p = p[p > -1]
            metrics["per_class_AP@0.5"][name] = round(float(p.mean()), 4) if p.size else 0.0

    return {
        "dataset": "IDD-Detection (val)", "images": len(samples), "variant": variant,
        "threshold": threshold, "gt_objects_scored": len(gts), "detections": len(dts),
        "metrics": metrics,
        "domain_gap_classes_in_GT": gap_gt_counts,
        "domain_gap_note": ("These IDD classes have NO COCO equivalent, so zero-shot RF-DETR "
                            "cannot detect them (~0 recall). Closing this needs the data-engine + "
                            "fine-tune (§3.10/§8). NOT included in the mAP above."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_detection_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Detection eval — IDD (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    m = result["metrics"]
    L.append(f"- Model: RF-DETR-{result['variant']} (zero-shot, COCO weights) @ thr={result['threshold']}")
    L.append(f"- Images: {result['images']} · GT scored: {result['gt_objects_scored']} · "
             f"Detections: {result['detections']}\n")
    L.append("## Quantitative (mAP, COCO-mappable classes)\n")
    L.append("| Metric | Value | §7 reference |")
    L.append("|---|---|---|")
    L.append(f"| mAP@0.5 | **{m['mAP@0.5']}** | DriveIndia ~0.787 (fine-tuned target) |")
    L.append(f"| mAP@0.5:0.95 | **{m['mAP@0.5:0.95']}** | — |")
    L.append("")
    L.append("| Class | AP@0.5 |")
    L.append("|---|---|")
    for n, ap in m["per_class_AP@0.5"].items():
        L.append(f"| {n} | {ap} |")
    L.append("\n## Structural domain gap (NOT scored — no COCO class)\n")
    if result["domain_gap_classes_in_GT"]:
        L.append("| IDD class | GT instances (undetectable zero-shot) |")
        L.append("|---|---|")
        for n, c in sorted(result["domain_gap_classes_in_GT"].items(), key=lambda x: -x[1]):
            L.append(f"| {n} | {c} |")
    L.append(f"\n> {result['domain_gap_note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    run_id = new_run_id()
    result = run(args.limit, args.variant, args.threshold)
    write_run_log("phase3", "detection", run_id, result)
    rp = write_report(result, run_id)

    if "error" in result:
        log(f"[eval_detection] ERROR: {result['error']}")
        append_run_history({"run_id": run_id, "phase": "phase3", "module": "detection",
                            "dataset": "IDD", "model": f"rfdetr-{args.variant}",
                            "metric": "mAP@0.5", "value": "ERROR", "target": 0.787,
                            "pass_fail": "error", "note": result["error"]})
        return 1
    m = result["metrics"]
    log(f"[eval_detection] mAP@0.5={m['mAP@0.5']} mAP@0.5:0.95={m['mAP@0.5:0.95']}  (report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "detection",
                        "dataset": "IDD", "model": f"rfdetr-{args.variant}-zeroshot",
                        "metric": "mAP@0.5", "value": m["mAP@0.5"], "target": 0.787,
                        "pass_fail": "baseline", "note": f"{result['images']} imgs; gap classes excluded"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
