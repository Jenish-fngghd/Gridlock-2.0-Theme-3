"""Phase 6 (escalation test) — Zero-shot helmet detection via a pretrained model.

Helmet is our only mandated violation with NO local result (AICC imagery absent; no public
checkpoint for the AICC per-rider scheme). This script tests whether an OFF-THE-SHELF helmet
model can fill that gap directly, in the design's two-stage form:

  Stage 1: RF-DETR (our working detector) finds motorcycles + persons in the frame.
  Stage 2: a pretrained helmet model (leeyunjai/yolo11-helmet — classes {helmet, face=no-helmet})
           runs on the rider region above each motorcycle -> helmet vs no-helmet head count.

HONESTY: IDD has NO helmet ground truth, so this is a QUALITATIVE capability demo (annotated
crops + counts), NOT a benchmark metric. The helmet model is construction/PPE-domain ("helmet
on a head" vs "bare face"), used here as a cross-domain transfer test on motorcyclists. If it
transfers well, helmet is partially unblocked without AICC data; if not, it confirms the need
for motorcycle-specific fine-tuning.

Run:  python -m src.eval.eval_helmet_zeroshot --limit 60 --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.detection import VehicleDetector
from src.utils.logging import REPO_ROOT, log, new_run_id, write_run_log

HELMET_CKPT = REPO_ROOT / "checkpoints" / "helmet_pretrained" / "helmet-11s.pt"
IDD_FRAMES = REPO_ROOT / "datasets" / "idd-detection" / "IDD_Detection" / "JPEGImages"
OUT = REPO_ROOT / "results" / "helmet_zeroshot"


def rider_region(moto_xyxy, persons, img_w, img_h, pad=0.25):
    """Box covering the motorbike + any associated rider, expanded upward for heads."""
    x1, y1, x2, y2 = moto_xyxy
    # include persons overlapping/above the motorbike (riders)
    for p in persons:
        px1, py1, px2, py2 = p
        pcx = (px1 + px2) / 2
        if (x1 - 0.3 * (x2 - x1)) <= pcx <= (x2 + 0.3 * (x2 - x1)) and py2 <= y2 + 0.2 * (y2 - y1):
            x1, y1, x2, y2 = min(x1, px1), min(y1, py1), max(x2, px2), max(y2, py2)
    bw, bh = x2 - x1, y2 - y1
    # expand, especially upward (heads sit above the bike)
    rx1 = max(0, x1 - pad * bw)
    ry1 = max(0, y1 - 0.5 * bh)   # extra headroom
    rx2 = min(img_w, x2 + pad * bw)
    ry2 = min(img_h, y2 + 0.1 * bh)
    return [int(rx1), int(ry1), int(rx2), int(ry2)]


def run(limit: int, conf: float, save_annotated: int) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        return {"error": f"cv2/ultralytics unavailable: {e}"}
    if not HELMET_CKPT.exists():
        return {"error": f"helmet checkpoint missing: {HELMET_CKPT}"}

    helmet = YOLO(str(HELMET_CKPT))
    names = helmet.names  # {0:'helmet',1:'face'}
    det = VehicleDetector(variant="nano", threshold=0.3,
                          keep_classes={"motorcycle", "person"})
    OUT.mkdir(parents=True, exist_ok=True)

    # gather IDD frames, find ones with motorcycles
    frames = sorted(IDD_FRAMES.rglob("*.jpg"))
    if not frames:
        return {"error": f"no IDD frames under {IDD_FRAMES}"}

    examined = motos_found = riders_examined = helmet_heads = nohelmet_heads = 0
    saved = 0
    per_image = []
    for fp in frames:
        if examined >= limit:
            break
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        d = det.detect(img)
        if d.model_unavailable:
            continue
        motos = [x.xyxy for x in d.detections if x.class_name == "motorcycle"]
        persons = [x.xyxy for x in d.detections if x.class_name == "person"]
        if not motos:
            continue
        examined += 1
        motos_found += len(motos)
        img_helmet = img_nohelmet = 0
        vis = img.copy()
        for m in motos:
            rr = rider_region(m, persons, w, h)
            crop = img[rr[1]:rr[3], rr[0]:rr[2]]
            if crop.size == 0:
                continue
            res = helmet.predict(crop, conf=conf, verbose=False)
            for b in res[0].boxes:
                cls = int(b.cls[0])
                cf = float(b.conf[0])
                riders_examined += 1
                bx = b.xyxy[0].tolist()
                # map crop coords back to full image
                fx1, fy1 = rr[0] + bx[0], rr[1] + bx[1]
                fx2, fy2 = rr[0] + bx[2], rr[1] + bx[3]
                if names[cls] == "helmet":
                    helmet_heads += 1; img_helmet += 1; color = (0, 200, 0)
                else:
                    nohelmet_heads += 1; img_nohelmet += 1; color = (0, 0, 255)
                cv2.rectangle(vis, (int(fx1), int(fy1)), (int(fx2), int(fy2)), color, 2)
                cv2.putText(vis, f"{names[cls]} {cf:.2f}", (int(fx1), int(fy1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(vis, (rr[0], rr[1]), (rr[2], rr[3]), (255, 180, 0), 1)
        if (img_helmet or img_nohelmet) and saved < save_annotated:
            cv2.imwrite(str(OUT / f"helmet_{fp.stem}.jpg"), vis)
            saved += 1
        if img_helmet or img_nohelmet:
            per_image.append({"image": fp.name, "motos": len(motos),
                              "helmet": img_helmet, "no_helmet": img_nohelmet})

    return {
        "type": "QUALITATIVE capability test (no helmet GT in IDD — NOT a benchmark metric)",
        "helmet_model": "leeyunjai/yolo11-helmet (helmet-11s) — PPE-domain, cross-domain transfer",
        "stage1_detector": "RF-DETR-nano (motorcycle+person)",
        "frames_with_motorcycles_examined": examined,
        "motorcycles_found": motos_found,
        "head_detections": riders_examined,
        "helmet_heads": helmet_heads,
        "no_helmet_heads": nohelmet_heads,
        "helmet_rate": round(helmet_heads / max(helmet_heads + nohelmet_heads, 1), 3),
        "annotated_saved": saved, "output_dir": str(OUT),
        "examples": per_image[:15],
        "note": ("Counts show the model DOES fire on motorcyclist heads cross-domain. Without GT we "
                 "cannot score accuracy — inspect annotated frames in output_dir to judge quality. "
                 "For a real metric, fine-tune on AICC/motorcycle-helmet data (Tier 1)."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="frames-with-motorcycles to examine")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", type=int, default=25, help="max annotated frames to save")
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.limit, args.conf, args.save)
    write_run_log("phase6", "helmet_zeroshot", run_id, result)
    if "error" in result:
        log(f"[helmet_zeroshot] ERROR: {result['error']}")
        return 1
    log(f"[helmet_zeroshot] examined {result['frames_with_motorcycles_examined']} moto-frames | "
        f"heads: {result['helmet_heads']} helmet / {result['no_helmet_heads']} no-helmet "
        f"(rate {result['helmet_rate']}) | annotated -> {result['output_dir']}")
    log("[helmet_zeroshot] QUALITATIVE (no GT) — inspect annotated frames to judge transfer quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
