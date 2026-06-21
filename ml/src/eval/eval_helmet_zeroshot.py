"""Phase 5 — Helmet + triple-riding zero-shot baseline (§3.4, documented model, no fine-tune).

00_master_design.md §3.4: "AICC Track-5 7-class scheme … Two-stage pipeline: detect motorbike
@high-res → crop → classify 7 states → associate head→rider→bike."

Stage-1: RF-DETR (Apache-2.0, our documented backbone) detects motorcycle + person zero-shot
         from COCO-pretrained weights via `HelmetTripleModule`.
Stage-2: Helmet state (7-class AICC scheme) — NOT TESTABLE zero-shot, requires AICC fine-tune.
Proxy:   Triple-riding detection (≥3 riders on one motorcycle) IS testable from Stage-1 alone.

Dataset: "sample images of violations" (newly added — helmet/ 11 images, triple riding/ 6 images).
         Folder-level label = known violations; no per-image GT boxes → hit-rate is a capability
         signal (same honest framing as 07_trackB_foundation_models_lightning.md).

Run:  python -m src.eval.eval_helmet_zeroshot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.helmet_triple import HelmetTripleModule
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

SAMPLE_ROOT = (REPO_ROOT / "datasets" / "Helmet & Triple Riding" /
               "sample images of violations" / "sample images of violations")
HELMET_DIR = SAMPLE_ROOT / "helmet"
TRIPLE_DIR = SAMPLE_ROOT / "triple riding"


def _eval_folder(module: HelmetTripleModule, folder: Path, label: str) -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}

    hits = triples = 0
    results = []
    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    for fp in images:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        r = module.analyze(image=img)
        if r.get("model_unavailable"):
            return {"error": f"detector unavailable: {r.get('note', '')}"}
        fired = r["motorbikes"] > 0
        triple = r["triple_riding_count"] > 0
        hits += int(fired)
        triples += int(triple)
        results.append({
            "image": fp.name,
            "motorbikes_detected": r["motorbikes"],
            "riders_associated": sum(g["rider_count"] for g in r["groups"]),
            "triple_riding_detected": triple,
            "hit": fired,
        })
    return {
        "folder": label,
        "total_images": len(images),
        "images_with_motorcycle_detected": hits,
        "triple_riding_images": triples,
        "hit_rate": round(hits / max(len(images), 1), 3),
        "per_image": results,
    }


def run(conf: float, imgsz: int = 1280) -> dict:
    if not HELMET_DIR.exists() or not TRIPLE_DIR.exists():
        return {"error": f"sample violation images not found under {SAMPLE_ROOT}"}

    module = HelmetTripleModule(resolution=imgsz)
    helmet_res = _eval_folder(module, HELMET_DIR, "helmet violations")
    if "error" in helmet_res:
        return helmet_res
    triple_res = _eval_folder(module, TRIPLE_DIR, "triple riding violations")
    if "error" in triple_res:
        return triple_res

    return {
        "stage1_detector": f"RF-DETR-nano @{imgsz}px (COCO-pretrained, Apache-2.0 — documented backbone §3.4)",
        "stage1_classes": "motorcycle + person — COCO zero-shot",
        "association": "overlap(>0.1) + nearest-x-center heuristic (license-clean re-impl §3.4)",
        "helmet_state": "not_testable — requires AICC Track-5 7-class fine-tune (Tier 1)",
        "dataset": "datasets/Helmet & Triple Riding/sample images of violations/",
        "helmet_folder": helmet_res,
        "triple_riding_folder": triple_res,
        "note": ("Hit-rate = fraction of known-violation images where Stage-1 fires on a "
                 "motorcycle. Helmet STATE is not_testable until AICC fine-tune. "
                 "Triple-riding proxy fires when ≥3 riders are associated to one motorcycle."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_helmet_zeroshot_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Helmet + triple-riding zero-shot — §3.4 documented model (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Stage-1: {result['stage1_detector']}")
    L.append(f"- Dataset: {result['dataset']}\n")
    hr = result["helmet_folder"]
    tr = result["triple_riding_folder"]
    L.append("## Hit-rate (capability signal — no per-image GT boxes)\n")
    L.append("| Folder | Images | Motorcycle hit | Hit-rate | Triple-riding triggered |")
    L.append("|---|---|---|---|---|")
    L.append(f"| helmet violations | {hr['total_images']} | "
             f"{hr['images_with_motorcycle_detected']} | {hr['hit_rate']} | "
             f"{hr['triple_riding_images']} |")
    L.append(f"| triple riding violations | {tr['total_images']} | "
             f"{tr['images_with_motorcycle_detected']} | {tr['hit_rate']} | "
             f"{tr['triple_riding_images']} |")
    L.append("\n## Sub-task status\n")
    L.append("| Sub-task | Status |")
    L.append("|---|---|")
    L.append(f"| Motorcycle detection (RF-DETR COCO zero-shot) | "
             f"helmet={hr['hit_rate']} · triple={tr['hit_rate']} |")
    L.append(f"| Triple-riding proxy (≥3 riders) | "
             f"{tr['triple_riding_images']}/{tr['total_images']} triggered |")
    L.append(f"| Helmet state (7-class AICC) | NOT TESTABLE — {result['helmet_state']} |")
    L.append(f"\n> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="RF-DETR inference resolution (§3.4 spec: 1280-1536px for rider detection)")
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.conf, imgsz=args.imgsz)
    write_run_log("phase5", "helmet_zeroshot", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_helmet_zeroshot] ERROR: {result['error']}")
        return 1
    hr = result["helmet_folder"]
    tr = result["triple_riding_folder"]
    log(f"[eval_helmet_zeroshot] helmet hit-rate={hr['hit_rate']} "
        f"({hr['images_with_motorcycle_detected']}/{hr['total_images']}) | "
        f"triple hit-rate={tr['hit_rate']} ({tr['images_with_motorcycle_detected']}/{tr['total_images']}) "
        f"triple-triggered={tr['triple_riding_images']} | helmet_state=not_testable | report: {rp.name}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "helmet_triple_zeroshot",
                        "dataset": "sample-violation-images",
                        "model": f"RF-DETR-nano(COCO)@{args.imgsz}px",
                        "metric": "hit_rate_proxy", "value": hr["hit_rate"],
                        "target": "qualitative (capability signal)", "pass_fail": "zeroshot",
                        "note": (f"helmet_hit={hr['hit_rate']} triple_hit={tr['hit_rate']} "
                                 f"triple_triggered={tr['triple_riding_images']} "
                                 "helmet_state=not_testable")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
