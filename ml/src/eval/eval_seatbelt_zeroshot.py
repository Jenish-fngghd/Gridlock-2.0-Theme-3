"""Phase 5 — Seatbelt zero-shot on sample violation images (§3.5, Stage-2 classifier only).

00_master_design.md §3.5: "YOLOv11/windshield detector → driver crop → CNN/CNN-SVM belt
classifier." The windshield detector (Stage-1) is being trained separately
(`train_windshield_detector.py` / `lightning_windshield_detector.py`). Until that
checkpoint is available, this script tests the Stage-2 belt classifier
(`checkpoints/seatbelt/v2/model.pt`, F1=0.678 on GT crops) on the sample seatbelt
violation images — treating the whole image as the crop (worst-case, no Stage-1 box).

Dataset: sample images of violations/seatbelt/ (10 images, all known seatbelt violations
         → label = no_seatbelt / violation).
Metric:  hit-rate: fraction where classifier predicts no_seatbelt (class 1).
         This is a capability signal, NOT accuracy (no per-pixel GT).

Run:  python -m src.eval.eval_seatbelt_zeroshot
      python -m src.eval.eval_seatbelt_e2e --det checkpoints/windshield/v1/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

SAMPLE_ROOT = (REPO_ROOT / "datasets" / "Helmet & Triple Riding" /
               "sample images of violations" / "sample images of violations")
SEATBELT_DIR = SAMPLE_ROOT / "seatbelt"
CLF_CKPT_DEFAULT = REPO_ROOT / "checkpoints" / "seatbelt" / "v2" / "model.pt"


def run(clf_ckpt: Path | None = None) -> dict:
    if clf_ckpt is None:
        clf_ckpt = CLF_CKPT_DEFAULT
    CLF_CKPT = clf_ckpt
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch/PIL unavailable: {e}"}
    if not CLF_CKPT.exists():
        return {"error": f"belt classifier checkpoint not found: {CLF_CKPT}"}
    if not SEATBELT_DIR.exists():
        return {"error": f"seatbelt sample folder not found: {SEATBELT_DIR}"}

    blob = torch.load(CLF_CKPT, map_location="cpu")
    backbone_key = blob.get("backbone", "mobilenet_v3_small")
    if "large" in backbone_key:
        clf = models.mobilenet_v3_large(weights=None)
    else:
        clf = models.mobilenet_v3_small(weights=None)
    clf.classifier[-1] = nn.Linear(clf.classifier[-1].in_features, 2)
    clf.load_state_dict(blob["state_dict"])
    clf.eval()
    input_size = blob.get("input", 224)
    class_names = blob.get("classes", {0: "seatbelt", 1: "no_seatbelt"})
    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in sorted(SEATBELT_DIR.iterdir()) if p.suffix.lower() in exts]
    hits = 0
    results = []
    for fp in images:
        img = Image.open(fp).convert("RGB")
        with torch.no_grad():
            pred = int(clf(tfm(img).unsqueeze(0)).argmax(1).item())
        label = class_names.get(pred, str(pred))
        hit = pred == 1  # class 1 = no_seatbelt (violation)
        hits += int(hit)
        results.append({"image": fp.name, "prediction": label, "hit": hit})

    return {
        "classifier": f"{backbone_key} (§3.5 CNN belt classifier, checkpoint: {CLF_CKPT.name})",
        "stage1_detector": "NOT APPLIED — whole image fed as crop (Stage-1 checkpoint pending)",
        "dataset": "sample images of violations/seatbelt/ (all known no-seatbelt violations)",
        "total_images": len(images),
        "no_seatbelt_predicted": hits,
        "hit_rate": round(hits / max(len(images), 1), 3),
        "per_image": results,
        "note": ("Whole-image fed to classifier without a windshield crop — harder than real "
                 "pipeline. Once the windshield detector checkpoint is available, run "
                 "eval_seatbelt_e2e.py for the true two-stage number."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_seatbelt_zeroshot_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Seatbelt zero-shot — §3.5 Stage-2 classifier on sample images (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Classifier: {result['classifier']}")
    L.append(f"- Stage-1: {result['stage1_detector']}")
    L.append(f"- Dataset: {result['dataset']}")
    L.append(f"- Images: {result['total_images']} · no_seatbelt predicted: "
             f"{result['no_seatbelt_predicted']} · **Hit-rate: {result['hit_rate']}**\n")
    L.append("| Image | Prediction | Hit |")
    L.append("|---|---|---|")
    for r in result["per_image"]:
        L.append(f"| {r['image']} | {r['prediction']} | {'✅' if r['hit'] else '❌'} |")
    L.append(f"\n> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default=None,
                    help="belt classifier checkpoint (default: checkpoints/seatbelt/v2/model.pt)")
    args = ap.parse_args()
    run_id = new_run_id()
    clf_path = Path(args.clf) if args.clf else None
    result = run(clf_ckpt=clf_path)
    write_run_log("phase5", "seatbelt_zeroshot", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_seatbelt_zeroshot] ERROR: {result['error']}")
        return 1
    log(f"[eval_seatbelt_zeroshot] hit-rate={result['hit_rate']} "
        f"({result['no_seatbelt_predicted']}/{result['total_images']}) | report: {rp.name}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "seatbelt_zeroshot",
                        "dataset": "sample-violation-images(seatbelt)",
                        "model": "MobileNetV3-small(v2,no-stage1)",
                        "metric": "hit_rate_proxy", "value": result["hit_rate"],
                        "target": "qualitative (capability signal)", "pass_fail": "zeroshot",
                        "note": f"no_stage1_crop hits={result['no_seatbelt_predicted']}/{result['total_images']}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
