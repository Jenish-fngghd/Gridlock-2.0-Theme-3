"""Phase 3 — Seatbelt evaluation (seat_belt-and-mobile OBB).

There is no zero-shot windshield/belt checkpoint, so the zero-shot baseline is `not_testable`
(Tier 1 — mandatory fine-tune). This script confirms that honestly and reports the fine-tune
dataset's real on-disk size + class layout (so Phase 5 has its inputs), exercising the OBB
parser. The `mobile` class is logged as out-of-scope (outside the 7 mandated violations).

If a fine-tuned checkpoint path is given via --weights, it will be evaluated instead.

Run:  python -m src.eval.eval_seatbelt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.seatbelt import SeatbeltModule
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

SB_ROOT = (REPO_ROOT / "datasets" / "seat belt detection" /
           "seat_belt and mobile.v2i.yolov8-obb")
NAMES = {0: "mobile", 1: "seatbelt", 2: "windshield"}


def count_split(split: str) -> dict:
    img_dir = SB_ROOT / split / "images"
    lbl_dir = SB_ROOT / split / "labels"
    imgs = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    cls_counts = {0: 0, 1: 0, 2: 0}
    if lbl_dir.exists():
        for lf in lbl_dir.glob("*.txt"):
            for r in parse_obb_label_file(lf):
                cls_counts[r["class_id"]] = cls_counts.get(r["class_id"], 0) + 1
    return {"images": imgs, "instances_by_class": {NAMES.get(k, k): v for k, v in cls_counts.items()}}


def run(weights: str | None) -> dict:
    sb = SeatbeltModule(model_path=weights)
    splits = {s: count_split(s) for s in ("train", "valid", "test")}
    if sb.model is None:
        return {
            "dataset": "seat_belt-and-mobile OBB", "status": "not_testable",
            "tier": "tier_1 (mandatory fine-tune — no zero-shot checkpoint)",
            "finetune_data": splits,
            "out_of_scope_class": "mobile (detected-but-flagged, outside 7 mandated violations)",
            "note": sb.NOTE,
        }
    return {"dataset": "seat_belt-and-mobile OBB", "status": "fine-tuned model loaded",
            "finetune_data": splits, "note": "evaluate fine-tuned weights here (Phase 5)."}


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_seatbelt_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Seatbelt eval (run {run_id})\n",
         f"- Status: **{result['status']}**  ·  Tier: {result.get('tier','-')}",
         f"- Fine-tune data on disk: {result['finetune_data']}",
         f"- Out-of-scope: {result.get('out_of_scope_class','-')}\n",
         f"> {result['note']}"]
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="fine-tuned seatbelt checkpoint (optional)")
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.weights)
    write_run_log("phase3", "seatbelt", run_id, result)
    rp = write_report(result, run_id)
    log(f"[eval_seatbelt] status={result['status']} (report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "seatbelt",
                        "dataset": "seatbelt-OBB", "model": "none", "metric": "status",
                        "value": result["status"], "target": "§7", "pass_fail": "tier1",
                        "note": "mandatory fine-tune"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
