"""Phase 3 — Illegal parking evaluation (ISLab-PVD, event-level).

Per the goal, illegal-parking eval is EVENT-LEVEL (predicted violation window vs GT window with
tolerance), NOT box mAP — kept deliberately separate from the box-level eval path.

On-disk reality (Phase 0b): ISLab-PVD shipped as 16 .mp4 videos with NO machine-readable GT, so
event-level precision/recall is NOT computable this run. This script:
  * reports that honestly (status = not_testable: GT absent);
  * provides the event-level scoring function (window-IoU style P/R) ready for when GT exists or
    the user supplies an alternative dataset;
  * can run the geometry dwell-rule over a video to PRODUCE candidate events for a qualitative
    demo (no metric claimed).

Run:  python -m src.eval.eval_illegal_parking
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

ISLAB_ROOT = REPO_ROOT / "datasets" / "Illegal Parking" / "IS_labPVD"


def event_pr(pred_windows, gt_windows, tol: float = 1.0):
    """Event-level P/R: a pred matches a GT if their [start,end] (seconds) overlap within tol.

    pred_windows/gt_windows: list of (start, end). Returns dict with precision/recall/f1.
    """
    def overlaps(a, b):
        return not (a[1] < b[0] - tol or a[0] > b[1] + tol)
    tp = sum(1 for p in pred_windows if any(overlaps(p, g) for g in gt_windows))
    fp = len(pred_windows) - tp
    matched_gt = sum(1 for g in gt_windows if any(overlaps(p, g) for p in pred_windows))
    fn = len(gt_windows) - matched_gt
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def run() -> dict:
    videos = sorted(ISLAB_ROOT.glob("*.mp4")) if ISLAB_ROOT.exists() else []
    gt_files = [p for p in ISLAB_ROOT.rglob("*") if p.suffix.lower() in
                {".json", ".csv", ".xml", ".txt", ".srt"}] if ISLAB_ROOT.exists() else []
    return {
        "dataset": "ISLab-PVD (illegal parking, event-level)",
        "videos": len(videos),
        "ground_truth_files": len(gt_files),
        "status": ("not_testable: GT absent on disk → event-level P/R not computable"
                   if not gt_files else "GT present — run event_pr()"),
        "eval_method": "event-level window P/R (event_pr), NOT box mAP",
        "tier": "blocked (data) → qualitative demo only until GT / alternative dataset supplied",
        "note": ("Geometry dwell-rule can still PRODUCE candidate parking events for a qualitative "
                 "demo (no metric). User is sourcing an alternative illegal-parking dataset; plug it "
                 "into event_pr() here when ready."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_illegal_parking_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Illegal-parking eval — ISLab-PVD (run {run_id})\n",
         f"- Videos: {result['videos']} · GT files: {result['ground_truth_files']}",
         f"- **Status: {result['status']}**",
         f"- Eval method: {result['eval_method']}",
         f"- Tier: {result['tier']}\n",
         f"> {result['note']}"]
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    argparse.ArgumentParser().parse_args()
    run_id = new_run_id()
    result = run()
    write_run_log("phase3", "illegal_parking", run_id, result)
    rp = write_report(result, run_id)
    log(f"[eval_illegal_parking] status={result['status']} (report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "illegal_parking",
                        "dataset": "ISLab-PVD", "model": "geometry-dwell", "metric": "event_PR",
                        "value": "not_testable", "target": "event P/R", "pass_fail": "blocked",
                        "note": "GT absent on disk"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
