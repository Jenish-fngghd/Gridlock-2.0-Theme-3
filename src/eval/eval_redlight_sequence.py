"""Phase 3 — Red-light full-event evaluation (RunningRedlight, clip-level).

Dataset #10: each sample is a vehicle trajectory clip with a clip-level binary label
`meta.cross` (ran-red-light yes/no). The design's interpretable path is the geometry rule
engine (signal=red AND stop-line crossing); the dataset's own approach is a learned sequence
classifier. Per the goal we must NOT auto-merge them — instead run both and report their
AGREEMENT RATE (J3 validation: does the auditable rule match the black-box model?).

Status this run: the learned sequence classifier is untrained (Tier 2) and the geometry rule
needs a per-clip scene config (stop-line + signal state) which these anonymized trajectory
clips don't carry → the zero-shot red-light *event* verdict is `not_testable` here. This script
loads the GT label distribution and lays out the agreement-rate harness for when either side is
runnable (Phase 5+).

Run:  python -m src.eval.eval_redlight_sequence --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

RR_ROOT = REPO_ROOT / "datasets" / "Red Light" / "namnv78_RunningRedlight"
LABELS_DIR = RR_ROOT / "combined_data_v2" / "processed_labels"


def load_labels(limit: int) -> list[dict]:
    out = []
    if not LABELS_DIR.exists():
        return out
    for jf in sorted(LABELS_DIR.glob("*.json")):
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = obj.get("meta", {})
        if "cross" in meta:
            out.append({"file": jf.name, "cross": bool(meta["cross"]),
                        "n_frames": len(obj.get("frames", []))})
        if limit and len(out) >= limit:
            break
    return out


def run(limit: int) -> dict:
    labels = load_labels(limit)
    if not labels:
        return {"error": f"no RunningRedlight clip labels under {LABELS_DIR}"}
    dist = Counter(l["cross"] for l in labels)
    return {
        "dataset": "RunningRedlight (clip-level meta.cross)",
        "clips": len(labels),
        "gt_distribution": {"violation(cross=true)": dist[True], "no_violation": dist[False]},
        "geometry_rule_verdict": "not_testable (clips lack per-clip stop-line/signal scene config)",
        "learned_classifier": "untrained (Tier 2) — train LSTM/Transformer on these sequences (Phase 5)",
        "agreement_rate": None,
        "tier": "tier_2 (independent cross-check vs rule engine; do NOT auto-merge — J3)",
        "note": ("Agreement-rate harness ready: once the rule engine has a scene config OR the "
                 "learned classifier is trained, run both per clip and report fraction-agree as a "
                 "validation of the interpretable rule (J3 'geometry over black-box')."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_redlight_sequence_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Red-light full-event eval — RunningRedlight (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
    else:
        L.append(f"- Clips: {result['clips']} · GT: {result['gt_distribution']}")
        L.append(f"- Geometry rule verdict: {result['geometry_rule_verdict']}")
        L.append(f"- Learned classifier: {result['learned_classifier']}")
        L.append(f"- Agreement rate: {result['agreement_rate']}")
        L.append(f"- Tier: {result['tier']}\n")
        L.append(f"> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.limit)
    write_run_log("phase3", "redlight_sequence", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_redlight_sequence] ERROR: {result['error']}")
        return 1
    log(f"[eval_redlight_sequence] clips={result['clips']} verdict=not_testable(Tier2) "
        f"(report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "redlight_event",
                        "dataset": "RunningRedlight", "model": "rule+learned", "metric": "agreement",
                        "value": "n/a", "target": "agreement-rate", "pass_fail": "tier2",
                        "note": f"{result['clips']} clips; harness ready"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
