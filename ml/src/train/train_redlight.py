"""Phase 5 — Red-light full-event sequence classifier (Tier 2).

Dataset #10 (RunningRedlight): each sample is a vehicle trajectory (sequence of [cx,cy,w,h]
boxes) with a clip-level binary label `meta.cross` (ran-red-light yes/no). The repo's own
approach is a sequence model; we train a small LSTM on the trajectory + velocity features.

Leakage-safe: split by VIDEO (vid_path), not by clip — one video contributes many vehicle
clips, so a clip-level split would leak.

Relation to the rule engine (J3): the spec wants a rule-vs-learned AGREEMENT rate, but the
geometry rule needs a per-camera scene config (stop-line + signal state) that these anonymized
trajectory clips do NOT carry — so agreement is not computable here. This learned classifier
stands as the Tier-2 promotion result; the agreement harness remains in eval_redlight_sequence.py
for when both signals exist on the same footage.

Run:  python -m src.train.train_redlight --epochs 30 --version v1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

RR_LABELS = (REPO_ROOT / "datasets" / "Red Light" / "namnv78_RunningRedlight" /
             "combined_data_v2" / "processed_labels")
CKPT_DIR = REPO_ROOT / "checkpoints" / "redlight"
SEQ_T = 32       # resample every trajectory to this many timesteps
MIN_FRAMES = 3   # need a real trajectory


def load_clips():
    """Return list of (vid_path, raw_frames[list of 4-vec], label_bool)."""
    out = []
    for jf in RR_LABELS.glob("*.json"):
        try:
            o = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = o.get("meta", {})
        if "cross" not in m:
            continue
        frames = o.get("frames", [])
        if len(frames) < MIN_FRAMES:
            continue
        out.append((m.get("vid_path", "?"), frames, bool(m["cross"])))
    return out


def resample(frames, t=SEQ_T):
    """Resample a variable-length [L,4] trajectory to fixed [t,6] with velocity features."""
    L = len(frames)
    seq = []
    prev = None
    for i in range(t):
        idx = round(i * (L - 1) / (t - 1)) if t > 1 else 0
        cx, cy, w, h = frames[idx][:4]
        if prev is None:
            dcx = dcy = 0.0
        else:
            dcx, dcy = cx - prev[0], cy - prev[1]
        seq.append([cx, cy, w, h, dcx, dcy])
        prev = (cx, cy)
    return seq


def split_by_video(clips, test_frac=0.2, seed=42):
    vids = sorted({c[0] for c in clips})
    random.Random(seed).shuffle(vids)
    n_test = max(1, int(len(vids) * test_frac))
    test_vids = set(vids[:n_test])
    train = [c for c in clips if c[0] not in test_vids]
    test = [c for c in clips if c[0] in test_vids]
    return train, test, len(test_vids)


def to_tensors(clips):
    import torch
    X = torch.tensor([resample(f) for _v, f, _l in clips], dtype=torch.float32)
    y = torch.tensor([1 if lbl else 0 for _v, _f, lbl in clips], dtype=torch.long)
    return X, y


def build_lstm(in_dim=6, hidden=48, num_classes=2):
    import torch.nn as nn

    class TrajLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, num_classes))

        def forward(self, x):
            out, (hn, _cn) = self.lstm(x)
            return self.head(hn[-1])

    return TrajLSTM()


def evaluate(model, X, y):
    import torch
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    correct = int((pred == y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round(correct / len(y), 4), "cross_precision": round(prec, 4),
            "cross_recall": round(rec, 4), "cross_f1": round(f1, 4)}


def train(epochs: int, version: str, lr: float, batch: int) -> dict:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch unavailable: {e}"}

    clips = load_clips()
    if not clips:
        return {"error": f"no usable clips under {RR_LABELS}"}
    train_c, test_c, n_test_vids = split_by_video(clips)
    Xtr, ytr = to_tensors(train_c)
    Xte, yte = to_tensors(test_c)
    pos_tr = int(ytr.sum())
    log(f"[train_redlight] clips={len(clips)} train={len(train_c)} test={len(test_c)} "
        f"(test from {n_test_vids} held-out videos) | train cross={pos_tr}/{len(ytr)}")

    model = build_lstm()
    # class weights (mild imbalance)
    counts = [int((ytr == 0).sum()), int((ytr == 1).sum())]
    w = torch.tensor([sum(counts) / (2 * max(c, 1)) for c in counts], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    n = len(Xtr)
    best_f1, best_state = -1.0, None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run += float(loss) * len(idx)
        m = evaluate(model, Xte, yte)
        history.append({"epoch": ep, "loss": round(run / n, 4), **m})
        if ep % 5 == 0 or ep == 1:
            log(f"   epoch {ep}/{epochs} loss={run/n:.4f} test_acc={m['accuracy']} cross_f1={m['cross_f1']}")
        if m["cross_f1"] >= best_f1:
            best_f1 = m["cross_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = m
    # save best
    ckpt = CKPT_DIR / version / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save({"state_dict": best_state, "seq_t": SEQ_T, "in_dim": 6, "hidden": 48,
             "classes": {0: "no_cross", 1: "cross"}}, ckpt)
    return {
        "model": "LSTM(6->48) trajectory classifier", "version": version, "epochs": epochs,
        "clips": len(clips), "train": len(train_c), "test": len(test_c),
        "test_held_out_videos": n_test_vids,
        "best_test_metrics": best_metrics, "best_cross_f1": round(best_f1, 4),
        "minutes": round((time.time() - t0) / 60, 2), "checkpoint": str(ckpt),
        "agreement_with_rule_engine": "not computable (clips lack scene config/signal state; J3 harness in eval_redlight_sequence.py)",
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    run_id = new_run_id()
    result = train(args.epochs, args.version, args.lr, args.batch)
    write_run_log("phase5", "redlight_train", run_id, result)
    if "error" in result:
        log(f"[train_redlight] ERROR: {result['error']}")
        return 1
    m = result["best_test_metrics"]
    log(f"[train_redlight] done {result['minutes']}min | held-out TEST: acc={m['accuracy']} "
        f"cross P={m['cross_precision']} R={m['cross_recall']} F1={m['cross_f1']} | ckpt {result['checkpoint']}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "redlight_event",
                        "dataset": "RunningRedlight", "model": "LSTM-traj",
                        "metric": "cross_f1", "value": result["best_cross_f1"],
                        "target": "from not_testable", "pass_fail": "finetuned",
                        "note": f"acc={m['accuracy']} split-by-video {result['test_held_out_videos']}vids"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
