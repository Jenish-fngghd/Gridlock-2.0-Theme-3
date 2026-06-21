# Wrong-side eval (LEARNED) — Wrong-Way OBB (run 20260619-191336-e45a)

- Model: mobilenet_v3_small (fine-tuned)  ·  Wrong-Way OBB (test, held-out)
- Instances: 720 {'right-side': 629, 'wrong-side': 91}

## Baseline → Learned (the Phase-5 escalation)

| Approach | Wrong-side verdict |
|---|---|
| Geometry rule (zero-shot) | not_testable (abstains on stills) |
| **Learned classifier (fine-tuned)** | **acc 0.9889 · wrong-side P 0.977 / R 0.9341 / F1 0.9551** |

> LEARNED classifier replaces geometry for the wrong-side verdict (Tier 2 promoted)