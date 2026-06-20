# Red-light full-event eval — RunningRedlight (run 20260619-150102-dc8c)

- Clips: 500 · GT: {'violation(cross=true)': 101, 'no_violation': 399}
- Geometry rule verdict: not_testable (clips lack per-clip stop-line/signal scene config)
- Learned classifier: untrained (Tier 2) — train LSTM/Transformer on these sequences (Phase 5)
- Agreement rate: None
- Tier: tier_2 (independent cross-check vs rule engine; do NOT auto-merge — J3)

> Agreement-rate harness ready: once the rule engine has a scene config OR the learned classifier is trained, run both per clip and report fraction-agree as a validation of the interpretable rule (J3 'geometry over black-box').