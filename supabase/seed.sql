-- Gridlock 2.0 — seed data
-- An "Uploads" pseudo-camera for ad-hoc photo/clip submissions, plus the deployed
-- model registry (mirrors ml/results/run_history.csv — honest, no silent substitution).

insert into cameras (id, name, location_name, latitude, longitude, is_active)
values ('00000000-0000-0000-0000-000000000001', 'Uploads', 'Ad-hoc submissions', null, null, true)
on conflict (id) do nothing;

insert into model_registry (module, model_name, variant, version, metric_name, metric_value, checkpoint_ref) values
  ('detection',        'RF-DETR',      'large',          'zeroshot', 'mAP@0.5',    0.5216, 'rfdetr-large (COCO, runtime)'),
  ('anpr',             'TrOCR',        'base-printed-ft','v1',       'exact_acc',  0.7811, 'ml/checkpoints/anpr/trocr_ft'),
  ('seatbelt',         'YOLOv11n+MobileNetV3-L', 'e2e',  'v4',       'f1',         0.8082, 'ml/checkpoints/windshield/v1 + ml/checkpoints/seatbelt/v4'),
  ('wrong_side',       'MobileNetV3',  'small-ft',       'v1',       'f1',         0.9551, 'ml/checkpoints/wrongside/v1/model.pt'),
  ('red_light_event',  'LSTM-traj',    'hidden48',       'v1',       'f1',         0.9000, 'ml/checkpoints/redlight/v1/model.pt'),
  ('signal_state',     'HSV-tier0',    'rule',           'v1',       'accuracy',   0.9967, 'rule-based (no checkpoint)'),
  ('helmet_triple',    'RF-DETR/SAM-3','nano@1280/openvocab', 'v1', 'hit_rate',   1.0000, 'runtime + SAM-3');
