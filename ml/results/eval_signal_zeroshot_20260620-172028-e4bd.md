# Signal-state zero-shot — §3.6 documented model (run 20260620-172028-e4bd)

- Classifier: SignalStateClassifier (HSV Tier-0, §3.6 documented zero-shot approach)
- Stage-1: RF-DETR-nano (COCO class=traffic light, Apache-2.0)
- Dataset: sample images of violations/Red light/ (all known red-light violations)
- Images: 7 · Red classified: 6 · **Hit-rate: 0.857**

| Image | Lights detected | States | Red hit |
|---|---|---|---|
| 039_frame.jpg | 0 | [] | ❌ |
| 071_frame.jpg | 6 | ['red', 'unknown', 'red', 'red', 'unknown', 'unknown'] | ✅ |
| dayClip1--02102.jpg | 13 | ['red', 'red', 'red', 'green', 'green', 'unknown', 'green', 'green', 'red', 'red', 'green', 'green', 'green'] | ✅ |
| dayClip7--00564.jpg | 4 | ['red', 'red', 'red', 'red'] | ✅ |
| dayClip7--00638.jpg | 4 | ['red', 'red', 'red', 'red'] | ✅ |
| dayClip8--00005.jpg | 6 | ['unknown', 'red', 'red', 'red', 'unknown', 'unknown'] | ✅ |
| dayClip9--00373.jpg | 3 | ['red', 'red', 'unknown'] | ✅ |

> Hit = traffic-light detected AND HSV classifier returns 'red'. Capability signal only — no per-image GT boxes. Full LISA benchmark is in eval_signal.py.