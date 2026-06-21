# Signal-state zero-shot — §3.6 documented model (run 20260620-173724-f13e)

- Classifier: SignalStateClassifier (HSV Tier-0, §3.6 documented zero-shot approach)
- Stage-1: RF-DETR-nano (COCO class=traffic light, Apache-2.0)
- Dataset: sample images of violations/Red light/ (all known red-light violations)
- Images: 7 · Red classified: 7 · **Hit-rate: 1.0**

| Image | Lights detected | States | Red hit |
|---|---|---|---|
| 039_frame.jpg | 3 | ['red', 'yellow', 'red'] | ✅ |
| 071_frame.jpg | 10 | ['red', 'unknown', 'red', 'red', 'unknown', 'unknown', 'red', 'unknown', 'red', 'red'] | ✅ |
| dayClip1--02102.jpg | 16 | ['red', 'red', 'red', 'green', 'green', 'unknown', 'green', 'green', 'red', 'red', 'green', 'green', 'green', 'unknown', 'green', 'red'] | ✅ |
| dayClip7--00564.jpg | 6 | ['red', 'red', 'red', 'red', 'unknown', 'red'] | ✅ |
| dayClip7--00638.jpg | 7 | ['red', 'red', 'red', 'red', 'unknown', 'red', 'red'] | ✅ |
| dayClip8--00005.jpg | 10 | ['unknown', 'red', 'red', 'red', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'] | ✅ |
| dayClip9--00373.jpg | 4 | ['red', 'red', 'unknown', 'unknown'] | ✅ |

> Hit = traffic-light detected AND HSV classifier returns 'red'. Capability signal only — no per-image GT boxes. Full LISA benchmark is in eval_signal.py.