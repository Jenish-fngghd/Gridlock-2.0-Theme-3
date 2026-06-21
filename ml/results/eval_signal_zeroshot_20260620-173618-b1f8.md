# Signal-state zero-shot — §3.6 documented model (run 20260620-173618-b1f8)

- Classifier: SignalStateClassifier (HSV Tier-0, §3.6 documented zero-shot approach)
- Stage-1: RF-DETR-nano (COCO class=traffic light, Apache-2.0)
- Dataset: sample images of violations/Red light/ (all known red-light violations)
- Images: 7 · Red classified: 7 · **Hit-rate: 1.0**

| Image | Lights detected | States | Red hit |
|---|---|---|---|
| 039_frame.jpg | 5 | ['red', 'red', 'yellow', 'red', 'red'] | ✅ |
| 071_frame.jpg | 6 | ['unknown', 'red', 'red', 'red', 'red', 'red'] | ✅ |
| dayClip1--02102.jpg | 14 | ['red', 'red', 'red', 'green', 'green', 'unknown', 'green', 'green', 'red', 'unknown', 'green', 'green', 'red', 'red'] | ✅ |
| dayClip7--00564.jpg | 5 | ['red', 'red', 'red', 'red', 'unknown'] | ✅ |
| dayClip7--00638.jpg | 5 | ['red', 'red', 'red', 'red', 'unknown'] | ✅ |
| dayClip8--00005.jpg | 9 | ['red', 'red', 'unknown', 'red', 'unknown', 'red', 'unknown', 'unknown', 'unknown'] | ✅ |
| dayClip9--00373.jpg | 3 | ['red', 'red', 'unknown'] | ✅ |

> Hit = traffic-light detected AND HSV classifier returns 'red'. Capability signal only — no per-image GT boxes. Full LISA benchmark is in eval_signal.py.