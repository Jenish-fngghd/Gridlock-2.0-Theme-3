# Seatbelt zero-shot — §3.5 Stage-2 classifier on sample images (run 20260620-170617-7bfe)

- Classifier: mobilenet_v3_small (§3.5 CNN belt classifier, checkpoint: model.pt)
- Stage-1: NOT APPLIED — whole image fed as crop (Stage-1 checkpoint pending)
- Dataset: sample images of violations/seatbelt/ (all known no-seatbelt violations)
- Images: 10 · no_seatbelt predicted: 0 · **Hit-rate: 0.0**

| Image | Prediction | Hit |
|---|---|---|
| 1_TRUCK_09-31-05-585_jpg.rf.5a6c636404b8d127839fa9e3540ece7a.jpg | seatbelt | ❌ |
| 1_TRUCK_10-17-41-100_jpg.rf.fb2afa927d64eac55f888513c745ae81.jpg | seatbelt | ❌ |
| 1_TRUCK_10-20-51-678_jpg.rf.026194d5d05f436efba2ad02778648e5.jpg | seatbelt | ❌ |
| 1_TRUCK_10-35-56-561_jpg.rf.92618c53a736f1298d9d621a82cf3a34.jpg | seatbelt | ❌ |
| 1_TRUCK_10-35-59-220_jpg.rf.d8b072d7e8038d61218ece5aed95806f.jpg | seatbelt | ❌ |
| 1_TRUCK_10-51-26-151_jpg.rf.83d1f9f6bbae357845847ea72438968d.jpg | seatbelt | ❌ |
| 1_TRUCK_10-53-53-430_jpg.rf.3e659eb96354b86d9654628611d48233.jpg | seatbelt | ❌ |
| 1_TRUCK_10-58-28-267_jpg.rf.694978991e1b5681ff3cc031c57413b3.jpg | seatbelt | ❌ |
| 1_TRUCK_11-00-03-166_jpg.rf.b851af9ca372a649d4e2291aee6b30aa.jpg | seatbelt | ❌ |
| _jpg.rf.672128429f36edc77107fa2467638606.jpg | seatbelt | ❌ |

> Whole-image fed to classifier without a windshield crop — harder than real pipeline. Once the windshield detector checkpoint is available, run eval_seatbelt_e2e.py for the true two-stage number.