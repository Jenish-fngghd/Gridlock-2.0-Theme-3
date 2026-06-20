# Helmet — Zero-shot pretrained-model test (Phase 6 escalation)

_Qualitative capability test — IDD has NO helmet ground truth, so these are NOT benchmark metrics._

## Setup
- **Stage 1:** RF-DETR-nano finds motorcycles + persons (our working detector).
- **Stage 2:** `leeyunjai/yolo11-helmet` (YOLO11, classes `helmet` / `face`=bare-head) on the rider crop.
- 50 IDD frames containing motorcycles · 70 motorcycles · 29 head detections.

## Result
| | count |
|---|---|
| helmet heads | 1 |
| no-helmet (`face`) heads | 28 |
| head detections / motorcycles | 29 / 70 (**~41% rider-head recall**) |

## Honest read (from inspecting annotated frames)
**The two-stage pipeline works** — it correctly fires on clear riders. Verified by eye:
- ✅ `000492_r.jpg` — scooter rider from behind, bare head → correctly flagged **no-helmet**. A real, correct violation detection.
- ⚠️ `000180_r.jpg` — the "helmet 0.66" hit is on a tiny, distant blob — unreliable at that scale.

**Two real limitations, both expected:**
1. **Low recall on IDD's viewpoint.** IDD is dashcam footage → most riders are far away and small; the helmet model only fired on ~41% of detected motorcycles' riders. Enforcement cameras (closer, angled) would do better.
2. **Helmet/no-helmet balance is unreliable.** The 1/28 split is partly real (these rural roads genuinely have low helmet use) but also partly a **cross-domain artifact** — a construction/PPE helmet model under-recognizes full-face *motorcycle* helmets (visually unlike hard hats), biasing toward `face`/no-helmet.

## Verdict
- **Capability: demonstrated.** A pretrained helmet model + our detector produces real helmet/no-helmet calls on Indian motorcyclists with zero AICC data — helmet is no longer a total blank.
- **Not a metric, and not deployment-ready.** Without GT we can't score it; cross-domain bias + low recall mean it needs a proper fine-tune to trust.
- **Path forward (unchanged, now evidenced):** fine-tune on motorcycle-helmet data (AICC 7-class when imagery is acquired, or a Roboflow motorcycle-helmet set) — Tier 1. This test confirms the *architecture* is right; only the *weights* need domain adaptation.

## Note
The motorcycle-specific Roboflow models (purpose-built, with/without-helmet + plate) need a free Roboflow API key to pull — a likely better zero-shot model than this PPE one if a quick upgrade is wanted before fine-tuning.
