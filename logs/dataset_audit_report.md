# Dataset Audit Report (Phase 0b / 0c)

_Generated: 2026-06-19T12:57:13_

> Hardware context: RTX 3050 Laptop, 4GB VRAM -> tier=cloud_required (see logs/environment_report.md)

## 0b — Per-dataset integrity

| # | Dataset | Status | Key counts | Notes |
|---|---|---|---|---|
| 1 | IDD | ✅ ready | images=46659; annotations_xml=41857 | VOC XML; JPEGImages/+Annotations/ pairing; ImageSets present |
| 2 | BDD100K | ❌ not_ready | — | ⚠️ Only .md5 checksum files present — no images/labels. SKIP this run, no placeholder. |
| 3 | CCPD | ✅ ready | total_images=11776; splits={'train': 5769, 'val': 1001, 'test': 5006} | GT encoded in filename (field 5 = LP char indices). Not a renamed mirror. |
| 4 | LISA | ✅ ready | frames=44075 | frame-level signal-state GT in frameAnnotationsBOX.csv (7 states) |
| 5 | IndianLP | ✅ ready | total_images=1741 | sirishan=1694 img, data-cluster-labs=47 img. Documented sirishan total ~16,192 img/21,683 plates — real on-disk count reported above (extraction may have skipped files). Sidecar labels: txt=1, json=0, xml=1740. |
| 6 | UA-DETRAC | 🟨 partial | images=140131; annotation_xml=0 | per-sequence MVI_* image folders + XML annos (optional detection backbone) |
| 7 | AICC | 🟨 partial | images=1; videos=0 | ⚠️ On disk = AI-City-Challenge-2023 winning-solution CODE repo + annotation CSVs ONLY. No Track-5 videos/frames present, so helmet+triple is NOT trainable/evaluable from this. Mark helmet module not_testable until raw AICC data is obtained. |
| 8 | Seatbelt | ✅ ready | splits={"train": 779, "valid": 337, "test": 0} | OBB polygons -> convert to axis-aligned via src/utils/obb_convert.py |
| 9 | ISLab-PVD | 🟨 partial | videos=16 | CCTV-style .mp4 videos present but NO machine-readable GT on disk -> event-level precision/recall NOT computable without manual annotation. Usable for qualitative demo only this run; build eval_illegal_parking.py event-level once GT exists. |
| 10 | RunningRedlight | ✅ ready | frame_images=15839; label_jsons=1331 | Arrived as frame-sequence folders (*.avi_save/) + per-clip JSON labels. Clip-level binary classification. Cross-check vs rule engine; do NOT auto-merge (J3). |
| 11 | WrongWay | ✅ ready | splits={"train": 426, "valid": 91, "test": 91} | OBB polygons -> convert to axis-aligned via src/utils/obb_convert.py |

### Selected details

```json
{
  "IDD": {
    "status": "ready",
    "images": 46659,
    "annotations_xml": 41857,
    "imagesets": {
      "train": true,
      "val": true,
      "test": true
    },
    "voc_xml_parses": true,
    "note": "VOC XML; JPEGImages/+Annotations/ pairing; ImageSets present",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\idd-detection\\IDD_Detection",
    "path_exists": true
  },
  "BDD100K": {
    "status": "not_ready",
    "ext_histogram": {
      ".md5": 3
    },
    "images_found": 0,
    "note": "⚠️ Only .md5 checksum files present — no images/labels. SKIP this run, no placeholder.",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\BDD100K",
    "path_exists": true
  },
  "CCPD": {
    "status": "ready",
    "edition": "CCPD2020 (green/new-energy plates)",
    "split_counts": {
      "train": 5769,
      "val": 1001,
      "test": 5006
    },
    "total_images": 11776,
    "sample_filename": "00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg",
    "filename_decode": {
      "fields": 7,
      "lp_index_count": 8,
      "lp_indices": [
        0,
        0,
        3,
        24,
        28,
        24,
        31,
        33
      ]
    },
    "note": "GT encoded in filename (field 5 = LP char indices). Not a renamed mirror.",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\CCPD",
    "path_exists": true
  },
  "LISA": {
    "status": "ready",
    "frames": 44075,
    "frameAnnotationsBOX_csv_count": 24,
    "csv_samples": [
      "Annotations\\Annotations\\daySequence1\\frameAnnotationsBOX.csv",
      "Annotations\\Annotations\\daySequence2\\frameAnnotationsBOX.csv",
      "Annotations\\Annotations\\dayTrain\\dayClip1\\frameAnnotationsBOX.csv",
      "Annotations\\Annotations\\dayTrain\\dayClip10\\frameAnnotationsBOX.csv"
    ],
    "note": "frame-level signal-state GT in frameAnnotationsBOX.csv (7 states)",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Red Light\\LISA Traffic Light Dataset",
    "path_exists": true
  },
  "IndianLP": {
    "status": "ready",
    "sirishan_images": 1694,
    "data_cluster_labs_images": 47,
    "total_images": 1741,
    "label_files": {
      "txt": 1,
      "json": 0,
      "xml": 1740
    },
    "note": "sirishan=1694 img, data-cluster-labs=47 img. Documented sirishan total ~16,192 img/21,683 plates — real on-disk count reported above (extraction may have skipped files). Sidecar labels: txt=1, json=0, xml=1740.",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Indian LP",
    "path_exists": true
  },
  "UA-DETRAC": {
    "status": "partial",
    "images": 140131,
    "annotation_xml": 0,
    "note": "per-sequence MVI_* image folders + XML annos (optional detection backbone)",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\DETRAC",
    "path_exists": true
  },
  "AICC": {
    "status": "partial",
    "images": 1,
    "videos": 0,
    "annotation_csvs": [
      "AI-City-Challenge-2023-main\\head_detection\\dataset\\trainset_head.csv",
      "AI-City-Challenge-2023-main\\helmet_detection_for_motorcyclists\\dataset\\trainset_ai.csv"
    ],
    "label_values": [
      1,
      2,
      3,
      4,
      5,
      7
    ],
    "edition_guess": "≤7-class style (2023 edition?)",
    "note": "⚠️ On disk = AI-City-Challenge-2023 winning-solution CODE repo + annotation CSVs ONLY. No Track-5 videos/frames present, so helmet+triple is NOT trainable/evaluable from this. Mark helmet module not_testable until raw AICC data is obtained.",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Helmet & Triple Riding",
    "path_exists": true
  },
  "Seatbelt": {
    "status": "ready",
    "names": {
      "0": "mobile",
      "1": "seatbelt",
      "2": "windshield"
    },
    "splits": {
      "train": {
        "images": 779,
        "labels": 779
      },
      "valid": {
        "images": 337,
        "labels": 337
      },
      "test": {
        "images": 0,
        "labels": 0
      }
    },
    "label_token_counts": [
      9
    ],
    "format": "YOLOv8-OBB (class + 4 corner xy)",
    "note": "OBB polygons -> convert to axis-aligned via src/utils/obb_convert.py",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\seat belt detection\\seat_belt and mobile.v2i.yolov8-obb",
    "path_exists": true
  },
  "ISLab-PVD": {
    "status": "partial",
    "videos": 16,
    "video_samples": [
      "ISLab-01.mp4",
      "ISLab-02.mp4",
      "ISLab-03.mp4",
      "ISLab-04.mp4",
      "ISLab-05.mp4"
    ],
    "ground_truth_files": 0,
    "gt_samples": [],
    "note": "CCTV-style .mp4 videos present but NO machine-readable GT on disk -> event-level precision/recall NOT computable without manual annotation. Usable for qualitative demo only this run; build eval_illegal_parking.py event-level once GT exists.",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Illegal Parking\\IS_labPVD",
    "path_exists": true
  },
  "RunningRedlight": {
    "status": "ready",
    "label_jsons": 1331,
    "frame_images": 15839,
    "clip_label_field": "meta.cross (bool) = ran-red-light",
    "cross_sample_counts(first200)": {
      "true": 130,
      "false": 70
    },
    "note": "Arrived as frame-sequence folders (*.avi_save/) + per-clip JSON labels. Clip-level binary classification. Cross-check vs rule engine; do NOT auto-merge (J3).",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Red Light\\namnv78_RunningRedlight",
    "path_exists": true
  },
  "WrongWay": {
    "status": "ready",
    "names": {
      "0": "right-side",
      "1": "wrong-side"
    },
    "splits": {
      "train": {
        "images": 426,
        "labels": 426
      },
      "valid": {
        "images": 91,
        "labels": 91
      },
      "test": {
        "images": 91,
        "labels": 91
      }
    },
    "label_token_counts": [
      9
    ],
    "format": "YOLOv8-OBB (class + 4 corner xy)",
    "note": "OBB polygons -> convert to axis-aligned via src/utils/obb_convert.py",
    "path": "C:\\Users\\sorat\\Downloads\\Gridlock 2.0\\Gridlock 2.0_R2\\datasets\\Wrong Side Driving\\Wrong Way Driving Detection.v1i.yolov8-obb",
    "path_exists": true
  }
}
```

## 0c — Violation coverage map

| Violation / capability | Dataset | Eval status |
|---|---|---|
| Detection backbone | IDD / UA-DETRAC (BDD100K NOT READY) | quantitative |
| ANPR | CCPD2020 / Indian-LP | quantitative |
| Helmet | AICC Track 5 | blocked: imagery absent (not_testable) |
| Triple riding | AICC Track 5 (same GT) | blocked: imagery absent (not_testable) |
| Seatbelt | Seatbelt+Mobile (OBB) | quantitative |
| Illegal parking | ISLab-PVD | blocked: videos present, GT absent -> qualitative only |
| Red-light (signal state) | LISA | quantitative (frame-level) |
| Red-light (full event) | RunningRedlight | quantitative (clip-level cross-check) |
| Wrong-side driving | Wrong-Way (OBB) | quantitative (frame-level) |
| Stop-line | — none — | qualitative spot-check only (no dataset anywhere) |

**Quantitative & ready now:** Detection (IDD/UA-DETRAC), ANPR (CCPD/Indian-LP), Seatbelt, Red-light signal-state (LISA), Red-light full-event (RunningRedlight), Wrong-side (Wrong-Way).

**Blocked / not quantitative this run:**
- **Helmet & Triple riding** — AICC imagery absent (only code repo + annotation CSVs). `not_testable` until the real Track-5 data is downloaded.
- **Illegal parking** — ISLab-PVD has videos but no GT on disk → qualitative only until annotated.
- **BDD100K** — only `.md5` checksums → NOT READY, skipped, no placeholder.
- **Stop-line** — no dataset exists anywhere → qualitative spot-check only (never a metric).
