# 04 — Dataset Acquisition & Preparation

**Purpose:** where to get every benchmark dataset (official + mirror), what format it
arrives in, the *exact* folder layout `src/pipeline.py` needs, and the precise commands
to get from "downloaded archive" to "loader sees it."

> **Key principle — format does not matter, final structure does.**
> The loaders in `DatasetManager` (`src/pipeline.py`) are decoupled from archive type.
> Whether a dataset ships as `.zip`, `.tar`, `.tar.gz`, or nested archives, you just
> need to extract it so the **final folder layout** matches the "Loader needs" box below.
> Anything not ready is skipped with a warning — it never blocks the rest of the run.

---

## 0. Universal extraction cheatsheet

| Archive | Command (Linux / Lightning AI) |
|---|---|
| `.zip` | `unzip file.zip -d datasets/<name>/` |
| `.tar` | `tar -xf file.tar -C datasets/<name>/` |
| `.tar.gz` / `.tgz` | `tar -xzf file.tar.gz -C datasets/<name>/` |
| `.tar.xz` | `tar -xJf file.tar.xz -C datasets/<name>/` |
| `.7z` | `7z x file.7z -o datasets/<name>/` (`apt install p7zip-full`) |
| nested (zip-in-tar etc.) | extract outer, then extract inner the same way |

Inspect what you got before moving things:
```bash
ls -R datasets/<name> | head -40        # see the real structure
find datasets/<name> -maxdepth 2 -type d # list top folders
```

Verify any loader sees data (fast, 20 samples):
```bash
python src/pipeline.py --mode eval --dataset <name> --data-root ./datasets --limit 20
```

---

## 1. IDD — India Driving Dataset (detection)  · **Indian · HIGH priority**

| | |
|---|---|
| **Official** | https://idd.insaan.iiit.ac.in/ → register → **IDD Detection** (VOC-style). License: research-only EULA. |
| **Kaggle mirror** | `manjotpahwa/indian-driving-dataset` (closest to official VOC packaging) |
| **Arrives as** | tar/zip containing `JPEGImages/` + `Annotations/` (Pascal-VOC), sometimes nested under `IDD_Detection/`. |

**Loader needs** (`load_idd` swaps `JPEGImages`→`Annotations` in the path and `.jpg`→`.xml`):
```
datasets/idd/JPEGImages/**/*.jpg
datasets/idd/Annotations/**/*.xml      # same relative path as the image
```

**Prep:**
```bash
# from official .tar
tar -xf idd-detection.tar.gz -C datasets/idd/
# if everything landed under an extra folder, lift the two dirs to the top:
mv datasets/idd/IDD_Detection/JPEGImages datasets/idd/JPEGImages
mv datasets/idd/IDD_Detection/Annotations datasets/idd/Annotations
```
> The image path must literally contain `JPEGImages` and have a parallel `Annotations`
> file with the same name. If that pairing holds, nesting depth doesn't matter.

---

## 2. BDD100K — Berkeley DeepDrive (detection)  · **Western · MEDIUM priority**

| | |
|---|---|
| **Official** | https://bdd-data.berkeley.edu/ → register. Download **`100k images`** + **`Detection 2020 labels`**. BSD-3 license. |
| **Kaggle mirror** | `solesensei/solesensei_bdd100k` (images, canonical) — but usually **lacks the det_20 JSON**. |
| **Arrives as** | images zip (`bdd100k_images_100k.zip`) + labels zip (`bdd100k_det_20_labels_trainval.zip`). |

**Loader needs** (`load_bdd100k` is strict on these exact paths):
```
datasets/bdd100k/images/100k/val/*.jpg
datasets/bdd100k/labels/det_20/det_val.json
```

**Prep:**
```bash
unzip bdd100k_images_100k.zip            -d datasets/bdd100k/   # -> images/100k/{train,val,test}
unzip bdd100k_det_20_labels_trainval.zip -d datasets/bdd100k/   # -> labels/det_20/det_{train,val}.json
# confirm:
ls datasets/bdd100k/images/100k/val | head
ls datasets/bdd100k/labels/det_20/det_val.json
```
> ⚠️ **Prefer the official site for BDD** — the Detection-2020 (`det_20`) labels are what the
> loader parses, and most Kaggle mirrors ship images only or the older 2018 format. Without
> `det_val.json`, BDD eval is skipped (no mAP). Images alone aren't enough.

---

## 3. CCPD — Chinese City Parking Dataset (ANPR)  · **Western · LOW priority**

| | |
|---|---|
| **Official** | https://github.com/detectRecog/CCPD (links via Google Drive / BaiduYun). |
| **Kaggle mirror** | `binh234/ccpd2019` (raw CCPD2019) |
| **Arrives as** | tar/zip of `.jpg` files; **the ground-truth plate is encoded in the filename**. |

**Loader needs** (`load_ccpd` globs images recursively, decodes filename):
```
datasets/ccpd/**/*.jpg          # any depth; folder names irrelevant
```

**Prep:**
```bash
tar -xf ccpd2019.tar -C datasets/ccpd/     # or unzip; structure doesn't matter
```
> 🚫 **Never use a "preprocess/cropped/renamed" CCPD mirror.** The labels live in the
> filename (`...-省A...-...jpg`); any mirror that renames files silently destroys the
> ground truth. Stick to raw CCPD2019. Chinese plates — OCR sanity-check only, not
> Indian-relevant, so this is the most skippable dataset.

---

## 4. LISA — Traffic Light Dataset (signal state)  · **Western · MEDIUM priority**

| | |
|---|---|
| **Official** | UCSD LISA (request form) / Kaggle is the easy path. |
| **Kaggle mirror** | `mbornoe/lisa-traffic-light-dataset` |
| **Arrives as** | zip with `dayTrain/`, `nightTrain/`, `daySequence*/` … each holding `frames/` + `frameAnnotationsBOX.csv`. |

**Loader needs** (`load_lisa` walks for the CSVs and finds frames near them):
```
datasets/lisa/**/frameAnnotationsBOX.csv      # + the frame images alongside
```

**Prep:**
```bash
unzip lisa-traffic-light-dataset.zip -d datasets/lisa/
# confirm at least one CSV exists:
find datasets/lisa -name frameAnnotationsBOX.csv | head
```
> Extract anywhere under `datasets/lisa/`; the walker finds the CSVs regardless of depth.

---

## 5. Indian-LP — Indian License Plates (ANPR)  · **Indian · HIGH priority**

| | |
|---|---|
| **Official** | No single canonical set; assembled from research/Kaggle sources. |
| **Kaggle mirrors** | `saisirishan/indian-vehicle-dataset` · `dataclusterlabs/indian-number-plates-dataset` |
| **Arrives as** | folders of `.jpg`, sometimes with sidecar `.txt`/`.json`/`.xml` labels, sometimes plate-in-filename. |

**Loader needs** (`load_indian_lp` globs images; reads optional sidecar `.txt`/`.json` or plate-in-filename):
```
datasets/indian_lp/**/*.jpg
# optional GT:  <image>.txt  (first line = plate)  OR  <image>.json {"plate": "..."}  OR plate in filename
```

**Prep:**
```bash
unzip indian-vehicle-dataset.zip -d datasets/indian_lp/
```
> Without per-image plate labels the loader can still measure *Indian-format rate* and
> *detection*, just not exact-match accuracy. For a real plate-accuracy number, prefer a
> mirror that ships sidecar labels or plate-in-filename.

---

## 6. UA-DETRAC — multi-object tracking/detection  · **Western · OPTIONAL**

| | |
|---|---|
| **Official** | https://detrac-db.rit.albany.edu/ |
| **Kaggle mirror** | none verified — search: `kaggle datasets list -s "ua-detrac"` |
| **Arrives as** | `Insight-MVT_Annotation_*` image-sequence folders + `DETRAC-*-Annotations-XML/`. |

**Loader needs:**
```
datasets/ua_detrac/Insight-MVT_Annotation_*/        # frame folders
datasets/ua_detrac/DETRAC-*-Annotations-XML/        # XML annotations
```
> Lowest priority — skip unless you specifically want a tracking/throughput benchmark.

---

## Priority order (if time/disk constrained)

1. **IDD** + **Indian-LP** — Indian-domain; these drive the domain-gap number AND are what you fine-tune on.
2. **BDD100K** + **LISA** — Western detection/signal baselines (need official BDD labels).
3. **CCPD** — Chinese-plate OCR sanity check; most skippable.
4. **UA-DETRAC** — optional tracking benchmark.

## What "not feasible right now" actually means

Nothing is blocked. Extract whichever datasets are convenient, point the pipeline at
`./datasets`, and it benchmarks exactly the ones that are ready — skipping the rest with a
clear log line. You can add datasets incrementally and re-run; no big-bang preparation needed.

> **Reportable numbers:** for anything that goes in the submission/report, re-pull from the
> **official** source and cite it (see `01_justifications.md` → dataset sourcing rationale).
> Kaggle mirrors are development proxies only.
