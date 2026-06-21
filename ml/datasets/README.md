# datasets/ — benchmark data placement

Place each dataset in the subfolder below. The pipeline's `DatasetManager`
locates them under `--data-root` (default `./datasets`) and **skips gracefully
with download instructions** if a dataset is absent. This folder's contents are
git-ignored (except this README).

## Western (domain gap expected)

| Subfolder | Dataset | Source | Expected layout |
|---|---|---|---|
| `datasets/bdd100k/` | BDD100K detection | register: https://bdd-data.berkeley.edu/ | `images/100k/val/*.jpg` + `labels/det_20/det_val.json` |
| `datasets/ccpd/` | CCPD (Chinese plates) | https://github.com/detectRecog/CCPD | any `*.jpg` (annotations encoded in the filename) |
| `datasets/lisa/` | LISA Traffic Light | Kaggle: `mbornoe/lisa-traffic-light-dataset` | `**/frameAnnotationsBOX.csv` + frame images |
| `datasets/ua_detrac/` | UA-DETRAC | https://detrac-db.rit.albany.edu/ | `Insight-MVT_Annotation_*/` + `DETRAC-*-Annotations-XML/` |

## Indian (primary target)

| Subfolder | Dataset | Source | Expected layout |
|---|---|---|---|
| `datasets/idd/` | IDD Detection | register: https://idd.insaan.iiit.ac.in/ | `JPEGImages/**/*.jpg` + `Annotations/**/*.xml` (VOC) |
| `datasets/indian_lp/` | Indian License Plates | Kaggle: `saisirishan/indian-vehicle-dataset` (or equivalent) | `**/*.jpg` (+ optional `*.txt`/`*.json` labels, or plate text in filename) |

## Run after placing data

```bash
python src/pipeline.py --mode eval --dataset idd --data-root ./datasets --limit 500
python src/pipeline.py --mode benchmark --data-root ./datasets --limit 500
```

Notes:
- **IDD domain gap:** auto-rickshaw / cycle-rickshaw / cart / animal are **not** COCO classes, so a COCO-pretrained detector misses them — this is the structural gap the baseline is meant to expose.
- **CCPD:** the Latin part of the plate is used for fair CER vs an English OCR model.
- **Indian LP:** if no ground-truth text is present, only the Indian-format pass-rate is reported.
