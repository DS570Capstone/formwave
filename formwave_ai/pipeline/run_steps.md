# 🚀 FULL PIPELINE GUIDE

This guide explains, step-by-step, how to run the FormWave pipeline from scratch, how to generate more data (videos → segments → labels), how to train and evaluate a model, and how to iterate at scale.

Follow this exactly from the project root (the repository root containing `formwave_ai/`). Use absolute or repository-root-relative paths shown below.

IMPORTANT: The canonical data root used by these scripts is formwave_ai/pipeline/data. Always refer to that directory when inspecting outputs.

## 0. Setup

- From the repository root:

```bash
cd /path/to/formwave            # your repo root, e.g. /home/you/formwave
source venv/bin/activate       # activate the project's venv
```

- Install Python dependencies (either system-wide or inside venv):

```bash
pip install -r requirements.txt || \
  pip install numpy pandas scipy opencv-python scikit-learn joblib yt-dlp ffmpeg-python
```

- Verify pipeline config: open and confirm `DATA_DIR` in `formwave_ai/pipeline/config.py` is correct. If you need a different location, pass `--data_dir` to scripts.

## 1. Download Videos (Step 1)

Use the downloader to fetch full videos (preferred) or run the orchestrator.

- Option A — run the orchestrator (recommended, `simple` mode runs the full simple pipeline):

```bash
python formwave_ai/pipeline/run_pipeline.py --mode simple --data_dir formwave_ai/pipeline/data --max_videos 10
```

- Option B — run the downloader directly (fine-grained control):

```bash
python formwave_ai/pipeline/step1_download_ytd.py \
  --data_dir formwave_ai/pipeline/data \
  --max_videos 10 \
  --force_reencode            # optional: force re-encode to H264 MP4
```

Validation checks after download:
- Verify the raw videos folder exists and has files:

```bash
ls -1 formwave_ai/pipeline/data/raw/videos | wc -l
```

- Check tracking summary (if present):

```bash
python -c "import json; print('segments tracked:', json.load(open('formwave_ai/pipeline/data/download_tracking.json')) if False else 'check tracking file manually')"
```

## 2. Extract Poses (Step 2)

Run the pose extractor on the downloaded videos (writes JSON annotations to `data/annotations`).

```bash
python formwave_ai/pipeline/step2_extract_poses.py \
  --data_dir formwave_ai/pipeline/data \
  --skip_existing            # skip clips that already have annotations
```

Validation checks after pose extraction:
- Count annotation files:

```bash
ls -1 formwave_ai/pipeline/data/annotations | wc -l
```

- Spot-check an annotation JSON:

```bash
python - <<'PY'
import json
f='formwave_ai/pipeline/data/annotations/example_video.json'
print('exists:', __import__('pathlib').Path(f).exists())
print('keys:', list(json.load(open(f)).keys())[:10])
PY
```

## 3. Detect & Cut High-Quality Segments (Step 2b)

Run the wave-based filter which scores and cuts candidate segments from the full-video pose signals. This writes `data/metadata/segments.json` and updates tracking.

```bash
python formwave_ai/pipeline/step2b_filter_segments.py \
  --data_dir formwave_ai/pipeline/data \
  --score_thresh 0.70 \
  --top_k 5 \
  --verbose
```

Validation checks after segment detection:
- Number of segments written:

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('formwave_ai/pipeline/data/metadata/segments.json')
print('segments.json exists:', p.exists())
if p.exists(): print('count:', len(json.load(open(p))))
PY
```

- Confirm a sample processed segment video exists under `data/processed/<exercise>/<camera>/<split>/videos`.

## 4. Build Per-Segment Annotations

Create normalized per-segment annotation JSONs (sliced and resampled pose trajectories + keypoints). These are written under `data/{exercise}/{camera}/{split}/annotations/{segment_id}.json`.

```bash
python formwave_ai/pipeline/build_annotation_jsons.py \
  --data_dir formwave_ai/pipeline/data \
  --target_len 100
```

Validation checks after building annotations:
- Count created per-segment annotation files (example):

```bash
find formwave_ai/pipeline/data -path '*/annotations/*.json' | wc -l
```

## 5. Label Segments (Interactive + Auto)

Interactive labeling (recommended for initial dataset):

```bash
python formwave_ai/pipeline/label_segments.py --data_dir formwave_ai/pipeline/data --interactive --auto_label
```

Auto-labeling only (fast, threshold-based):

```bash
python formwave_ai/pipeline/label_segments.py --data_dir formwave_ai/pipeline/data --auto_label --auto_threshold_good 0.85 --auto_threshold_bad 0.65
```

Build dataset CSV (consolidates `segments.json` + labels into `data/metadata/dataset.csv`):

```bash
python formwave_ai/pipeline/label_segments.py --data_dir formwave_ai/pipeline/data --build_dataset --dataset_out formwave_ai/pipeline/data/metadata/dataset.csv
```

Validation checks after labeling & dataset build:
- Count labeled rows:

```bash
python - <<'PY'
import csv, pathlib
p=pathlib.Path('formwave_ai/pipeline/data/metadata/dataset.csv')
print('dataset exists:', p.exists())
if p.exists(): print('rows:', sum(1 for _ in open(p)) - 1)
PY
```

Labeling strategy notes:
- Interactive: good for quality and edge cases. Use `--auto_label` to pre-fill obvious cases.
- Auto thresholds: choose `--auto_threshold_good` and `--auto_threshold_bad` to speed labeling (higher good threshold => fewer false positives).
- Prioritize segments by score (highest first) to label the most informative examples early.
- To speed up labeling at scale: filter with `--top_k` in detection, batch auto-label, then manually review ambiguous middle-scored items.

## 6. Extract Features

Compute numerical features from per-video annotations and dataset entries. Output defaults to `data/metadata/features.csv`.

```bash
python formwave_ai/pipeline/feature_extraction.py \
  --data_dir formwave_ai/pipeline/data \
  --dataset formwave_ai/pipeline/data/metadata/dataset.csv \
  --out formwave_ai/pipeline/data/metadata/features.csv
```

Validation checks after feature extraction:
- Preview features and row count:

```bash
python - <<'PY'
import csv, pathlib
p=pathlib.Path('formwave_ai/pipeline/data/metadata/features.csv')
print('features exists:', p.exists())
if p.exists():
  r = csv.reader(open(p))
  header = next(r)
  print('cols:', header)
  print('rows:', sum(1 for _ in r))
PY
```

## 7. Train Model

Train a RandomForest on the extracted features. The script will guard against missing dependencies and tiny datasets.

```bash
python formwave_ai/pipeline/train_model.py \
  --features formwave_ai/pipeline/data/metadata/features.csv \
  --out formwave_ai/pipeline/data/metadata/model.pkl
```

Training rules and warnings:
- Minimum: scripts will refuse to train if < 2 labeled samples; in practice, you should have at least 30 labeled examples per class for anything meaningful.
- Small datasets will produce misleadingly high accuracy (overfitting). Always hold out a test set and use cross-validation.
- If you see Accuracy==1.0 on a very small dataset, treat it as a warning, not success.

Suggested evaluation (manual cross-val example using scikit-learn):

```python
# quick check (run in Python / notebook)
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('formwave_ai/pipeline/data/metadata/features.csv')
df = df.dropna(subset=['label'])
X = df[['mean_velocity','std_velocity','rep_count','amplitude']].values
Y = df['label'].map({'g':1,'b':0}).values
clf = RandomForestClassifier(n_estimators=100, random_state=42)
print('cv:', cross_val_score(clf, X, Y, cv=5))
```

## 8. Evaluate Model

- Inspect `formwave_ai/pipeline/data/metadata/model.pkl` (saved by `train_model.py`). The script prints accuracy and a classification report for a held-out test split.
- For stronger evaluation, use an explicit hold-out test set (reserve >=20% data and stratify by label) and compute precision/recall/F1.

## 9. Generate More Data (LOOP)

To scale the dataset repeat the cycle below. Keep everything under `formwave_ai/pipeline/data`.

1. Add new video sources:
   - Add YouTube URLs to your `urls.txt` or update `formwave_ai/pipeline/data/meta.csv` used by the orchestrator.
2. Re-run downloader:

```bash
python formwave_ai/pipeline/step1_download_ytd.py --data_dir formwave_ai/pipeline/data --max_videos 50
```

3. Re-extract poses:

```bash
python formwave_ai/pipeline/step2_extract_poses.py --data_dir formwave_ai/pipeline/data
```

4. Re-run segment detection and cutting (adjust thresholds if you need more/less segments):

```bash
python formwave_ai/pipeline/step2b_filter_segments.py --data_dir formwave_ai/pipeline/data --score_thresh 0.7 --top_k 5
```

5. Build per-segment annotations and label new segments (auto then manual):

```bash
python formwave_ai/pipeline/build_annotation_jsons.py --data_dir formwave_ai/pipeline/data
python formwave_ai/pipeline/label_segments.py --data_dir formwave_ai/pipeline/data --auto_label
```

6. Rebuild dataset, extract features, and retrain:

```bash
python formwave_ai/pipeline/label_segments.py --data_dir formwave_ai/pipeline/data --build_dataset
python formwave_ai/pipeline/feature_extraction.py --data_dir formwave_ai/pipeline/data
python formwave_ai/pipeline/train_model.py --features formwave_ai/pipeline/data/metadata/features.csv --out formwave_ai/pipeline/data/metadata/model.pkl
```

Automation tip: make a short shell script `update_cycle.sh` with the above commands and run it on a schedule or manually after adding new URLs.

## 10. Best Practices

- Always run from the repository root and use the `--data_dir` flag when in doubt.
- Use absolute or repo-root-relative paths (e.g. `formwave_ai/pipeline/data`) in commands and scripts.
- Keep a single source-of-truth dataset CSV: `formwave_ai/pipeline/data/metadata/dataset.csv`. Avoid copies in other folders.
- Back up `data/metadata/model.pkl` and `data/metadata/features.csv` with timestamped filenames when experimenting.
- When adjusting thresholds (`--score_thresh`, `--auto_threshold_good`, `--auto_threshold_bad`), re-run only downstream steps — avoid re-downloading unless needed.
- Track experiments (thresholds, labels counts) in a simple log or spreadsheet.

## Quick Troubleshooting

- If a script complains about missing packages: `pip install -r requirements.txt`.
- If segments.json is missing, you can attempt to rebuild from tracking: `python formwave_ai/pipeline/rebuild_segments_meta.py` (if present) or re-run `step2b_filter_segments.py`.
- If per-segment annotation generation writes 0 files: check `data/annotations` existence and that `data/metadata/segments.json` entries have valid `video_id` and `start`/`end` values.

## Useful one-liners

- Count segments:
```bash
python - <<'PY'
import json
p='formwave_ai/pipeline/data/metadata/segments.json'
print(len(json.load(open(p))) if __import__('pathlib').Path(p).exists() else 0)
PY
```

- Count labeled features:
```bash
python - <<'PY'
import csv
p='formwave_ai/pipeline/data/metadata/features.csv'
print('features rows:', sum(1 for _ in open(p)) - 1 if __import__('pathlib').Path(p).exists() else 0)
PY
```

---

If you want, I can now:
- run a smoke test of the guide on your workspace, or
- produce a small `update_cycle.sh` automation script that runs the loop.
