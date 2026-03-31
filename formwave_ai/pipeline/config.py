from pathlib import Path

# Project-root relative defaults (run from repository root)
ROOT = Path(__file__).resolve().parents[2]

# Data layout (keep compatibility with existing data/ folder)
DATA_DIR = ROOT / "formwave_ai/pipeline/data"
RAW_DIR = DATA_DIR / "raw"
LOWRES_DIR = RAW_DIR / "raw_lowres"
CURATED_DIR = RAW_DIR / "curated"
SEGMENTS_FILE = RAW_DIR / "segment_times.json"

# Output area for pipeline artifacts
OUTPUT_DIR = ROOT / "pipeline_outputs"

# Ensure directories exist when modules import config
for p in (RAW_DIR, LOWRES_DIR, CURATED_DIR, OUTPUT_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Failed to create directory {p}: {e}")

# Push-up quality filter thresholds (conservative defaults)
# These can be adjusted by editing this file or overridden at runtime.
PUSHUP_MIN_VALID_FRAME_RATIO = 0.50
PUSHUP_MIN_AVG_JOINT_CONF = 0.55
PUSHUP_MIN_ELBOW_RANGE_DEGREES = 20.0
PUSHUP_MAX_ZERO_CONF_FRAMES_RATIO = 0.30
PUSHUP_MIN_BODY_ALIGNMENT = 0.65
PUSHUP_REVIEW_BODY_ALIGNMENT_MARGIN = 0.07
PUSHUP_MOTION_AMPLITUDE_REVIEW = 0.08
PUSHUP_ACCEPT_SCORE = 0.75
