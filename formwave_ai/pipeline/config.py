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
