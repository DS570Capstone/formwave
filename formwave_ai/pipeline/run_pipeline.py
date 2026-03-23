"""
Orchestrator for the FormWave pipeline.

Usage:
  python pipeline/run_pipeline.py --mode smart
  python pipeline/run_pipeline.py --mode simple
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path
import types
from typing import Dict

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import config


# =========================================================
# 🔥 GLOBALS
# =========================================================
TRACKING_FILE = "download_tracking.json"


# =========================================================
# LOGGING
# =========================================================
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


# =========================================================
# 🔥 ARGUMENTS (UPDATED)
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Run unified FormWave pipeline")

    p.add_argument("--mode", choices=["smart", "simple"], required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--verbose", action="store_true")

    # 🔥 NEW FLAGS
    p.add_argument("--reset_tracking", action="store_true", help="Reset download tracking file")
    p.add_argument("--clean_data", action="store_true", help="Delete all downloaded data")

    return p.parse_args()


# =========================================================
# 🔥 CLEAN FUNCTIONS
# =========================================================
def reset_tracking(data_dir: Path):
    tracking_path = data_dir / TRACKING_FILE
    if tracking_path.exists():
        tracking_path.unlink()
        print(f"🧹 Tracking reset: {tracking_path}")
    else:
        print("⚠️ No tracking file found")


def clean_data(data_dir: Path):
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"🧹 Data directory removed: {data_dir}")
    else:
        print("⚠️ Data directory not found")


# =========================================================
# SMART MODE
# =========================================================
def run_smart(data_dir: Path, max_videos: int):
    logger = logging.getLogger(__name__)

    from modules import downloader, segment_detector, highres_downloader

    meta_csv = data_dir / "meta.csv"

    logger.info("Step A: Download low-res videos")
    n_low = downloader.download_lowres(meta_csv, data_dir / "raw" / "raw_lowres")
    logger.info("Low-res downloaded: %d", n_low)

    logger.info("Step B: Multi-signal detection")

    multi_out = data_dir / "raw" / "multi_segment_times.json"
    segment_detector.detect_directory_for_signals(
        data_dir / "raw" / "raw_lowres",
        multi_out,
        signals=["wrist", "knee", "hip"],
    )

    import json

    nested = json.load(open(multi_out))

    weights = {"wrist": 1 / 3, "knee": 1 / 3, "hip": 1 / 3}
    aggregated: Dict[str, dict] = {}

    for vid, sigs in nested.items():
        raw_scores = {s: float(sigs.get(s, {}).get("score", 0.0)) for s in weights}
        max_score = max(raw_scores.values()) if raw_scores else 1e-8

        final_score = 0.0
        signals_out = {}

        for s, w in weights.items():
            entry = sigs.get(s, {})
            score = float(entry.get("score", 0.0))
            norm = score / max_score
            final_score += w * norm

            signals_out[s] = {
                "score": round(score, 4),
                "start": entry.get("start"),
                "end": entry.get("end"),
            }

        best_signal = max(raw_scores.items(), key=lambda kv: kv[1])[0]

        aggregated[vid] = {
            "best_signal": best_signal,
            "final_score": round(final_score, 4),
            "signals": signals_out,
        }

    seg_out = data_dir / "raw" / "segment_times.json"
    with open(seg_out, "w") as f:
        json.dump(aggregated, f, indent=2)

    logger.info("Step C: Download high-res segments")

    selected = {}
    for vid, info in aggregated.items():
        best = info["best_signal"]
        sig = info["signals"][best]
        selected[vid] = sig

    tmp_selected = data_dir / "raw" / "selected_segment_times.json"
    with open(tmp_selected, "w") as f:
        json.dump(selected, f, indent=2)

    n_high = highres_downloader.download_highres_segments(
        meta_csv, tmp_selected, data_dir / "raw" / "curated"
    )

    logger.info("High-res slices downloaded: %d", n_high)


# =========================================================
# SIMPLE MODE
# =========================================================
def run_simple(data_dir: Path, max_videos: int):
    logger = logging.getLogger(__name__)

    logger.info("Running simple downloader")

    try:
        import simple_downloader as sd

        args = types.SimpleNamespace(
            data_dir=str(data_dir),
            exercise=None,
            camera=None,
            max_videos=max_videos,
            seed=42,
            force_reprocess=False,
            reset_tracking=False,
        )

        # Step 1: download full videos
        sd.run(args)

        # Step 2: extract poses from full videos
        logger.info("Step 2: Extract poses (full videos)")
        try:
            import step2_extract_poses as s2
            s2_args = types.SimpleNamespace(data_dir=str(data_dir), skip_existing=False, save_keypoints=False, verbose=False, force_reprocess=False, reset_tracking=False)
            s2.run(s2_args)
        except Exception as e:
            logger.error("Pose extraction failed: %s", e)

        # Step 2b: wave-based filtering -> cut valid segments
        logger.info("Step 2b: Wave-based filtering")
        try:
            import step2b_filter_segments as s2b
            s2b.process_directory(data_dir)
        except Exception as e:
            logger.error("Wave filtering failed: %s", e)

    except Exception as e:
        logger.error("Downloader failed: %s", e)
        return


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    setup_logging(args.verbose)

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR

    print(f"📁 Using data dir: {data_dir}")

    # 🔥 CLEAN BEFORE RUN
    if args.clean_data:
        clean_data(data_dir)

    if args.reset_tracking:
        reset_tracking(data_dir)

    # 🔥 RUN
    if args.mode == "smart":
        run_smart(data_dir, args.max_videos)
    else:
        run_simple(data_dir, args.max_videos)


if __name__ == "__main__":
    main()