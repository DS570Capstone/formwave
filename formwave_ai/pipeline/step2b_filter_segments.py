#!/usr/bin/env python3
"""Step 2b: Filter extracted pose signals into valid exercise segments.

Reads annotation JSONs produced by `step2_extract_poses.py`, selects a
primary 1D signal (arm/legs/trajectory), runs `wave_filter.detect_valid_segments`,
scores segments, and writes cut video segments into `data/processed/{split}/`.

Usage:
    python pipeline/step2b_filter_segments.py --data_dir ./data --score_thresh 0.75 --top_k 5
"""
from pathlib import Path
import argparse
import json
import logging
import hashlib
import sys
from typing import List

import numpy as np

# Ensure top-level package imports work (match other pipeline steps)
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules import wave_filter
from pipeline.tracking import load_tracking, save_tracking, mark_clip_status


SPLIT_THRESH = (70, 85)  # <70 train, <85 val, else test
DEFAULT_SCORE_THRESH = 0.75
MIN_SEGMENT_DURATION = 3.0  # seconds
MERGE_GAP = 2.0  # seconds
DEFAULT_TOP_K = 5


def deterministic_split(video_id: str) -> str:
    h = hashlib.md5(video_id.encode("utf-8")).digest()
    v = int.from_bytes(h, "big") % 100
    if v < SPLIT_THRESH[0]:
        return "train"
    if v < SPLIT_THRESH[1]:
        return "val"
    return "test"


PREFERRED_SIGNALS = [
    "arm_trajectory",
    "arm_Trajectory",
    "legs_trajectory",
    "shoulder_trajectory",
    "bar_path_trajectory",
    "trajectory",
    "core_",
]


def select_signal_from_annotation(ann: dict) -> (str, List[float]):
    for key in PREFERRED_SIGNALS:
        if key in ann and ann.get(key) is not None:
            return key, ann.get(key)
    # fallback: look for any list-valued field
    for k, v in ann.items():
        if isinstance(v, list) and len(v) > 5 and all(isinstance(x, (int, float)) for x in v[:5]):
            return k, v
    return None, []


def process_directory(data_root: Path, score_thresh: float = DEFAULT_SCORE_THRESH, top_k: int = DEFAULT_TOP_K):
    logger = logging.getLogger(__name__)
    tracking = load_tracking(data_root)

    ann_paths = sorted(data_root.rglob("annotations/*.json"))
    if not ann_paths:
        logger.error("No annotation JSONs found under %s", data_root)
        return

    created = 0

    for ann_path in ann_paths:
        try:
            ann = json.load(open(ann_path))
        except Exception as e:
            logger.warning("Failed to load %s: %s", ann_path, e)
            continue

        clip_id = ann.get("video_id") or ann_path.stem
        exercise = ann.get("exercise", "unknown")
        split_folder = ann.get("split") or deterministic_split(clip_id)

        sig_key, sig = select_signal_from_annotation(ann)
        if not sig_key or not sig:
            logger.info("[VIDEO SKIPPED] %s no usable signal", clip_id)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="no_signal")
            save_tracking(tracking, data_root)
            continue

        fps = float(ann.get("fps", 30.0))
        sig_arr = np.asarray(sig, dtype=float)

        logger.info("Processing %s — signal=%s fps=%.2f", clip_id, sig_key, fps)

        segments = wave_filter.detect_valid_segments(sig_arr, int(round(fps)))
        if not segments:
            logger.info("[VIDEO SKIPPED] %s no valid motion detected", clip_id)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="no_valid_segments")
            save_tracking(tracking, data_root)
            continue

        # Merge nearby segments (gap < MERGE_GAP)
        segs_sorted = sorted(segments, key=lambda x: x[0])
        merged = []
        cur_s, cur_e = segs_sorted[0]
        for s, e in segs_sorted[1:]:
            if s - cur_e <= MERGE_GAP:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        # Score and filter segments
        candidates = []  # list of (start, end, score)
        for s, e in merged:
            duration = e - s
            if duration < MIN_SEGMENT_DURATION:
                logger.info("[SEGMENT DROPPED SHORT] %s %.2f-%.2f dur=%.2f", clip_id, s, e, duration)
                continue

            s_frame = max(0, int(round(s * fps)))
            e_frame = min(len(sig_arr), int(round(e * fps)))
            sig_segment = sig_arr[s_frame:e_frame]
            score = wave_filter.score_segment(sig_segment)

            if score < score_thresh:
                logger.info("[SEGMENT DROPPED LOW SCORE] %s %.2f-%.2f score=%.3f", clip_id, s, e, score)
                continue

            candidates.append((s, e, score))

        if not candidates:
            logger.info("[VIDEO SKIPPED] %s no segments passed filters", clip_id)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="no_good_segments")
            save_tracking(tracking, data_root)
            continue

        # Keep top-k by score, then sort by time for deterministic output
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:top_k]
        candidates = sorted(candidates, key=lambda x: x[0])

        # locate original full video under data/raw/videos/{video_id}.mp4
        video_path = data_root / "raw" / "videos" / f"{clip_id}.mp4"
        if not video_path.exists():
            logger.warning("Original video not found for %s — expected: %s", clip_id, video_path)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="video_missing")
            save_tracking(tracking, data_root)
            continue

        out_base = data_root / "processed" / split_folder
        out_base.mkdir(parents=True, exist_ok=True)

        # maintain per-clip generated list to avoid duplicates
        clip_entry = tracking.setdefault("clips", {}).setdefault(clip_id, {})
        gen_list = clip_entry.setdefault("generated_segments", [])

        seg_idx = 0
        for s, e, score in candidates:
            seg_idx += 1
            out_name = f"{clip_id}_seg{seg_idx:02d}.mp4"
            out_path = out_base / out_name

            # skip existing
            if str(out_path) in gen_list or out_path.exists():
                logger.info("[SEGMENT DROPPED LOW SCORE] %s already exists", out_path.name)
                continue

            ok = wave_filter.cut_video_segment(video_path, out_path, s, e)
            if ok:
                gen_list.append(str(out_path))
                logger.info("[SEGMENT KEPT] %s start=%.2f end=%.2f score=%.3f", clip_id, s, e, score)
                created += 1
            else:
                logger.warning("[SEGMENT FAILED] %s start=%.2f end=%.2f", clip_id, s, e)

        # persist generated list and mark clip
        clip_entry["generated_segments"] = gen_list
        if gen_list:
            mark_clip_status(tracking, clip_id, processed=True, status="clips_generated")
        else:
            mark_clip_status(tracking, clip_id, processed=False, status="no_segments")

        save_tracking(tracking, data_root)

    logging.info("Segment extraction complete — created=%d", created)


def parse_args():
    p = argparse.ArgumentParser(description="Step 2b: Filter segments from pose signals")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--score_thresh", type=float, default=DEFAULT_SCORE_THRESH,
                   help=f"Minimum score threshold (default: {DEFAULT_SCORE_THRESH})")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                   help=f"Maximum segments to keep per video (default: {DEFAULT_TOP_K})")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")
    data_root = Path(args.data_dir)
    process_directory(data_root, score_thresh=args.score_thresh, top_k=args.top_k)


if __name__ == "__main__":
    main()
