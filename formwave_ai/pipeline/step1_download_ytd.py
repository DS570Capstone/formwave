#!/usr/bin/env python3
"""
FormWave — Step 1: Download & Clip YouTube Videos
==================================================
Downloads exercise videos from YouTube using yt-dlp, then clips each
video into fixed-length segments and organises them into:

    data/{exercise}/{camera}/{split}/videos/clip_XXXX.mp4

Usage
-----
  pip install yt-dlp
  python pipeline/step1_download.py
  python pipeline/step1_download.py --exercise squat --camera FRONT --max_videos 20
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import shutil
import cv2
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Video sources: curated YouTube search terms per exercise + camera angle
# Add more URLs or search terms as needed.
# ─────────────────────────────────────────────────────────────────────────────

# Format: (exercise_name, camera_angle, youtube_search_query)
# yt-dlp will pull the top N results from each query.
EXERCISE_SOURCES = [
    # ── Push-Up ────────────────────────────────────────────────────────────
    ("push_up",   "FRONT", "correct push up form tutorial front view"),
    ("push_up",   "SIDE",  "push up side view form check"),
    ("push_up",   "SIDE",  "push up form mistakes side view"),

    # # ── Squat ──────────────────────────────────────────────────────────────
    # ("squat",     "FRONT", "barbell squat front view correct form"),
    # ("squat",     "SIDE",  "squat side view form tutorial"),
    # ("squat",     "SIDE",  "squat depth mistake side view"),

    # # ── Deadlift ───────────────────────────────────────────────────────────
    # ("deadlift",  "SIDE",  "deadlift side view correct form tutorial"),
    # ("deadlift",  "SIDE",  "deadlift form mistakes analysis"),

    # # ── Overhead Press ─────────────────────────────────────────────────────
    # ("ohp",       "FRONT", "overhead press front view form tutorial"),
    # ("ohp",       "SIDE",  "overhead press side view analysis"),

    # # ── Barbell Row ────────────────────────────────────────────────────────
    # ("barbell_row", "SIDE", "barbell row side view form tutorial"),
    # ("barbell_row", "SIDE", "barbell row mistakes side view"),

    # # ── Bicep Curl ─────────────────────────────────────────────────────────
    # ("curl",       "FRONT", "bicep curl front view form check"),
    # ("curl",       "SIDE",  "bicep curl cheating form side view"),
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# clip duration in seconds (each video → multiple clips of this length)
CLIP_DURATION_S = 8
# max simultaneous downloads per query
MAX_PER_QUERY   = 5


def check_deps():
    for cmd in ["yt-dlp", "ffmpeg"]:
        r = subprocess.run(["which", cmd], capture_output=True)
        if r.returncode != 0:
            print(f"[ERROR] '{cmd}' not found. Install with:")
            if cmd == "yt-dlp":
                print("  pip install yt-dlp")
            else:
                print("  brew install ffmpeg  (macOS)")
            sys.exit(1)
    print("[OK] yt-dlp and ffmpeg found.")


# Tracking helpers (lazy import to avoid CLI import issues)
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from pipeline.tracking import (
        load_tracking, save_tracking, is_video_downloaded,
        mark_video_downloaded, mark_video_clips_generated, mark_clip_status, reset_tracking,
    )
except Exception:
    # fallback if module not available as package
    from tracking import (
        load_tracking, save_tracking, is_video_downloaded,
        mark_video_downloaded, mark_video_clips_generated, mark_clip_status, reset_tracking,
    )


def download_video(query: str, out_dir: Path, max_results: int) -> list[Path]:
    """Search top N results, skip already-tracked videos and download new ones."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Get candidate ids
    cmd_ids = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        f"ytsearch{max_results}:{query}",
        "--get-id",
        "--no-playlist",
        "--quiet",
    ]
    print(f"  Searching: '{query}' (top {max_results})")
    res = subprocess.run(cmd_ids, capture_output=True, text=True)
    ids = [l.strip() for l in res.stdout.splitlines() if l.strip()]

    # data_root is the project data root passed to the CLI
    data_root = out_dir.parents[2]
    tracking = load_tracking(data_root)
    downloaded_files = []

    for vid in ids:
        # Prefer checking the actual file on disk first. If the file exists,
        # skip downloading. If tracking says downloaded but file is missing,
        # log a warning and proceed to (re)download.
        out_path = out_dir / f"{vid}.mp4"
        if out_path.exists() and not getattr(download_video, "force", False):
            print(f"  [VIDEO SKIP] {vid} already exists on disk")
            continue

        if is_video_downloaded(tracking, vid) and not out_path.exists():
            print(f"  [WARNING] tracking mismatch → redownloading {vid}")

        # Download this specific video id. Try H264/MP4 preferred formats to avoid AV1.
        url = f"https://www.youtube.com/watch?v={vid}"
        out_template = str(out_dir / "%(id)s.%(ext)s")

        # Prefer MP4 with h264 (avc1). Try preferred formats sequentially.
        formats_to_try = [
            "best[ext=mp4][vcodec*=avc1]",
            "bestvideo[vcodec*=avc1]+bestaudio/best[vcodec*=avc1]",
            # legacy fallback
            "bv*[vcodec!=av01][height<=720]+ba/b[height<=720]",
        ]

        out_path = out_dir / f"{vid}.mp4"
        success = False
        for fmt in formats_to_try:
            cmd_dl = [
                "yt-dlp",
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                    "--format", fmt,
                    "--merge-output-format", "mp4",
                    "--recode-video", "mp4",
                "--output", out_template,
                url,
                "--no-playlist",
                "--quiet",
                "--restrict-filenames",
            ]

            r = subprocess.run(cmd_dl, check=False)
            # If got file, break
            if out_path.exists():
                success = True
                break

        if success:
            downloaded_files.append(out_path)
            # mark video downloaded in tracking
            mark_video_downloaded(tracking, vid, url, str(out_path))
            save_tracking(tracking, data_root)
            print(f"  [VIDEO DOWNLOADED] {vid} -> {out_path.name}")
        else:
            print(f"  [VIDEO FAILED] {vid}")

    return sorted(downloaded_files)


    # NOTE: This downloader no longer clips videos. It downloads full
    # videos into data/raw/videos/ and relies on later pipeline steps
    # (pose extraction + wave-based filtering) to cut segments.


def run(args):
    data_root = Path(args.data_dir)
    raw_videos_dir = data_root / "raw" / "videos"
    raw_videos_dir.mkdir(parents=True, exist_ok=True)

    # Setup tracking
    if getattr(args, "reset_tracking", False):
        reset_tracking(data_root)
    tracking = load_tracking(data_root)

    # Filter sources if exercise / camera specified
    sources = EXERCISE_SOURCES
    if args.exercise:
        sources = [s for s in sources if s[0] == args.exercise.lower()]
    if args.camera:
        sources = [s for s in sources if s[1] == args.camera.upper()]

    if not sources:
        print("[ERROR] No matching sources found. Check --exercise and --camera args.")
        sys.exit(1)

    check_deps()

    all_stats = []
    global_clip_id = 1

    # Download videos for all queries into data/raw/videos (no clipping).
    for exercise, camera, query in sources:
        print(f"\n{'='*60}")
        print(f"  Query: {query}  (exercise={exercise} camera={camera})")
        print(f"{'='*60}")

        # Respect force_reprocess flag via attribute on function
        download_video.force = bool(getattr(args, "force_reprocess", False))
        downloaded = download_video(
            query, raw_videos_dir,
            max_results=min(args.max_videos, MAX_PER_QUERY),
        )
        print(f"  Downloaded {len(downloaded)} video(s)")

        if not downloaded:
            print("  [WARN] No videos downloaded for this query. Skipping.")
            continue

        for vid in downloaded:
            # Validate downloaded video and re-encode if necessary
            frames = validate_video(vid)
            if frames is None:
                print(f"  [VIDEO SKIPPED] {vid.name} (corrupt or unreadable)")
                try:
                    vid.unlink()
                except Exception:
                    pass
                continue

            if frames < 30 or getattr(args, "force_reencode", False):
                # try re-encoding to h264/mp4
                print(f"  [VIDEO VALIDATION] {vid.name} frames={frames} -> re-encoding")
                fixed = reencode_to_h264(vid)
                if fixed:
                    frames2 = validate_video(vid)
                    print(f"  [VIDEO FIXED] {vid.name} frames={frames2}")
                else:
                    print(f"  [VIDEO SKIPPED] {vid.name} re-encode failed")
                    try:
                        vid.unlink()
                    except Exception:
                        pass
                    continue

            all_stats.append({"exercise": exercise, "camera": camera, "downloaded_videos": 1, "video_path": str(vid)})

    # Summary
    print("\n" + "="*60)
    print("  Step 1 Complete — Download & Clip Summary")
    print("="*60)
    # Summary
    print("\n" + "="*60)
    print("  Step 1 Complete — Download Summary")
    print("="*60)
    for s in all_stats:
        print(f"  Query (exercise={s['exercise']}) {s['camera']}  | downloaded: {s['downloaded_videos']}  path={s.get('video_path')}")

    print(f"\n  Data root: {data_root}")
    print("  Run Step 2 next:  python pipeline/step2_extract_poses.py")


def validate_video(video_path: Path) -> Optional[int]:
    """Return frame count or None if file unreadable."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return frame_count
    except Exception:
        return None


def reencode_to_h264(video_path: Path) -> bool:
    """Re-encode input video to H264 mp4 in-place (writes temp then replaces).

    Returns True on success.
    """
    inp = str(video_path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    tmp_p = Path(tmp_path)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", inp,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(tmp_p),
    ]
    try:
        subprocess.run(cmd, check=True)
        # replace original
        shutil.move(str(tmp_p), inp)
        return True
    except Exception:
        try:
            if tmp_p.exists():
                tmp_p.unlink()
        except Exception:
            pass
        return False


def parse_args():
    p = argparse.ArgumentParser(description="FormWave Step 1: Download YouTube videos")
    p.add_argument("--data_dir",    default="./data",         help="Root data dir")
    p.add_argument("--exercise",    default=None,             help="Filter by exercise (e.g. squat)")
    p.add_argument("--camera",      default=None,             help="Filter by camera (e.g. SIDE)")
    p.add_argument("--max_videos",  type=int, default=5,      help="Max videos per query (default: 5)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--force_reprocess", action="store_true",
                   help="Ignore tracking and force redownload/reprocess")
    p.add_argument("--force_reencode", action="store_true",
                   help="Force re-encoding of downloaded videos to H264 MP4")
    p.add_argument("--reset_tracking", action="store_true",
                   help="Delete tracking.json and start fresh")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    run(args)
