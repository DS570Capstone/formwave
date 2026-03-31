#!/usr/bin/env python3
"""
FormWave — Import Your Own Videos
===================================
Drop any exercise videos into a folder.
This script will:
  1. Ask you what exercise + camera angle each video is
  2. Organise them into the correct folder structure
  3. Run pose extraction (MMPose / MediaPipe)
  4. Run Gemini annotation
  5. Output the final LLM training dataset

Usage
-----
  python pipeline/import_videos.py --videos_dir ~/Desktop/my_videos
  python pipeline/import_videos.py --videos_dir ~/Desktop/my_videos --exercise squat --camera SIDE
  python pipeline/import_videos.py --videos_dir ~/Desktop/my_videos --dry_run
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

SUPPORTED_EXTS  = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CLIP_DURATION_S = 8       # seconds per clip segment
SPLIT_RATIOS    = {"train": 0.70, "val": 0.15, "test": 0.15}

EXERCISES = [
    "squat", "push_up", "deadlift", "ohp",
    "barbell_row", "curl", "bench_press", "lunge",
]
CAMERAS = ["FRONT", "SIDE", "BACK"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_videos(videos_dir: Path) -> list[Path]:
    vids = []
    for ext in SUPPORTED_EXTS:
        vids.extend(videos_dir.rglob(f"*{ext}"))
    return sorted(vids)


def get_video_info(path: Path) -> dict:
    """Use ffprobe to get duration and fps."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(probe.stdout)
        fmt  = info.get("format", {})
        duration = float(fmt.get("duration", 0))
        fps = 30.0
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                r = stream.get("r_frame_rate", "30/1").split("/")
                fps = float(r[0]) / max(float(r[1]), 1)
                break
        return {"duration": duration, "fps": fps}
    except Exception:
        return {"duration": 0, "fps": 30.0}


def clip_video(src: Path, out_dir: Path, clip_id_start: int) -> list[Path]:
    """
    If video is short (< 12s), keep as-is.
    If long, split into CLIP_DURATION_S segments.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info     = get_video_info(src)
    duration = info["duration"]

    clips    = []
    clip_n   = clip_id_start

    if duration <= CLIP_DURATION_S * 1.5:
        # Short video — use as single clip
        dst = out_dir / f"clip_{clip_n:05d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-an",
            "-vf", "scale=-2:720",
            str(dst), "-loglevel", "quiet",
        ]
        if subprocess.run(cmd).returncode == 0:
            clips.append(dst)
    else:
        # Long video — split into segments
        start = 0
        while start + CLIP_DURATION_S <= duration:
            dst = out_dir / f"clip_{clip_n:05d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(src),
                "-t", str(CLIP_DURATION_S),
                "-c:v", "libx264", "-an",
                "-vf", "scale=-2:720",
                str(dst), "-loglevel", "quiet",
            ]
            if subprocess.run(cmd).returncode == 0 and dst.exists():
                clips.append(dst)
                clip_n += 1
            start += CLIP_DURATION_S

    return clips


def assign_splits(clips: list[Path]) -> dict:
    random.shuffle(clips)
    n     = len(clips)
    n_tr  = max(1, int(n * SPLIT_RATIOS["train"]))
    n_val = max(0, int(n * SPLIT_RATIOS["val"]))
    return {
        "train": clips[:n_tr],
        "val":   clips[n_tr : n_tr + n_val],
        "test":  clips[n_tr + n_val :],
    }


def ask(prompt: str, choices: list, default: str = None) -> str:
    """Simple CLI prompt with numbered choices."""
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        marker = " ← (default)" if c == default else ""
        print(f"  {i}. {c}{marker}")
    while True:
        raw = input("  Enter number or name: ").strip()
        if not raw and default:
            return default
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        if raw.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == raw.lower())
        print("  Please enter a valid number or name.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    videos_dir = Path(args.videos_dir).expanduser()
    data_root  = Path(args.data_dir)

    if not videos_dir.exists():
        print(f"[ERROR] Videos directory not found: {videos_dir}")
        sys.exit(1)

    videos = find_videos(videos_dir)
    if not videos:
        print(f"[ERROR] No video files found in {videos_dir}")
        print(f"  Supported formats: {SUPPORTED_EXTS}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FormWave — Import Your Videos")
    print(f"{'='*60}")
    print(f"  Found {len(videos)} video(s) in {videos_dir}")
    for v in videos:
        info = get_video_info(v)
        print(f"    {v.name:<40} {info['duration']:.1f}s  {info['fps']:.0f}fps")

    # ── Determine exercise + camera ───────────────────────────────────────────
    if args.exercise and args.camera:
        exercise = args.exercise.lower().replace(" ", "_")
        camera   = args.camera.upper()
        print(f"\n  Using: exercise={exercise}, camera={camera}")
    elif args.batch:
        # Batch mode: use filename pattern {exercise}_{camera}_*.mp4
        print("\n  [BATCH MODE] Inferring exercise+camera from filenames...")
        exercise = None
        camera   = None
    else:
        # Interactive
        print("\n  Let's set up your videos (press Enter to accept defaults).")
        exercise = ask(
            "What exercise is in these videos?",
            EXERCISES, default="squat"
        ).lower().replace(" ", "_")
        camera = ask(
            "What camera angle?",
            CAMERAS, default="SIDE"
        ).upper()

    # ── Process each video ────────────────────────────────────────────────────
    all_clips     = []
    global_clip_n = 1

    for vid in videos:
        # Batch: try to parse from filename
        if args.batch or (not args.exercise and not args.camera):
            vid_exercise, vid_camera = _parse_filename(vid.name)
            if not vid_exercise:
                print(f"\n  [SKIP] Cannot infer exercise from: {vid.name}")
                print(f"  Rename files as:  squat_SIDE_myvideo.mp4")
                continue
        else:
            vid_exercise = exercise
            vid_camera   = camera

        if args.dry_run:
            print(f"\n  [DRY RUN] Would process: {vid.name} → {vid_exercise}/{vid_camera}")
            continue

        print(f"\n  Processing: {vid.name}  ({vid_exercise} / {vid_camera})")

        # Clip into segments
        tmp_dir = data_root / "_tmp_clips"
        clips   = clip_video(vid, tmp_dir, global_clip_n)
        global_clip_n += len(clips)
        print(f"  → {len(clips)} clip(s) extracted")

        # Tag with metadata
        for c in clips:
            all_clips.append((c, vid_exercise, vid_camera))

    if args.dry_run:
        print("\n  [DRY RUN] Done. Remove --dry_run to actually process videos.")
        return

    if not all_clips:
        print("[WARN] No clips were generated.")
        return

    # ── Organise into data/ folder structure ──────────────────────────────────
    print(f"\n  Organising {len(all_clips)} clips into data/ structure...")

    # Group by (exercise, camera)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for clip_path, ex, cam in all_clips:
        groups[(ex, cam)].append(clip_path)

    placed_clips = []
    for (ex, cam), clips in groups.items():
        # New workflow: do not create train/val/test splits. Place clips
        # into a single videos folder per exercise/camera.
        dest_dir = data_root / ex / cam / "videos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for c in clips:
            dst = dest_dir / c.name
            shutil.move(str(c), str(dst))
            placed_clips.append(dst)

        print(f"  {ex}/{cam}: placed={len(clips)} -> {dest_dir}")

    # Cleanup temp dir
    tmp_dir = data_root / "_tmp_clips"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  {len(placed_clips)} clips placed into {data_root}/")

    # ── Run Steps 2 & 3 automatically ─────────────────────────────────────────
    if args.skip_pipeline:
        print("\n  [--skip_pipeline] Stopping here. Run manually:")
        print(f"    python pipeline/step2_extract_poses.py --data_dir {data_root}")
        print(f"    python annotate_with_gemini.py --data_dir {data_root}")
        return

    print("\n" + "="*60)
    print("  Running Step 2: Pose Extraction...")
    print("="*60)
    step2 = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "pipeline" / "step2_extract_poses.py"),
         "--data_dir", str(data_root), "--skip_existing"],
        cwd=str(SCRIPT_DIR),
    )

    if step2.returncode != 0:
        print("\n[ERROR] Step 2 failed. Fix errors above then re-run:")
        print(f"  python pipeline/step2_extract_poses.py --data_dir {data_root}")
        return

    print("\n" + "="*60)
    print("  Running Step 3: Gemini Annotation...")
    print("="*60)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\n  [WARN] No GEMINI_API_KEY found — running in dry-run mode for Step 3.")
        print("  Set GEMINI_API_KEY to enable real annotations.")

    step3_cmd = [
        sys.executable, str(SCRIPT_DIR / "annotate_with_gemini.py"),
        "--data_dir", str(data_root),
    ]
    if not api_key:
        step3_cmd.append("--dry_run")

    subprocess.run(step3_cmd, cwd=str(SCRIPT_DIR))

    print("\n" + "="*60)
    print("  ✅ All done!")
    print("="*60)
    print(f"  Training data: {SCRIPT_DIR / 'outputs' / 'train_wave_alpaca.json'}")
    print(f"  Val data:      {SCRIPT_DIR / 'outputs' / 'val_wave_alpaca.json'}")


def _parse_filename(name: str) -> tuple[str, str]:
    """
    Try to infer exercise + camera from filename.
    Expected patterns:
      squat_SIDE_anything.mp4
      push_up_FRONT_clip1.mp4
      ohp_BACK.mp4
    """
    name_lower = name.lower()
    found_ex  = None
    found_cam = None

    for ex in EXERCISES:
        if ex in name_lower or ex.replace("_", "") in name_lower:
            found_ex = ex
            break

    for cam in CAMERAS:
        if cam.lower() in name_lower:
            found_cam = cam
            break

    return found_ex, found_cam or "FRONT"


def parse_args():
    p = argparse.ArgumentParser(
        description="FormWave: Import your own exercise videos"
    )
    p.add_argument("--videos_dir",    required=True,
                   help="Folder containing your video files")
    p.add_argument("--data_dir",      default="./data",
                   help="Root data directory for the pipeline (default: ./data)")
    p.add_argument("--exercise",      default=None,
                   help="Exercise name for all videos (e.g. squat). "
                        "Interactive if not set.")
    p.add_argument("--camera",        default=None,
                   help="Camera angle for all videos: FRONT / SIDE / BACK. "
                        "Interactive if not set.")
    p.add_argument("--batch",         action="store_true",
                   help="Infer exercise+camera from filename: squat_SIDE_clip1.mp4")
    p.add_argument("--api_key",       default=None,
                   help="Gemini API key (or set GEMINI_API_KEY)")
    p.add_argument("--skip_pipeline", action="store_true",
                   help="Only organise files, don't run pose extraction or Gemini")
    p.add_argument("--dry_run",       action="store_true",
                   help="Show what would happen without doing anything")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    run(args)
