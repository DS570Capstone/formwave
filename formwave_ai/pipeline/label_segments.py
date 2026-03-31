#!/usr/bin/env python3
"""Interactive labeling tool for segments produced by the pipeline.

Creates/updates `data/labels/labels.csv` and can optionally build a
simple dataset CSV combining features and labels.

Usage:
  python pipeline/label_segments.py --data_dir ./data --interactive
"""
from pathlib import Path
import argparse
import csv
import json
import time
import subprocess
from typing import Optional


def ensure_dirs(data_root: Path):
    (data_root / "labels").mkdir(parents=True, exist_ok=True)
    (data_root / "metadata").mkdir(parents=True, exist_ok=True)


def load_segments(data_root: Path):
    seg_file = data_root / "metadata" / "segments.json"
    if not seg_file.exists():
        return []
    try:
        with open(seg_file, "r") as f:
            return json.load(f)
    except Exception:
        return []


def append_label_csv(data_root: Path, row: dict):
    labels_csv = data_root / "labels" / "labels.csv"
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["video_id", "segment_id", "start", "end", "score", "file_path", "label", "timestamp"])
        w.writerow([row.get(k, "") for k in ["video_id", "segment_id", "start", "end", "score", "file_path", "label", "timestamp"]])


def label_with_llm(segment: dict) -> Optional[str]:
    """Stub for LLM-based labeling. Replace with real API call.

    As a simple heuristic, return 'g' for high-score segments.
    """
    try:
        score = float(segment.get("score", 0.0))
        if score >= 0.9:
            return "g"
        return None
    except Exception:
        return None


def play_segment(segment: dict):
    path = Path(segment["file_path"])
    if not path.exists():
        print(f"File not found: {path}")
        return
    # spawn ffplay for user to view quickly; non-blocking
    try:
        subprocess.run(["ffplay", "-autoexit", "-nodisp", str(path)], check=False)
    except FileNotFoundError:
        print("ffplay not available — skipping preview")


def build_dataset(data_root: Path, out_csv: Path):
    # Merge segments.json with labels.csv into a dataset CSV
    segments = load_segments(data_root)
    labels_path = data_root / "labels" / "labels.csv"
    label_map = {}
    if labels_path.exists():
        with open(labels_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                label_map[row["segment_id"]] = row["label"]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "file_path", "duration", "score", "label"])
        for s in segments:
            seg_id = s.get("segment_id")
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            duration = end - start
            score = s.get("score", 0.0)
            label = label_map.get(seg_id, "")
            w.writerow([seg_id, s.get("file_path", ""), duration, score, label])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--auto_label", action="store_true", help="Use simple LLM stub or thresholds to auto-label when available")
    p.add_argument("--auto_threshold_good", type=float, default=0.85,
                   help="Score threshold >= this => auto-label GOOD (default: 0.85)")
    p.add_argument("--auto_threshold_bad", type=float, default=0.65,
                   help="Score threshold <= this => auto-label BAD (default: 0.65)")
    p.add_argument("--build_dataset", action="store_true")
    p.add_argument("--dataset_out", default="./data/metadata/dataset.csv")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_dir)
    ensure_dirs(data_root)

    segments = load_segments(data_root)
    if not segments:
        print("No segments metadata found at data/metadata/segments.json")
        return

    if args.interactive:
        for seg in segments:
            seg_id = seg.get("segment_id")
            print("\nSegment:", seg_id)
            print(f" video: {seg.get('video_id')} start={seg.get('start')} end={seg.get('end')} score={seg.get('score')}")
            if args.auto_label:
                # Try threshold-based auto-labeling first
                try:
                    score = float(seg.get("score", 0.0))
                except Exception:
                    score = 0.0

                if score >= args.auto_threshold_good:
                    label = "g"
                    print(f"[AUTO GOOD] score={score}")
                elif score <= args.auto_threshold_bad:
                    label = "b"
                    print(f"[AUTO BAD] score={score}")
                else:
                    # fallback to LLM stub if available
                    auto = label_with_llm(seg)
                    if auto:
                        label = auto
                        print(f"Auto-label: {label}")
                    else:
                        label = None
            else:
                label = None

            if label is None:
                if not args.interactive:
                    # Non-interactive and no auto label: skip
                    continue
                try:
                    ans = input("Is this GOOD or BAD? (g/b) [s=skip, p=play]: ").strip().lower()
                except KeyboardInterrupt:
                    print("\nInterrupted")
                    break
                if ans == "p":
                    play_segment(seg)
                    ans = input("Is this GOOD or BAD? (g/b) [s=skip]: ").strip().lower()
                if ans == "g":
                    label = "g"
                elif ans == "b":
                    label = "b"
                else:
                    print("Skipping")
                    continue

            row = {
                "video_id": seg.get("video_id"),
                "segment_id": seg.get("segment_id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "score": seg.get("score"),
                "file_path": seg.get("file_path"),
                "label": label,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_label_csv(data_root, row)
            print(f"Saved label {label} for {seg_id}")

    if args.build_dataset:
        out = Path(args.dataset_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        build_dataset(data_root, out)
        print(f"Wrote dataset to {out}")


if __name__ == "__main__":
    main()
