#!/usr/bin/env python3
"""Rebuild data/metadata/segments.json from tracking.json generated_segments lists.

Creates best-effort segment entries (start/end/score=0) so labeling can proceed.
"""
from pathlib import Path
import json
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    args = p.parse_args()

    data_root = Path(args.data_dir)
    tracking_path = data_root / "metadata" / "tracking.json"
    out_path = data_root / "metadata" / "segments.json"

    if not tracking_path.exists():
        print(f"Tracking file not found: {tracking_path}")
        return

    try:
        with open(tracking_path, "r") as f:
            tracking = json.load(f)
    except Exception as e:
        print("Failed to load tracking.json:", e)
        return

    segments = []
    clips = tracking.get("clips", {})
    for clip_id, info in clips.items():
        gen = info.get("generated_segments", [])
        for gp in gen:
            # Ensure path is relative to data_root when possible
            gp_path = Path(gp)
            try:
                rel = str(gp_path)
            except Exception:
                rel = gp

            seg_meta = {
                "video_id": clip_id,
                "segment_id": Path(rel).name,
                "start": 0.0,
                "end": 0.0,
                "score": 0.0,
                "file_path": rel,
            }
            segments.append(seg_meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"Wrote {len(segments)} segment entries to {out_path}")


if __name__ == "__main__":
    main()
