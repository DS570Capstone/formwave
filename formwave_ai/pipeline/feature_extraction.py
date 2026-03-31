#!/usr/bin/env python3
"""Extract simple numerical features from segment videos using pose annotations.

Outputs data/metadata/features.csv with columns:
  segment_id, mean_velocity, std_velocity, rep_count, amplitude, label

Uses dataset CSV or segments.json to locate segments and annotations.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from typing import List

from scipy.signal import find_peaks


PREFERRED_SIGNALS = [
    "arm_trajectory",
    "arm_Trajectory",
    "legs_trajectory",
    "shoulder_trajectory",
    "bar_path_trajectory",
    "trajectory",
    "core_",
]


def select_signal(ann: dict):
    for key in PREFERRED_SIGNALS:
        if key in ann and isinstance(ann.get(key), list) and len(ann.get(key)) > 5:
            return key, np.asarray(ann.get(key), dtype=float)
    # fallback: first sufficiently long numeric list
    for k, v in ann.items():
        if isinstance(v, list) and len(v) > 5 and all(isinstance(x, (int, float)) for x in v[:5]):
            return k, np.asarray(v, dtype=float)
    return None, None


def extract_features_from_segment(sig: np.ndarray, fps: float, start: float, end: float):
    # select segment slice; if start==end or invalid, use full signal
    if start is None or end is None or end <= start:
        seg = sig
        duration = len(sig) / fps if fps and fps > 0 else len(sig)
    else:
        s_idx = max(0, int(round(start * fps)))
        e_idx = min(len(sig), int(round(end * fps)))
        if e_idx <= s_idx:
            seg = sig
            duration = len(sig) / fps if fps and fps > 0 else len(sig)
        else:
            seg = sig[s_idx:e_idx]
            duration = (e_idx - s_idx) / fps if fps and fps > 0 else (e_idx - s_idx)

    if len(seg) < 2:
        return {"mean_velocity": 0.0, "std_velocity": 0.0, "rep_count": 0, "amplitude": 0.0}

    # velocity (per second)
    vel = np.diff(seg) * fps
    mean_v = float(np.mean(np.abs(vel)))
    std_v = float(np.std(vel))

    # amplitude
    amp = float(np.ptp(seg))

    # rep count via peak detection on smoothed signal
    try:
        # simple smoothing
        from scipy.ndimage import gaussian_filter1d

        smooth = gaussian_filter1d(seg, sigma=max(1, int(fps / 10)))
    except Exception:
        smooth = seg

    # find peaks (prominence relative to amplitude)
    prom = max(amp * 0.2, 0.01)
    peaks, _ = find_peaks(smooth, prominence=prom, distance=max(1, int(fps * 0.8)))
    rep_count = int(len(peaks))

    return {"mean_velocity": mean_v, "std_velocity": std_v, "rep_count": rep_count, "amplitude": amp}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--dataset", default="./data/metadata/dataset.csv")
    p.add_argument("--segments", default="./data/metadata/segments.json")
    p.add_argument("--out", default="./data/metadata/features.csv")
    args = p.parse_args()

    data_root = Path(args.data_dir)

    # Load segments metadata to map segment_id -> video/start/end
    seg_map = {}
    seg_file = data_root / "metadata" / Path(args.segments).name
    if seg_file.exists():
        try:
            with open(seg_file, "r") as f:
                segs = json.load(f)
            for s in segs:
                seg_map[s.get("segment_id")] = s
        except Exception:
            seg_map = {}

    # Load dataset CSV (if exists) as guidance for labels
    df = None
    ds_file = data_root / "metadata" / Path(args.dataset).name
    if ds_file.exists():
        df = pd.read_csv(ds_file)

    features = []

    # iterate over seg_map if dataset missing
    if df is None:
        items = list(seg_map.values())
    else:
        items = []
        for _, row in df.iterrows():
            sid = row.get("segment_id")
            seg_entry = seg_map.get(sid, {})
            seg_entry = dict(seg_entry)
            seg_entry.setdefault("segment_id", sid)
            seg_entry.setdefault("file_path", row.get("file_path", ""))
            seg_entry.setdefault("score", row.get("score", 0.0))
            seg_entry.setdefault("label", row.get("label", ""))
            items.append(seg_entry)

    for seg in items:
        seg_id = seg.get("segment_id")
        video_id = seg.get("video_id")
        start = seg.get("start")
        end = seg.get("end")
        label = seg.get("label", "")

        # locate annotation
        ann_path = data_root / "annotations" / f"{video_id}.json"
        if not ann_path.exists():
            # skip if no annotation
            continue

        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)
        except Exception:
            continue

        sig_key, sig = select_signal(ann)
        if sig is None:
            # skip
            continue

        fps = float(ann.get("fps", 30.0))

        feats = extract_features_from_segment(sig, fps, start, end)
        features.append({
            "segment_id": seg_id,
            "mean_velocity": feats["mean_velocity"],
            "std_velocity": feats["std_velocity"],
            "rep_count": feats["rep_count"],
            "amplitude": feats["amplitude"],
            "label": label,
        })

    out_path = data_root / "metadata" / Path(args.out).name
    df_out = pd.DataFrame(features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote features for {len(df_out)} segments to {out_path}")


if __name__ == "__main__":
    main()
