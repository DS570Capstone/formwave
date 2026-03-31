#!/usr/bin/env python3
"""Build per-segment annotation JSONs from full-video pose annotations.

For each entry in data/metadata/segments.json this script:
- loads the original pose JSON (data/annotations/{video_id}.json)
- slices trajectories/keypoints between start/end (seconds)
- normalizes trajectories to a fixed length
- writes per-segment JSON to data/{exercise}/{camera}/{split}/annotations/{segment_id}.json

If start/end are invalid, the full signal is used. Ensures arrays are non-empty.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import math
import re
from typing import List
import csv
import logging


PREFERRED_SIGNALS = [
    "arm_trajectory",
    "arm_Trajectory",
    "legs_trajectory",
    "shoulder_trajectory",
    "bar_path_trajectory",
    "trajectory",
    "core_",
]


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(s))


def select_signal(ann: dict, prefer_prefixes=PREFERRED_SIGNALS):
    for key in prefer_prefixes:
        if key in ann and isinstance(ann.get(key), list) and len(ann.get(key)) > 0:
            return key, np.asarray(ann.get(key), dtype=float)
    # fallback: pick first numeric list-valued field
    for k, v in ann.items():
        if isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v[:min(5, len(v))]):
            return k, np.asarray(v, dtype=float)
    return None, None


def find_keypoints(ann: dict):
    # look for a key whose value is a list of lists (per-frame keypoints)
    for k, v in ann.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return k, v
    # not found
    return None, []


def normalize_array(arr: np.ndarray, target_len: int) -> List[float]:
    if arr is None or len(arr) == 0:
        return [0.0] * target_len
    arr = np.asarray(arr, dtype=float)
    if len(arr) == target_len:
        return arr.tolist()
    # use linear interpolation over index
    x_old = np.linspace(0.0, 1.0, num=len(arr))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    try:
        arr_new = np.interp(x_new, x_old, arr)
        return arr_new.tolist()
    except Exception:
        # fallback: pad or trim
        if len(arr) < target_len:
            pad = [float(arr[-1])] * (target_len - len(arr))
            return np.concatenate([arr, pad]).tolist()
        else:
            return arr[:target_len].tolist()


def slice_signal(sig: np.ndarray, fps: float, start: float, end: float):
    if sig is None:
        return np.array([])
    n = len(sig)
    if fps is None or fps <= 0:
        return sig
    if start is None or end is None or end <= start:
        return sig
    s_idx = max(0, int(round(start * fps)))
    e_idx = min(n, int(round(end * fps)))
    if e_idx <= s_idx:
        return sig
    return sig[s_idx:e_idx]


def slice_keypoints(kps: list, fps: float, start: float, end: float, target_len: int):
    if not kps:
        return []
    n = len(kps)
    if fps is None or fps <= 0 or start is None or end is None or end <= start:
        # normalize length
        if n == target_len:
            return kps
        # resample frames by interpolation of indices
        idx_old = np.linspace(0, n - 1, num=n)
        idx_new = np.linspace(0, n - 1, num=target_len)
        new_frames = []
        for i in idx_new:
            lower = int(math.floor(i))
            upper = int(math.ceil(i))
            if lower == upper:
                new_frames.append(kps[lower])
            else:
                # linear blend between frames
                f = i - lower
                a = np.asarray(kps[lower], dtype=float)
                b = np.asarray(kps[upper], dtype=float)
                try:
                    blended = (1 - f) * a + f * b
                    new_frames.append(blended.tolist())
                except Exception:
                    new_frames.append(kps[lower])
        return new_frames

    s_idx = max(0, int(round(start * fps)))
    e_idx = min(n, int(round(end * fps)))
    if e_idx <= s_idx:
        slice_frames = kps
    else:
        slice_frames = kps[s_idx:e_idx]

    # resample slice_frames to target_len
    n2 = len(slice_frames)
    if n2 == 0:
        return []
    if n2 == target_len:
        return slice_frames
    idx_old = np.linspace(0, n2 - 1, num=n2)
    idx_new = np.linspace(0, n2 - 1, num=target_len)
    new_frames = []
    for i in idx_new:
        lower = int(math.floor(i))
        upper = int(math.ceil(i))
        if lower == upper:
            new_frames.append(slice_frames[lower])
        else:
            f = i - lower
            a = np.asarray(slice_frames[lower], dtype=float)
            b = np.asarray(slice_frames[upper], dtype=float)
            try:
                blended = (1 - f) * a + f * b
                new_frames.append(blended.tolist())
            except Exception:
                new_frames.append(slice_frames[lower])
    return new_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--segments", default="./data/metadata/segments.json")
    p.add_argument("--annotations_dir", default="./data/annotations")
    p.add_argument("--out_root", default="./data")
    p.add_argument("--target_len", type=int, default=100)
    args = p.parse_args()

    data_root = Path(args.data_dir)
    seg_path = data_root / "metadata" / Path(args.segments).name
    # resolve annotations dir relative to data_root when appropriate
    ann_dir = Path(args.annotations_dir)
    # Prefer annotations path under data_root when possible
    if not ann_dir.is_absolute():
        candidate = data_root / ann_dir
        if candidate.exists():
            ann_dir = candidate
        elif ann_dir.exists():
            ann_dir = ann_dir
        else:
            # fallback to data_root/annotations
            ann_dir = data_root / "annotations"

    # default out_root: if user left default './data' or 'data', write under the provided data_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute() and out_root in (Path("./data"), Path("data")):
        out_root = data_root
    target_len = int(args.target_len)

    if not seg_path.exists():
        print("Segments metadata not found:", seg_path)
        return

    try:
        with open(seg_path, "r") as f:
            segs = json.load(f)
    except Exception as e:
        print("Failed to read segments.json:", e)
        return

    created = 0
    # Build set of accepted curated segments from metadata (CSV preferred)
    accepted_segments = set()
    curated_meta_dir = data_root / "curated_pushups" / "metadata"
    csv_meta = curated_meta_dir / "segments.csv"
    json_meta = curated_meta_dir / "segments.json"
    if csv_meta.exists():
        try:
            with open(csv_meta, "r") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    try:
                        if row.get("status", "").lower() == "accepted":
                            sid = row.get("segment_id") or row.get("segment_id")
                            if sid:
                                accepted_segments.add(str(sid))
                    except Exception:
                        continue
            logging.info("Loaded %d accepted segments from %s", len(accepted_segments), csv_meta)
        except Exception:
            logging.exception("Failed to read curated CSV metadata %s", csv_meta)
    elif json_meta.exists():
        try:
            with open(json_meta, "r") as jf:
                entries = json.load(jf)
                for e in entries:
                    try:
                        if str(e.get("status", "")).lower() == "accepted":
                            sid = e.get("segment_id")
                            if sid:
                                accepted_segments.add(str(sid))
                    except Exception:
                        continue
            logging.info("Loaded %d accepted segments from %s", len(accepted_segments), json_meta)
        except Exception:
            logging.exception("Failed to read curated JSON metadata %s", json_meta)
    else:
        logging.info("No curated metadata found at %s; falling back to status_from_path behavior", curated_meta_dir)
    for s in segs:
        seg_id = s.get("segment_id")
        video_id = s.get("video_id")
        start = float(s.get("start", 0.0)) if s.get("start") is not None else None
        end = float(s.get("end", 0.0)) if s.get("end") is not None else None
        file_path = s.get("file_path", "")

        # If curated metadata exists, only process accepted segments
        if accepted_segments:
            if str(seg_id) not in accepted_segments:
                logging.info("Skipping segment %s — not marked accepted in curated metadata", seg_id)
                continue

        ann_path = ann_dir / f"{video_id}.json"
        if not ann_path.exists():
            # skip
            continue

        try:
            with open(ann_path, "r") as f:
                ann = json.load(f)
        except Exception:
            continue

        fps = float(ann.get("fps", 30.0))

        # select trajectories
        arm_key, arm_sig = select_signal(ann, ["arm_Trajectory", "arm_trajectory"])
        legs_key, legs_sig = select_signal(ann, ["legs_trajectory"])
        core_key, core_sig = select_signal(ann, ["core_"])

        # fallbacks
        if arm_sig is None:
            arm_key, arm_sig = select_signal(ann)
        if legs_sig is None:
            legs_key, legs_sig = select_signal(ann)
        if core_sig is None:
            core_key, core_sig = select_signal(ann)

        # slice signals
        arm_slice = slice_signal(arm_sig, fps, start, end) if arm_sig is not None else np.array([])
        legs_slice = slice_signal(legs_sig, fps, start, end) if legs_sig is not None else np.array([])
        core_slice = slice_signal(core_sig, fps, start, end) if core_sig is not None else np.array([])

        # normalize length
        arm_norm = normalize_array(arm_slice, target_len)
        legs_norm = normalize_array(legs_slice, target_len)
        core_norm = normalize_array(core_slice, target_len)

        # keypoints
        kps_key, kps = find_keypoints(ann)
        kps_slice = slice_keypoints(kps, fps, start, end, target_len) if kps else []

        exercise = sanitize_name(ann.get("exercise", "unknown"))
        camera = sanitize_name(ann.get("CAMERA_POSITION", ann.get("camera", "unknown")))

        # derive split/status from file_path if possible — support legacy processed
        # folders and new curated_pushups status directories
        split = "train"
        status_from_path = None
        if file_path and isinstance(file_path, str):
            if "/processed/" in file_path:
                parts = file_path.split("/")
                # find 'processed' and take next part as split
                try:
                    i = parts.index("processed")
                    if i + 1 < len(parts):
                        split = parts[i + 1]
                except ValueError:
                    pass
            elif "/curated_pushups/" in file_path:
                parts = file_path.split("/")
                try:
                    i = parts.index("curated_pushups")
                    if i + 1 < len(parts):
                        status_dir = parts[i + 1]
                        if "accepted" in status_dir:
                            status_from_path = "accepted"
                        elif "review" in status_dir:
                            status_from_path = "review"
                        elif "reject" in status_dir:
                            status_from_path = "rejected"
                        else:
                            status_from_path = status_dir
                except ValueError:
                    pass

        # If curated metadata exists, always write accepted per-segment JSONs to
        # the curated accepted_annotations folder. Otherwise fall back to legacy layout.
        if accepted_segments:
            out_dir = data_root / "curated_pushups" / "accepted_annotations"
        else:
            if status_from_path == "accepted":
                out_dir = out_root / "curated_pushups" / "accepted_annotations"
            else:
                out_dir = out_root / exercise / camera / split / "annotations"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_obj = {
            "video_id": seg_id,
            "exercise": ann.get("exercise", "unknown"),
            "CAMERA_POSITION": ann.get("CAMERA_POSITION", ann.get("camera", "unknown")),
            "arm_trajectory": arm_norm,
            "legs_trajectory": legs_norm,
            "core_trajectory": core_norm,
            "keypoints": kps_slice,
            "error_rate": [],
            "expert": bool(ann.get("expert", False)),
        }

        out_path = out_dir / f"{seg_id}.json"
        try:
            with open(out_path, "w") as f:
                json.dump(out_obj, f, indent=2)
            created += 1
        except Exception:
            continue

    print(f"Wrote {created} per-segment annotation JSONs to {out_root}")


if __name__ == "__main__":
    main()
