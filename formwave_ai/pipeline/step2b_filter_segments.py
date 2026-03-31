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
import csv
from typing import List

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

# Ensure top-level package imports work (match other pipeline steps)
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from modules import wave_filter
from modules import signal_similarity
from pipeline.tracking import load_tracking, save_tracking, mark_clip_status
from pipeline.pushup_quality_filter import score_pushup_segment


SPLIT_THRESH = (70, 85)  # <70 train, <85 val, else test
DEFAULT_SCORE_THRESH = 0.70
MIN_SEGMENT_DURATION = 3.0  # seconds
MERGE_GAP = 2.0  # seconds
DEFAULT_TOP_K = 5


# NOTE: deterministic train/val/test split removed. This pipeline uses a
# status-based workflow (accepted / review / rejected) for curated push-up
# segments. Legacy split generation is intentionally disabled.


def count_reps(signal):
    import numpy as np

    signal = np.array(signal)

    # normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    # detect peaks
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    return len(peaks)


def is_valid_wave(signal):
    import numpy as np

    signal = np.array(signal)

    if len(signal) < 30:
        return False

    # motion range
    if np.max(signal) - np.min(signal) < 0.4:
        return False

    # flat clipping
    flat_ratio = np.mean((signal > 0.95) | (signal < -0.95))
    if flat_ratio > 0.3:
        return False

    # spikes
    diffs = np.abs(np.diff(signal))
    if np.max(diffs) > 1.5:
        return False

    # unique values
    if len(set(signal)) < 15:
        return False

    # 🚀 NEW: repetition logic
    reps = count_reps(signal)

    if reps < 1:
        return False

    if reps > 20:
        return False

    return True


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


def process_directory(data_root: Path, score_thresh: float = DEFAULT_SCORE_THRESH, top_k: int = DEFAULT_TOP_K,
                      reference_video_ids: List[str] = None, similarity_thresh: float = 0.75,
                      use_dtw: bool = False, signal_key: str = "core_", target_len: int = 100):
    logger = logging.getLogger(__name__)
    tracking = load_tracking(data_root)

    ann_paths = sorted(data_root.rglob("annotations/*.json"))
    if not ann_paths:
        logger.error("No annotation JSONs found under %s", data_root)
        return

    created = 0
    accepted_count = 0
    review_count = 0
    rejected_count = 0

    # Load reference signals once (support comma-separated ids)
    references = []
    if reference_video_ids:
        if isinstance(reference_video_ids, str):
            ref_ids = [r.strip() for r in reference_video_ids.split(",") if r.strip()]
        elif isinstance(reference_video_ids, list):
            ref_ids = reference_video_ids
        else:
            ref_ids = []

        for rid in ref_ids:
            ref_path = data_root / "annotations" / f"{rid}.json"
            try:
                ref_ann = json.load(open(ref_path))
            except Exception:
                logger.warning("Reference annotation not found or unreadable: %s", ref_path)
                continue
            ref_sig = signal_similarity.extract_signal(ref_ann, key=signal_key)
            if ref_sig is None:
                logger.warning("Reference signal key '%s' not found in %s", signal_key, ref_path)
                continue
            ref_resampled = signal_similarity.resample_signal(ref_sig, target_len=target_len)
            if ref_resampled is None:
                logger.warning("Failed to resample reference signal for %s", ref_path)
                continue
            references.append(ref_resampled)

    if references:
        logger.info("Loaded %d reference signals (key=%s target_len=%d)", len(references), signal_key, target_len)

    for ann_path in ann_paths:
        try:
            ann = json.load(open(ann_path))
        except Exception as e:
            logger.warning("Failed to load %s: %s", ann_path, e)
            continue

        clip_id = ann.get("video_id") or ann_path.stem
        exercise = ann.get("exercise", "unknown")
        # determine whether this clip should be treated as a push-up candidate.
        exercise_lower = str(exercise).lower() if exercise else ""
        has_push_signals = any(k in ann and ann.get(k) is not None for k in ("arm_trajectory", "arm_Trajectory", "trajectory", "core_", "legs_trajectory"))
        is_push_candidate = ("push" in exercise_lower) or (exercise_lower in ("unknown", "unspecified", "other") and (has_push_signals or ann.get("keypoints") or ann.get("confidence")))
        # No train/val/test split generation. We'll assign per-segment status
        # (accepted / review / rejected) later and route push-up segments into
        # data/curated_pushups/<status>_segments
        split_folder = "ungrouped"

        # select the requested signal key explicitly (honor --signal_key)
        sig = signal_similarity.extract_signal(ann, key=signal_key)
        if sig is None:
            logger.info("[VIDEO SKIPPED] %s missing signal key %s", clip_id, signal_key)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="no_signal")
            save_tracking(tracking, data_root)
            continue
        sig_key = signal_key

        fps = float(ann.get("fps", 30.0))
        sig_arr = np.asarray(sig, dtype=float)

        logger.info("Processing %s — signal=%s fps=%.2f", clip_id, sig_key, fps)

        # Rep-based segmentation: smooth, find peaks, convert successive peaks -> rep-to-rep segments
        try:
            smooth = gaussian_filter1d(sig_arr, sigma=max(1, int(round(fps / 4.0))))
            amp = float(np.ptp(smooth)) if len(smooth) > 0 else 0.0
            prom = max(amp * 0.02, 0.01)
            distance = max(1, int(round(fps * 0.8)))
            peaks, _ = find_peaks(smooth, prominence=prom, distance=distance)
            if len(peaks) >= 2:
                segments = []
                for i in range(len(peaks) - 1):
                    s_idx = peaks[i]
                    e_idx = peaks[i + 1]
                    segments.append((float(s_idx) / float(fps), float(e_idx) / float(fps)))
            else:
                # fallback to sliding-window detection when peaks are insufficient
                segments = wave_filter.detect_valid_segments(sig_arr, int(round(fps)))
        except Exception:
            segments = wave_filter.detect_valid_segments(sig_arr, int(round(fps)))
        if not segments:
            logger.info("[VIDEO SKIPPED] %s no valid motion detected", clip_id)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="no_valid_segments")
            save_tracking(tracking, data_root)
            continue

        # NOTE: aggressive merging caused whole-video segments; disable merging and use detected segments as-is
        merged = sorted(segments, key=lambda x: x[0])

        # Score and filter segments
        candidates = []  # list of (start, end, score)

        # Prepare paths and tracking for generated segments
        video_path = data_root / "raw" / "videos" / f"{clip_id}.mp4"
        if not video_path.exists():
            logger.warning("Original video not found for %s — expected: %s", clip_id, video_path)
            mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="video_missing")
            save_tracking(tracking, data_root)
            continue

        # metadata file under data/metadata (legacy) — we'll also write curated metadata for push-ups
        meta_dir = data_root / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        seg_meta_path = meta_dir / "segments.json"
        if seg_meta_path.exists():
            try:
                with open(seg_meta_path, "r") as mf:
                    existing = json.load(mf)
            except Exception:
                existing = []
        else:
            existing = []

        clip_entry = tracking.setdefault("clips", {}).setdefault(clip_id, {})
        gen_list = clip_entry.setdefault("generated_segments", [])
        # Pre-resolve any existing generated paths for robust comparisons
        gen_resolved = set()
        for p in gen_list:
            try:
                gen_resolved.add(str(Path(p).resolve()))
            except Exception:
                gen_resolved.add(str(p))

        seg_idx = 0

        for s, e in merged:
            duration = e - s
            if duration < MIN_SEGMENT_DURATION:
                logger.info("[SEGMENT DROPPED SHORT] %s %.2f-%.2f dur=%.2f", clip_id, s, e, duration)
                continue

            s_frame = max(0, int(round(s * fps)))
            e_frame = min(len(sig_arr), int(round(e * fps)))
            sig_segment = sig_arr[s_frame:e_frame]
            # Optional smoothing before validation
            try:
                sig_segment = savgol_filter(sig_segment, 11, 3)
            except Exception:
                pass

            # HARD FILTER: discard clearly invalid waves
            if not is_valid_wave(sig_segment):
                logger.info("[SEGMENT DROPPED INVALID_WAVE] %s %.2f-%.2f", clip_id, s, e)
                continue

            score = wave_filter.score_segment(sig_segment)

            if score < score_thresh:
                logger.info("[SEGMENT DROPPED LOW SCORE] %s %.2f-%.2f score=%.3f", clip_id, s, e, score)
                continue

            similarity = None

            # compute similarity if references provided
            if references:
                ann_sig = signal_similarity.extract_signal(ann, key=signal_key)
                if ann_sig is None:
                    similarity = 0.0
                else:
                    s_frame = max(0, int(round(s * fps)))
                    e_frame = min(len(ann_sig), int(round(e * fps)))
                    seg_sig = ann_sig[s_frame:e_frame]
                    if seg_sig is None or len(seg_sig) < 2:
                        similarity = 0.0
                    else:
                        seg_res = signal_similarity.resample_signal(seg_sig, target_len=target_len)
                        if seg_res is None:
                            similarity = 0.0
                        else:
                            sims = []
                            for ref in references:
                                try:
                                    if use_dtw:
                                        dist = signal_similarity.compute_dtw(ref, seg_res)
                                        sim = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
                                    else:
                                        sim = signal_similarity.compute_correlation(ref, seg_res)
                                    sims.append(float(sim))
                                except Exception:
                                    sims.append(0.0)
                            similarity = float(sum(sims) / len(sims)) if sims else 0.0

            # Decide segment status (accepted / review / rejected)
            # For push-up exercises, prefer model-based quality filter
            status = None
            push_reasons = []
            push_score = None
            if is_push_candidate:
                # prepare a light-weight per-segment annotation slice for the filter
                seg_kps = None
                seg_conf = None
                try:
                    kps = ann.get("keypoints")
                    conf = ann.get("confidence") or ann.get("conf")
                    if kps is not None:
                        seg_kps = kps[s_frame:e_frame]
                    if conf is not None:
                        seg_conf = conf[s_frame:e_frame]
                except Exception:
                    seg_kps = None
                    seg_conf = None

                seg_data = {"keypoints": seg_kps, "confidence": seg_conf, "fps": fps}
                # include trajectory candidates if present (slice them)
                for cand in ("arm_trajectory", "arm_Trajectory", "trajectory", "core_", "legs_trajectory"):
                    try:
                        if cand in ann and ann.get(cand) is not None:
                            seg_data[cand] = ann.get(cand)[s_frame:e_frame]
                    except Exception:
                        pass

                try:
                    pf_res = score_pushup_segment(seg_data)
                    status = pf_res.get("status") or "review"
                    push_score = pf_res.get("score")
                    push_reasons = pf_res.get("reasons") or []
                except Exception:
                    logger.exception("Push-up quality filter failed for %s seg %.2f-%.2f", clip_id, s, e)
                    status = "review"
            else:
                # Rules for non-push exercises (legacy fallback)
                if similarity is not None and references:
                    if similarity >= similarity_thresh:
                        status = "accepted"
                    elif score >= score_thresh:
                        status = "review"
                    else:
                        status = "rejected"
                else:
                    status = "review" if score >= score_thresh else "rejected"

            # Decide output base path depending on exercise and status
            if is_push_candidate:
                curated_base = data_root / "curated_pushups"
                if status == "accepted":
                    out_base = curated_base / "accepted_segments"
                elif status == "review":
                    out_base = curated_base / "review_segments"
                else:
                    out_base = curated_base / "rejected_segments"
            else:
                # Non-push exercises: keep legacy processed folder but without splits
                out_base = data_root / "processed" / "ungrouped"

            out_base.mkdir(parents=True, exist_ok=True)

            # build per-segment filename and path
            seg_idx += 1
            out_name = f"{clip_id}_seg{seg_idx:02d}.mp4"
            out_path = (out_base / out_name).resolve()

            # Log decision clearly
            logger.info("[DECISION] video=%s seg=%s score=%.3f similarity=%s status=%s", clip_id, out_name, score, (f"{similarity:.3f}" if similarity is not None else "NA"), status)
            print(f"[SEGMENT] video={clip_id} seg={out_name} score={score:.3f} similarity={(f'{similarity:.3f}' if similarity is not None else 'NA')} status={status}")

            # create metadata entry for this candidate
            segment_id = out_name
            segment_meta = {
                "video_id": clip_id,
                "segment_id": segment_id,
                "start": float(s),
                "end": float(e),
                "score": float(score),
                "similarity": (float(similarity) if similarity is not None else None),
                "status": status,
                "file_path": "",
            }
            # attach push-up filter details when available
            if push_score is not None:
                segment_meta["pushup_score"] = float(push_score)
            if push_reasons:
                segment_meta["reasons"] = list(push_reasons)

            # skip if already recorded or physically exists
            if str(out_path) in gen_resolved or out_path.exists():
                logger.info("[SEGMENT EXISTS] %s already exists", out_path)
                segment_meta["file_path"] = str(out_path)
                existing.append(segment_meta)
                gen_list.append(str(out_path))
                gen_resolved.add(str(out_path))
                continue

            # PRE-CUT GUARD: for push-up exercises, do not cut or write clip files when
            # the pushup_quality_filter explicitly returns 'rejected'. Still record
            # metadata so rejected candidates are tracked, but avoid expensive I/O.
            if "push" in str(exercise).lower() and status == "rejected":
                logger.info("[PRE-CUT REJECTED] %s %.2f-%.2f status=%s — skipping cut and file write", clip_id, s, e, status)
                rejected_count += 1
                # Persist a single-row CSV metadata entry for curated push-ups (append-safe)
                try:
                    curated_meta_dir = data_root / "curated_pushups" / "metadata"
                    curated_meta_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = curated_meta_dir / "segments.csv"
                    fieldnames = [
                        "segment_id", "source_video_id", "start_time", "end_time",
                        "status", "score", "avg_confidence", "valid_frame_ratio",
                        "elbow_angle_min", "elbow_angle_max", "elbow_angle_range",
                        "body_alignment_score", "horizontal_body_score", "motion_amplitude",
                        "reasons"
                    ]
                    write_header = not csv_path.exists()
                    with open(csv_path, "a", newline="") as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()
                        metrics = {}
                        try:
                            if push_score is not None:
                                metrics = pf_res.get("metrics", {}) if 'pf_res' in locals() else {}
                        except Exception:
                            metrics = {}

                        row = {
                            "segment_id": segment_id,
                            "source_video_id": clip_id,
                            "start_time": float(s),
                            "end_time": float(e),
                            "status": status,
                            "score": (push_score if push_score is not None else float(score)),
                            "avg_confidence": metrics.get("avg_required_joint_conf", ""),
                            "valid_frame_ratio": metrics.get("valid_frame_ratio", ""),
                            "elbow_angle_min": metrics.get("elbow_angle_min", ""),
                            "elbow_angle_max": metrics.get("elbow_angle_max", ""),
                            "elbow_angle_range": metrics.get("elbow_angle_range", ""),
                            "body_alignment_score": metrics.get("body_alignment_score", ""),
                            "horizontal_body_score": metrics.get("horizontal_stability", ""),
                            "motion_amplitude": metrics.get("motion_amplitude", ""),
                            "reasons": ";".join(segment_meta.get("reasons", [])) if segment_meta.get("reasons") else "",
                        }
                        writer.writerow(row)
                except Exception:
                    logger.exception("Failed to append curated CSV metadata for %s", segment_id)

                # Append metadata entry and persist curated JSON list
                existing.append(segment_meta)
                try:
                    curated_meta_dir = data_root / "curated_pushups" / "metadata"
                    curated_meta_dir.mkdir(parents=True, exist_ok=True)
                    curated_meta_path = curated_meta_dir / "segments.json"
                    if curated_meta_path.exists():
                        try:
                            with open(curated_meta_path, "r") as cf:
                                curated_existing = json.load(cf)
                        except Exception:
                            curated_existing = []
                    else:
                        curated_existing = []

                    curated_existing.append(segment_meta)
                    with open(curated_meta_path, "w") as cf:
                        json.dump(curated_existing, cf, indent=2)
                except Exception:
                    logger.exception("Failed to write curated metadata for %s", clip_id)

                continue

            ok = wave_filter.cut_video_segment(video_path, out_path, s, e)
            if ok:
                gen_list.append(str(out_path))
                gen_resolved.add(str(out_path))
                logger.info("[SEGMENT SAVED] %s start=%.2f end=%.2f score=%.3f status=%s", clip_id, s, e, score, status)
                created += 1
                # increment counters for reporting
                if status == "accepted":
                    accepted_count += 1
                elif status == "review":
                    review_count += 1
                elif status == "rejected":
                    rejected_count += 1
                segment_meta["file_path"] = str(out_path)
                # For accepted push-up segments, also save the clip annotation
                try:
                    if status == "accepted" and is_push_candidate:
                        ann_out_dir = data_root / "curated_pushups" / "accepted_annotations"
                        ann_out_dir.mkdir(parents=True, exist_ok=True)
                        ann_out_path = ann_out_dir / f"{segment_id}.json"
                        # Save a small record with metadata and the sliced annotation
                        ann_record = dict(ann) if isinstance(ann, dict) else {}
                        # attach sliced keypoints/confidence when available
                        try:
                            if seg_kps is not None:
                                ann_record["keypoints"] = seg_kps if isinstance(seg_kps, list) else (seg_kps.tolist() if hasattr(seg_kps, 'tolist') else list(seg_kps))
                        except Exception:
                            ann_record["keypoints"] = []
                        try:
                            if seg_conf is not None:
                                ann_record["confidence"] = seg_conf if isinstance(seg_conf, list) else (seg_conf.tolist() if hasattr(seg_conf, 'tolist') else list(seg_conf))
                        except Exception:
                            ann_record["confidence"] = []

                        with open(ann_out_path, "w") as af:
                            json.dump({"segment_meta": segment_meta, "annotation": ann_record}, af, indent=2)
                except Exception:
                    logger.exception("Failed to write accepted annotation for %s", segment_id)
                # Persist a single-row CSV metadata entry for curated push-ups (append-safe)
                try:
                    if is_push_candidate:
                        curated_meta_dir = data_root / "curated_pushups" / "metadata"
                        curated_meta_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = curated_meta_dir / "segments.csv"
                        fieldnames = [
                            "segment_id", "source_video_id", "start_time", "end_time",
                            "status", "score", "avg_confidence", "valid_frame_ratio",
                            "elbow_angle_min", "elbow_angle_max", "elbow_angle_range",
                            "body_alignment_score", "horizontal_body_score", "motion_amplitude",
                            "reasons"
                        ]
                        write_header = not csv_path.exists()
                        with open(csv_path, "a", newline="") as cf:
                            writer = csv.DictWriter(cf, fieldnames=fieldnames)
                            if write_header:
                                writer.writeheader()
                            metrics = {}
                            try:
                                # metrics come from pushup filter result when available
                                if push_score is not None:
                                    metrics = pf_res.get("metrics", {}) if 'pf_res' in locals() else {}
                            except Exception:
                                metrics = {}

                            row = {
                                "segment_id": segment_id,
                                "source_video_id": clip_id,
                                "start_time": float(s),
                                "end_time": float(e),
                                "status": status,
                                "score": (push_score if push_score is not None else float(score)),
                                "avg_confidence": metrics.get("avg_required_joint_conf", ""),
                                "valid_frame_ratio": metrics.get("valid_frame_ratio", ""),
                                "elbow_angle_min": metrics.get("elbow_angle_min", ""),
                                "elbow_angle_max": metrics.get("elbow_angle_max", ""),
                                "elbow_angle_range": metrics.get("elbow_angle_range", ""),
                                "body_alignment_score": metrics.get("body_alignment_score", ""),
                                "horizontal_body_score": metrics.get("horizontal_stability", ""),
                                "motion_amplitude": metrics.get("motion_amplitude", ""),
                                "reasons": ";".join(segment_meta.get("reasons", [])) if segment_meta.get("reasons") else "",
                            }
                            writer.writerow(row)
                except Exception:
                    logger.exception("Failed to append curated CSV metadata for %s", segment_id)
            else:
                logger.warning("[SEGMENT FAILED] %s start=%.2f end=%.2f", clip_id, s, e)

            # append metadata entry (kept/accepted/review/rejected)
            existing.append(segment_meta)

            # Also persist into curated_pushups metadata for push-up exercises
            try:
                if is_push_candidate:
                    curated_meta_dir = data_root / "curated_pushups" / "metadata"
                    curated_meta_dir.mkdir(parents=True, exist_ok=True)
                    curated_meta_path = curated_meta_dir / "segments.json"
                    if curated_meta_path.exists():
                        try:
                            with open(curated_meta_path, "r") as cf:
                                curated_existing = json.load(cf)
                        except Exception:
                            curated_existing = []
                    else:
                        curated_existing = []

                    curated_existing.append(segment_meta)
                    with open(curated_meta_path, "w") as cf:
                        json.dump(curated_existing, cf, indent=2)
            except Exception:
                logger.exception("Failed to write curated metadata for %s", clip_id)

        # atomic write segments metadata
        try:
            tmp = seg_meta_path.with_suffix(".tmp")
            with open(tmp, "w") as mf:
                json.dump(existing, mf, indent=2)
            tmp.replace(seg_meta_path)
        except Exception:
            logger.exception("Failed to write segments metadata for clip %s", clip_id)

        # persist generated list and mark clip
        clip_entry["generated_segments"] = gen_list
        if gen_list:
            mark_clip_status(tracking, clip_id, processed=True, status="clips_generated")
        else:
            mark_clip_status(tracking, clip_id, processed=False, status="no_segments")

        save_tracking(tracking, data_root)

    logging.info("Segment extraction complete — created=%d", created)
    # summary of curated push-up decisions
    logging.info("Curated push-up summary — accepted=%d review=%d rejected=%d", accepted_count, review_count, rejected_count)
    print(f"Curated push-up summary — accepted={accepted_count} review={review_count} rejected={rejected_count}")


def parse_args():
    p = argparse.ArgumentParser(description="Step 2b: Filter segments from pose signals")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--score_thresh", type=float, default=DEFAULT_SCORE_THRESH,
                   help=f"Minimum score threshold (default: {DEFAULT_SCORE_THRESH})")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                   help=f"Maximum segments to keep per video (default: {DEFAULT_TOP_K})")
    p.add_argument("--verbose", action="store_true")
    # reference-based similarity options
    p.add_argument("--reference_video_id", default=None,
                   help="Reference video id (or comma-separated ids) to compare segments against")
    p.add_argument("--similarity_thresh", type=float, default=0.75,
                   help="Minimum similarity to reference to keep segment (default: 0.75)")
    p.add_argument("--use_dtw", action="store_true",
                   help="Use DTW distance converted to similarity instead of correlation")
    p.add_argument("--signal_key", default="core_",
                   help="Signal key to use from annotations (default: core_)")
    p.add_argument("--target_len", type=int, default=100,
                   help="Target length to resample signals to (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")
    data_root = Path(args.data_dir)
    process_directory(
        data_root,
        score_thresh=args.score_thresh,
        top_k=args.top_k,
        reference_video_ids=args.reference_video_id,
        similarity_thresh=args.similarity_thresh,
        use_dtw=args.use_dtw,
        signal_key=args.signal_key,
        target_len=args.target_len,
    )


if __name__ == "__main__":
    main()
