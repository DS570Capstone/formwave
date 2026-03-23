"""Segment detection utilities (refactored from data/b_detect_segment.py).

Provides a reusable detect_best_segment() with pluggable signal and scoring.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks
from typing import Callable, Iterable, List, Tuple, Dict, Optional

mp_pose = mp.solutions.pose


def default_scoring(reps: int, amplitude: float, confidence: float) -> float:
    """Default score: reps * amplitude * confidence."""
    return reps * amplitude * confidence


def _extract_signal_from_landmarks(landmarks, indices: Iterable[int], min_visibility: float = 0.5):
    vals = []
    confs = []
    for i in indices:
        lm = landmarks[i]
        if getattr(lm, "visibility", 0) >= min_visibility:
            vals.append(lm.y)
            confs.append(getattr(lm, "visibility", 1.0))
        else:
            vals.append(np.nan)
            confs.append(0.0)

    # average ignoring nans
    vals = np.array(vals)
    confs = np.array(confs)
    valid = ~np.isnan(vals)
    if valid.any():
        return float(np.nanmean(vals)), float(np.nanmean(confs))
    else:
        return float(np.nan), float(0.0)


def detect_best_segment(
    video_path: Path,
    window_sec: int = 20,
    frame_skip: int = 4,
    max_minutes: int = 5,
    keypoint_indices: Iterable[int] = (9, 10),
    min_visibility: float = 0.5,
    scoring_fn: Callable[[int, float, float], float] = None,
) -> Tuple[float, float, int]:
    """Detect the best segment in a video based on a 1D signal extracted
    from pose landmarks.

    Returns (start_sec, end_sec, score)
    """
    # scoring_fn remains a legacy override; if provided we call it.
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    effective_fps = fps / frame_skip
    max_frames = int(fps * 60 * max_minutes)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    signal = []
    conf_signal = []
    # optional per-side signals for symmetry
    left_sig = []
    right_sig = []
    frame_idx = 0

    # accept either a pair of indices for symmetry or a list of arbitrary indices
    try:
        kp_list = list(keypoint_indices)
    except Exception:
        kp_list = [keypoint_indices]

    use_symmetry = len(kp_list) >= 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx > max_frames:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        frame_idx += 1

        frame = cv2.resize(frame, (256, 256))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # compute per-index values
            vals = []
            confs = []
            for i in kp_list:
                lm = landmarks[i]
                if getattr(lm, "visibility", 0) >= min_visibility:
                    vals.append(lm.y)
                    confs.append(getattr(lm, "visibility", 1.0))
                else:
                    vals.append(np.nan)
                    confs.append(0.0)

            avg_val = float(np.nanmean(vals)) if any(~np.isnan(vals)) else float(np.nan)
            avg_conf = float(np.nanmean(confs)) if len(confs) > 0 else 0.0
            signal.append(avg_val)
            conf_signal.append(avg_conf)

            if use_symmetry:
                left_sig.append(vals[0] if len(vals) > 0 else np.nan)
                right_sig.append(vals[1] if len(vals) > 1 else np.nan)
        else:
            signal.append(np.nan)
            conf_signal.append(0.0)
            if use_symmetry:
                left_sig.append(np.nan)
                right_sig.append(np.nan)

    cap.release()
    pose.close()

    signal = np.array(signal)
    conf_signal = np.array(conf_signal)

    valid = ~np.isnan(signal)
    signal = signal[valid]
    conf_signal = conf_signal[valid]

    if use_symmetry:
        left_sig = np.array(left_sig)[valid]
        right_sig = np.array(right_sig)[valid]
    else:
        left_sig = right_sig = None

    if len(signal) < effective_fps * window_sec:
        return 0.0, min(window_sec, len(signal) / effective_fps), 0

    # Pre-smooth for noise estimation but keep original for residuals
    try:
        smooth_signal = savgol_filter(signal, 9, 2)
    except Exception:
        smooth_signal = signal.copy()

    residual = signal - smooth_signal

    window_size = int(window_sec * effective_fps)
    step = int(effective_fps)

    best_score = 0.0
    best_start = 0

    eps = 1e-8

    for start in range(0, len(signal) - window_size, step):
        segment = smooth_signal[start : start + window_size]
        orig_segment = signal[start : start + window_size]
        seg_conf = conf_signal[start : start + window_size]

        if len(segment) == 0:
            continue

        # repetitions via peak detection
        peaks, _ = find_peaks(-segment, distance=max(1, int(effective_fps // 2)))
        rep_count = len(peaks)
        amplitude = float(np.nanmax(segment) - np.nanmin(segment)) if not np.isnan(segment).all() else 0.0

        # confidence
        confidence = float(np.nanmean(seg_conf)) if len(seg_conf) > 0 else 0.0

        # Smoothness: inverse scaled variance of derivative (higher -> smoother)
        deriv = np.diff(segment)
        var_deriv = float(np.nanvar(deriv))
        smoothness = 1.0 / (1.0 + (var_deriv / (max(amplitude, eps) ** 2)))

        # Periodicity: FFT peak ratio
        seg_demean = segment - np.nanmean(segment)
        try:
            fft = np.fft.rfft(seg_demean)
            power = np.abs(fft) ** 2
            total_power = power.sum() + eps
            peak_power = power.max()
            psd_peak_ratio = float(peak_power / total_power)
        except Exception:
            psd_peak_ratio = 0.0

        # Autocorrelation consistency (normalized)
        try:
            ac = np.correlate(seg_demean, seg_demean, mode="full")
            ac = ac[ac.size // 2 :]
            ac0 = ac[0] if ac.size > 0 else 0.0
            ac_peak = np.max(ac[1:]) if ac.size > 1 else 0.0
            ac_consistency = float(ac_peak / (ac0 + eps))
        except Exception:
            ac_consistency = 0.0

        periodicity_consistency = psd_peak_ratio * ac_consistency

        # Symmetry: compare left/right signals if available
        if left_sig is not None and right_sig is not None:
            seg_left = left_sig[start : start + window_size]
            seg_right = right_sig[start : start + window_size]
            # ignore NaNs
            valid_lr = ~np.isnan(seg_left) & ~np.isnan(seg_right)
            if valid_lr.any():
                mean_diff = float(np.nanmean(np.abs(seg_left[valid_lr] - seg_right[valid_lr])))
                # normalize by average magnitude
                norm = float((np.nanmean(np.abs(seg_left[valid_lr])) + np.nanmean(np.abs(seg_right[valid_lr]))) / 2.0 + eps)
                symmetry = max(0.0, 1.0 - (mean_diff / norm))
            else:
                symmetry = 1.0
        else:
            symmetry = 1.0

        # Noise level: residual energy vs amplitude
        seg_residual = residual[start : start + window_size]
        noise = float(min(1.0, np.nanstd(seg_residual) / (max(amplitude, eps))))

        # Combine periodicity as requested, incorporating smoothness and symmetry
        periodicity = float(periodicity_consistency * smoothness * symmetry)

        # Final score formula: reps * amplitude * periodicity * (1 - noise)
        if scoring_fn is not None:
            score = scoring_fn(rep_count, amplitude, confidence)
        else:
            score = rep_count * amplitude * periodicity * (1.0 - noise)

        if rep_count >= 3 and score > best_score:
            best_score = score
            best_start = start

    start_time = best_start / effective_fps
    end_time = (best_start + window_size) / effective_fps

    return float(start_time), float(end_time), int(best_score)


def detect_directory(lowres_dir: Path, output_file: Path, **kwargs):
    """Run detection over all mp4 files in lowres_dir and write JSON to output_file."""
    results = {}
    lowres_dir = Path(lowres_dir)
    for video in sorted(lowres_dir.glob("*.mp4")):
        print(f"🔎 Detecting segment: {video.name}")
        start, end, score = detect_best_segment(video, **kwargs)
        results[video.stem] = {"start": round(start, 2), "end": round(end, 2), "score": int(score)}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Segment detection complete.")


# --- Multi-signal convenience helpers ---------------------------------
SIGNAL_KEYPOINTS: Dict[str, Tuple[int, ...]] = {
    # COCO keypoint indices: wrists, knees, hips
    "wrist": (9, 10),   # push/pull
    "knee": (13, 14),   # squat
    "hip": (11, 12),    # deadlift
}


def detect_best_segments(
    video_path: Path,
    signals: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, Dict]:
    """Run detect_best_segment for multiple named signals.

    Returns a dict mapping signal name -> {start, end, score}.
    """
    results: Dict[str, Dict] = {}
    if signals is None:
        signals = ["wrist"]

    for s in signals:
        kp = SIGNAL_KEYPOINTS.get(s)
        if kp is None:
            # allow numeric index tuples passed as signal name
            raise ValueError(f"Unknown signal: {s}")

        start, end, score = detect_best_segment(video_path, keypoint_indices=kp, **kwargs)
        results[s] = {"start": round(start, 2), "end": round(end, 2), "score": int(score)}

    return results


def detect_directory_for_signals(lowres_dir: Path, output_file: Path, signals: Optional[Iterable[str]] = None, **kwargs):
    """Run multi-signal detection over a directory and write nested JSON.

    Output structure: { video_id: { signal_name: {start,end,score}, ... } }
    """
    lowres_dir = Path(lowres_dir)
    out = {}
    for video in sorted(lowres_dir.glob("*.mp4")):
        print(f"🔎 Detecting multi-signal for: {video.name}")
        out[video.stem] = detect_best_segments(video, signals=signals, **kwargs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)

    print("✅ Multi-signal detection complete.")
