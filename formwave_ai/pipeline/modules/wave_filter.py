"""Wave-based segment filtering utilities.

Provides signal validation, sliding-window segmentation, scoring, and a
video-cutting helper using ffmpeg. Designed to be lightweight and
extensible for multiple exercises/signals.
"""
from pathlib import Path
import subprocess
from typing import List, Tuple, Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ----------------------- CONFIG / HYPERPARAMS -----------------------
STD_THRESHOLD = 0.02        # minimum stddev to consider movement present
MIN_REPS = 2                # minimum number of peaks (repetitions)
PEAK_PROMINENCE = 0.01     # peak prominence for find_peaks
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
MIN_SEGMENT_SEC = 2.0


def _safe_smooth(sig: np.ndarray) -> np.ndarray:
    if sig.size < SMOOTH_WINDOW or SMOOTH_WINDOW % 2 == 0:
        return sig
    try:
        return savgol_filter(sig, SMOOTH_WINDOW, SMOOTH_POLY)
    except Exception:
        return sig


def is_valid_exercise_signal(signal: np.ndarray) -> bool:
    """Return True when `signal` looks like an exercise wave (repetitive movement).

    Criteria:
    - non-empty
    - std(signal) > STD_THRESHOLD
    - contains at least MIN_REPS peaks
    - noise level reasonable compared to amplitude
    """
    if signal is None:
        return False
    sig = np.asarray(signal, dtype=float)
    if sig.size < 5:
        return False

    # drop NaNs
    sig = sig[~np.isnan(sig)]
    if sig.size < 5:
        return False

    sstd = float(np.nanstd(sig))
    if sstd <= STD_THRESHOLD:
        return False

    smooth = _safe_smooth(sig)
    # find valleys/peaks depending on expected polarity — be permissive
    peaks, _ = find_peaks(-smooth, prominence=PEAK_PROMINENCE)
    if peaks.size < MIN_REPS:
        # try the positive peaks as a fallback
        peaks_pos, _ = find_peaks(smooth, prominence=PEAK_PROMINENCE)
        if peaks_pos.size < MIN_REPS:
            return False

    # amplitude-based noise check
    amp = float(np.nanmax(smooth) - np.nanmin(smooth))
    if amp <= 0:
        return False
    residual = sig - smooth
    noise_ratio = float(np.nanstd(residual) / (amp + 1e-8))
    # require residual noise to be not too large (<= 0.6)
    if noise_ratio > 0.6:
        return False

    return True


def detect_valid_segments(signal: np.ndarray, fps: int, window_sec: float = 2.0, step_sec: Optional[float] = None) -> List[Tuple[float, float]]:
    """Sliding-window detect valid segments in a 1D signal.

    Returns list of (start_sec, end_sec) merged and filtered by MIN_SEGMENT_SEC.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0 or fps <= 0:
        return []
    sig = sig.copy()
    # map to samples
    window = max(1, int(window_sec * fps))
    step = int((step_sec or (window_sec / 2.0)) * fps)
    if step <= 0:
        step = 1

    valid_windows = []
    for start in range(0, max(1, len(sig) - window + 1), step):
        seg = sig[start : start + window]
        if is_valid_exercise_signal(seg):
            valid_windows.append((start, start + window))

    if not valid_windows:
        return []

    # merge consecutive/overlapping windows
    merged = []
    cur_s, cur_e = valid_windows[0]
    for s, e in valid_windows[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # convert to seconds and discard short segments
    out = []
    for s, e in merged:
        start_t = float(s) / float(fps)
        end_t = float(e) / float(fps)
        if (end_t - start_t) >= MIN_SEGMENT_SEC:
            out.append((round(start_t, 3), round(end_t, 3)))

    return out


def score_segment(signal_segment: np.ndarray) -> float:
    """Score a signal segment between 0 and 1 based on amplitude, periodicity and smoothness."""
    seg = np.asarray(signal_segment, dtype=float)
    if seg.size < 5:
        return 0.0
    seg = seg[~np.isnan(seg)]
    if seg.size < 5:
        return 0.0

    # amplitude score (normalized)
    amp = float(np.nanmax(seg) - np.nanmin(seg))
    amp_score = np.tanh(amp * 5.0)  # maps to (0,1)

    # periodicity: consistency of peak distances
    smooth = _safe_smooth(seg)
    peaks, _ = find_peaks(-smooth, prominence=PEAK_PROMINENCE)
    if peaks.size < 2:
        periodicity_score = 0.0
    else:
        dists = np.diff(peaks).astype(float)
        mean = float(np.mean(dists))
        std = float(np.std(dists))
        cv = std / (mean + 1e-8)
        periodicity_score = max(0.0, 1.0 - cv)

    # smoothness: low derivative variance
    deriv = np.diff(smooth)
    var_deriv = float(np.nanvar(deriv))
    smooth_score = 1.0 / (1.0 + var_deriv)

    # Combine with weights
    w_amp, w_per, w_smooth = 0.4, 0.4, 0.2
    raw = w_amp * amp_score + w_per * periodicity_score + w_smooth * smooth_score

    # clamp 0..1
    return float(max(0.0, min(1.0, raw)))


def cut_video_segment(input_path: str | Path, output_path: str | Path, start_time: float, end_time: float) -> bool:
    """Cut out a segment with ffmpeg. Returns True on success.

    Tries a stream-copy first for speed; falls back to re-encode if that fails.
    """
    inp = str(input_path)
    out = str(output_path)
    duration = max(0.001, float(end_time) - float(start_time))

    # Attempt fast stream copy (may be slightly imprecise but faster)
    cmd_copy = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        inp,
        "-t",
        str(duration),
        "-c",
        "copy",
        out,
    ]

    try:
        subprocess.run(cmd_copy, check=True)
        return True
    except subprocess.CalledProcessError:
        # fallback: re-encode
        cmd_enc = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            inp,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out,
        ]
        try:
            subprocess.run(cmd_enc, check=True)
            return True
        except subprocess.CalledProcessError:
            return False


def plot_signal_with_segments(signal: np.ndarray, segments: List[Tuple[float, float]], fps: int):
    """Optional: plot signal and overlay segments. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise

    sig = np.asarray(signal, dtype=float)
    t = np.arange(len(sig)) / float(fps)
    plt.figure(figsize=(10, 3))
    plt.plot(t, sig, label="signal")
    for (s, e) in segments:
        plt.axvspan(s, e, color="green", alpha=0.2)
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
