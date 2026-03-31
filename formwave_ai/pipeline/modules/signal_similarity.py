"""Signal similarity utilities for reference-based filtering.

Provides helpers to normalize, resample and compare 1D trajectory signals.
"""
from typing import List, Optional
import numpy as np


def normalize_signal(sig: List[float]) -> Optional[np.ndarray]:
    if sig is None:
        return None
    arr = np.asarray(sig, dtype=float)
    if arr.size == 0:
        return None
    mean = arr.mean()
    std = arr.std()
    if std == 0 or np.isnan(std):
        return None
    return (arr - mean) / std


def resample_signal(sig: List[float], target_len: int = 100) -> Optional[np.ndarray]:
    if sig is None:
        return None
    arr = np.asarray(sig, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    if n == target_len:
        return arr.astype(float)
    # interpolate over normalized index
    x_old = np.linspace(0.0, 1.0, num=n)
    x_new = np.linspace(0.0, 1.0, num=target_len)
    try:
        new = np.interp(x_new, x_old, arr)
        return new.astype(float)
    except Exception:
        return None


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation in [ -1, 1 ] then map to [0,1] similarity.

    If signals are invalid returns 0.0
    """
    try:
        if a is None or b is None:
            return 0.0
        a_norm = normalize_signal(a)
        b_norm = normalize_signal(b)
        if a_norm is None or b_norm is None:
            return 0.0
        # ensure same length
        if a_norm.shape != b_norm.shape:
            return 0.0
        corr = float(np.corrcoef(a_norm, b_norm)[0, 1])
        if np.isnan(corr):
            return 0.0
        # map [-1,1] -> [0,1]
        return max(0.0, (corr + 1.0) / 2.0)
    except Exception:
        return 0.0


def compute_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance (simple O(N*M) implementation).

    Returns a non-negative distance. Caller may convert to similarity.
    """
    if a is None or b is None:
        return float('inf')
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float('inf')

    n = a.size
    m = b.size
    # initialize cost matrix with large values
    DTW = np.full((n + 1, m + 1), np.inf, dtype=float)
    DTW[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            last = min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])
            DTW[i, j] = cost + last
    return float(DTW[n, m])


def extract_signal(annotation: dict, key: str = "arm_Trajectory") -> Optional[List[float]]:
    # Prefer the exact key, fallback to other keys if present
    if not annotation or not isinstance(annotation, dict):
        return None
    if key in annotation and isinstance(annotation.get(key), list):
        return annotation.get(key)
    # fallback: pick a key that contains the requested prefix
    for k, v in annotation.items():
        if key.lower() in k.lower() and isinstance(v, list) and len(v) > 0:
            return v
    # last resort: find any numeric list
    for k, v in annotation.items():
        if isinstance(v, list) and len(v) > 5 and all(isinstance(x, (int, float)) for x in v[:5]):
            return v
    return None
