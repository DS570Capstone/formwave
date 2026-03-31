"""Push-up quality filter

Evaluate a pose-based segment annotation JSON and classify it as
`accepted`, `review`, or `rejected` based on pose confidences and
geometric / motion checks.

Public API:
  score_pushup_segment(segment_data: dict) -> dict

Return format:
  {
    "status": "accepted"|"review"|"rejected",
    "score": float,            # 0.0..1.0 combined quality score
    "reasons": [str, ...],
    "metrics": { ... }
  }

The implementation is defensive about missing fields and uses small
helper functions to keep checks readable and testable.
"""
from typing import List, Dict, Any
import math
import numpy as np

try:
    # import pipeline config if available for runtime thresholds
    from pipeline import config as cfg
except Exception:
    cfg = None

# Thresholds and constants (tunable)
# Defaults can be overridden in pipeline.config (recommended)
# valid-frame ratio: fraction of frames with per-frame mean confidence >= FRAME_CONF_THRESHOLD
FRAME_CONF_THRESHOLD = getattr(cfg, 'PUSHUP_FRAME_CONF_THRESHOLD', 0.20)
MIN_VALID_FRAME_RATIO = getattr(cfg, 'PUSHUP_MIN_VALID_FRAME_RATIO', 0.40)
MIN_AVG_JOINT_CONF = getattr(cfg, 'PUSHUP_MIN_AVG_JOINT_CONF', 0.55)
MAX_ELBOW_ANGLE_DEGREES = getattr(cfg, 'PUSHUP_MAX_ELBOW_ANGLE_DEGREES', 180.0)
MIN_ELBOW_ANGLE_DEGREES = getattr(cfg, 'PUSHUP_MIN_ELBOW_ANGLE_DEGREES', 30.0)
# Minimum observed elbow angle range (max-min) across frames to indicate motion
MIN_ELBOW_RANGE_DEGREES = getattr(cfg, 'PUSHUP_MIN_ELBOW_RANGE_DEGREES', 20.0)
MAX_ZERO_CONF_FRAMES_RATIO = getattr(cfg, 'PUSHUP_MAX_ZERO_CONF_FRAMES_RATIO', 0.3)
ALIGNMENT_ANGLE_IDEAL = getattr(cfg, 'PUSHUP_ALIGNMENT_ANGLE_IDEAL', 180.0)  # straight line (shoulder-hip-ankle)
MIN_BODY_ALIGNMENT = getattr(cfg, 'PUSHUP_MIN_BODY_ALIGNMENT', 0.65)
REVIEW_BODY_ALIGNMENT_MARGIN = getattr(cfg, 'PUSHUP_REVIEW_BODY_ALIGNMENT_MARGIN', 0.07)
MOTION_AMPLITUDE_REVIEW = getattr(cfg, 'PUSHUP_MOTION_AMPLITUDE_REVIEW', 0.08)
PUSHUP_TORSO_HORIZONTAL_MIN = getattr(cfg, 'PUSHUP_TORSO_HORIZONTAL_MIN', 0.40)
PUSHUP_ARM_MOTION_MIN_AMPLITUDE = getattr(cfg, 'PUSHUP_ARM_MOTION_MIN_AMPLITUDE', 0.05)
PUSHUP_KEYPOINTS_MISSING_REJECT_CONF = getattr(cfg, 'PUSHUP_KEYPOINTS_MISSING_REJECT_CONF', 0.45)


def _safe_array(x) -> np.ndarray:
    """Return numpy array or empty array if input invalid."""
    try:
        arr = np.asarray(x, dtype=float)
        return arr
    except Exception:
        return np.array([])


def _mean_conf_per_frame(confidence: Any) -> np.ndarray:
    """Compute per-frame mean confidence across joints.

    `confidence` expected shape: [T, 17] or similar.
    Returns a 1D array of length T (may be empty).
    """
    conf = _safe_array(confidence)
    if conf.ndim == 1:
        # single-frame vector
        return conf
    if conf.ndim >= 2:
        # mean across joints (axis=1)
        with np.errstate(invalid='ignore'):
            m = np.nanmean(conf, axis=1)
        return m
    return np.array([])


def _joint_index_map() -> Dict[str, int]:
    """COCO 17 keypoint indices mapping for joints we care about."""
    return {
        'l_shoulder': 5, 'r_shoulder': 6,
        'l_elbow': 7, 'r_elbow': 8,
        'l_wrist': 9, 'r_wrist': 10,
        'l_hip': 11, 'r_hip': 12,
        'l_knee': 13, 'r_knee': 14,
        'l_ankle': 15, 'r_ankle': 16,
    }


def _avg_joint_confidence(confidence: Any, joints: List[str]) -> float:
    """Average confidence for requested joint names across frames.

    Returns nan-safe float in [0,1].
    """
    conf = _safe_array(confidence)
    if conf.size == 0:
        return 0.0
    idx = _joint_index_map()
    cols = [idx.get(j) for j in joints if idx.get(j) is not None]
    cols = [c for c in cols if c is not None and c < conf.shape[1]]
    if not cols:
        return 0.0
    try:
        vals = conf[:, cols]
        return float(np.nanmean(vals))
    except Exception:
        return 0.0


def _angle_at_joint(kpts: np.ndarray, a_idx: int, b_idx: int, c_idx: int) -> np.ndarray:
    """Compute interior angle at joint B for each frame (degrees).

    kpts shape expected: [T, 17, 2]
    Returns 1D array length T (or empty if insufficient data).
    """
    if kpts is None:
        return np.array([])
    arr = _safe_array(kpts)
    if arr.ndim != 3 or arr.shape[1] <= max(a_idx, b_idx, c_idx):
        return np.array([])
    A = arr[:, a_idx, :]
    B = arr[:, b_idx, :]
    C = arr[:, c_idx, :]
    BA = A - B
    BC = C - B
    # normalize
    with np.errstate(invalid='ignore', divide='ignore'):
        BA_n = BA / (np.linalg.norm(BA, axis=1, keepdims=True) + 1e-8)
        BC_n = BC / (np.linalg.norm(BC, axis=1, keepdims=True) + 1e-8)
        dot = np.sum(BA_n * BC_n, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        angles = np.degrees(np.arccos(dot))
    return angles


def _alignment_angle(kpts: np.ndarray, side: str = 'left') -> np.ndarray:
    """Compute approximate shoulder-hip-ankle angle for side ('left'|'right').

    Angle near 180 means straight (good); lower means bent.
    """
    idx = _joint_index_map()
    if side == 'left':
        s, h, a = idx['l_shoulder'], idx['l_hip'], idx['l_ankle']
    else:
        s, h, a = idx['r_shoulder'], idx['r_hip'], idx['r_ankle']
    return _angle_at_joint(kpts, s, h, a)


def _horizontal_stability(kpts: np.ndarray) -> float:
    """Measure stability of shoulder_y - hip_y across frames.

    Lower variance -> more stable horizontal alignment (good). Returns a
    score in [0,1] where 1 is very stable.
    """
    arr = _safe_array(kpts)
    if arr.ndim != 3:
        return 0.0
    idx = _joint_index_map()
    # compute shoulder midpoint Y and hip midpoint Y per frame
    try:
        sho = (arr[:, idx['l_shoulder'], 1] + arr[:, idx['r_shoulder'], 1]) / 2.0
        hip = (arr[:, idx['l_hip'], 1] + arr[:, idx['r_hip'], 1]) / 2.0
    except Exception:
        return 0.0
    diff = sho - hip
    var = float(np.nanvar(diff))
    # map variance -> stability score via simple sigmoid-like mapping
    score = 1.0 / (1.0 + var * 50.0)
    return float(max(0.0, min(1.0, score)))


def _torso_horizontal_score(kpts: np.ndarray) -> float:
    """Estimate how horizontal the torso is across frames.

    Returns mean score in [0,1] where 1.0 == perfectly horizontal, 0.0 == vertical.
    """
    arr = _safe_array(kpts)
    if arr.ndim != 3:
        return 0.0
    idx = _joint_index_map()
    try:
        sho = (arr[:, idx['l_shoulder'], :] + arr[:, idx['r_shoulder'], :]) / 2.0
        hip = (arr[:, idx['l_hip'], :] + arr[:, idx['r_hip'], :]) / 2.0
    except Exception:
        return 0.0
    vec = sho - hip  # [T, 2]
    # angle relative to horizontal (degrees)
    with np.errstate(invalid='ignore'):
        ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    ang = np.abs(ang)
    # fold >90 to mirror (e.g., 100 -> 80)
    ang = np.where(ang <= 90.0, ang, 180.0 - ang)
    # map to score: 0 deg -> 1.0, 90 deg -> 0.0
    scores = 1.0 - (ang / 90.0)
    scores = np.nan_to_num(scores, nan=0.0)
    return float(np.clip(np.nanmean(scores), 0.0, 1.0))


def _arm_motion_amplitude(segment: dict) -> float:
    """Compute motion amplitude specifically from arm trajectory keys.

    Returns None if no arm trajectory available, else normalized amplitude in [0,1].
    """
    for key in ('arm_trajectory', 'arm_Trajectory'):
        sig = segment.get(key)
        if sig is None:
            continue
        arr = _safe_array(sig)
        if arr.size < 2:
            continue
        amp = float(np.nanmax(arr) - np.nanmin(arr))
        norm = amp / (abs(np.nanmean(arr)) + 1.0)
        return float(max(0.0, min(1.0, norm)))
    return None


def _motion_amplitude(segment: dict) -> float:
    """Estimate motion amplitude from available arm trajectory signal or fallbacks.

    Returns amplitude normalized roughly into [0,1] (clamped).
    """
    candidates = ['arm_trajectory', 'arm_Trajectory', 'trajectory', 'core_', 'legs_trajectory']
    for c in candidates:
        sig = segment.get(c)
        if sig is None:
            continue
        arr = _safe_array(sig)
        if arr.size < 2:
            continue
        amp = float(np.nanmax(arr) - np.nanmin(arr))
        # heuristic normalization (depends on scale after scale_normalise in step2)
        norm = amp / (abs(np.nanmean(arr)) + 1.0)
        return float(max(0.0, min(1.0, norm)))
    return 0.0


def score_pushup_segment(segment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main scoring function for a single segment annotation JSON.

    Returns the dict described in the module docstring.
    """
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    # 1) Confidence-based checks
    conf = segment_data.get('confidence') or segment_data.get('conf') or []
    mean_per_frame = _mean_conf_per_frame(conf)
    T = len(mean_per_frame)
    # stronger per-frame validity threshold (configurable)
    valid_frames = float(np.sum(mean_per_frame >= FRAME_CONF_THRESHOLD)) if T > 0 else 0.0
    valid_frame_ratio = (valid_frames / T) if T > 0 else 0.0
    metrics['valid_frame_ratio'] = round(float(valid_frame_ratio), 3)
    metrics['frame_conf_threshold'] = float(FRAME_CONF_THRESHOLD)

    # too many zero-confidence frames -> immediate reject
    zero_conf_frames = float(np.sum(mean_per_frame <= 0.01)) if T > 0 else 0.0
    zero_conf_ratio = (zero_conf_frames / T) if T > 0 else 1.0
    metrics['zero_conf_ratio'] = round(float(zero_conf_ratio), 3)
    if zero_conf_ratio > MAX_ZERO_CONF_FRAMES_RATIO:
        reasons.append('Too many zero-confidence frames')

    # 2) average confidence on required joints
    required_joints = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                       'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    avg_req_conf = _avg_joint_confidence(conf, required_joints)
    metrics['avg_required_joint_conf'] = round(float(avg_req_conf), 3)
    if avg_req_conf < MIN_AVG_JOINT_CONF:
        reasons.append('Low average joint confidence')

    # 3) elbow angle range
    kpts = segment_data.get('keypoints') or segment_data.get('kps') or []
    kpts_arr = _safe_array(kpts)
    elbow_angles = np.array([])
    if kpts_arr.ndim == 3:
        idx = _joint_index_map()
        # Use left elbow angle (shoulder-elbow-wrist)
        elbow_angles_l = _angle_at_joint(kpts_arr, idx['l_shoulder'], idx['l_elbow'], idx['l_wrist'])
        elbow_angles_r = _angle_at_joint(kpts_arr, idx['r_shoulder'], idx['r_elbow'], idx['r_wrist'])
        if elbow_angles_l.size:
            elbow_angles = elbow_angles_l
        if elbow_angles_r.size:
            if elbow_angles.size:
                elbow_angles = np.vstack([elbow_angles, elbow_angles_r]).mean(axis=0)
            else:
                elbow_angles = elbow_angles_r
    if elbow_angles.size:
        mean_elbow = float(np.nanmean(elbow_angles))
        min_elbow = float(np.nanmin(elbow_angles))
        max_elbow = float(np.nanmax(elbow_angles))
        metrics['elbow_angle_min'] = round(min_elbow, 1)
        metrics['elbow_angle_max'] = round(max_elbow, 1)
        metrics['elbow_angle_range'] = round(max_elbow - min_elbow, 1)
        metrics['mean_elbow_angle'] = round(mean_elbow, 1)
        if mean_elbow < MIN_ELBOW_ANGLE_DEGREES or mean_elbow > MAX_ELBOW_ANGLE_DEGREES:
            reasons.append('Elbow angle outside plausible range')
    else:
        metrics['mean_elbow_angle'] = None
        # explicit note: no keypoint geometry available for elbow checks
        reasons.append('Elbow angle not available')

    # 4) body alignment score (shoulder-hip-ankle angle)
    align_left = _alignment_angle(kpts_arr, 'left')
    align_right = _alignment_angle(kpts_arr, 'right')
    align_scores = []
    if align_left.size:
        align_scores.append(np.nanmean(align_left) / ALIGNMENT_ANGLE_IDEAL)
    if align_right.size:
        align_scores.append(np.nanmean(align_right) / ALIGNMENT_ANGLE_IDEAL)
    if align_scores:
        body_alignment_score = float(np.nanmean(align_scores))
    else:
        body_alignment_score = 0.0
    metrics['body_alignment_score'] = round(body_alignment_score, 3)
    if body_alignment_score < 0.7:
        reasons.append('Poor body alignment')

    # 5) horizontal body score (shoulder vs hip stability)
    horiz_score = _horizontal_stability(kpts_arr)
    metrics['horizontal_stability'] = round(horiz_score, 3)
    if horiz_score < 0.5:
        reasons.append('Unstable shoulder-hip vertical relation')

    # torso horizontalness (prefer evaluating on valid frames only)
    torso_score = 0.0
    try:
        if T > 0:
            valid_mask = (mean_per_frame >= FRAME_CONF_THRESHOLD)
            if kpts_arr.ndim == 3 and np.sum(valid_mask) > 0:
                kpts_valid = kpts_arr[valid_mask]
                torso_score = _torso_horizontal_score(kpts_valid)
            else:
                torso_score = _torso_horizontal_score(kpts_arr)
        else:
            torso_score = _torso_horizontal_score(kpts_arr)
    except Exception:
        torso_score = 0.0
    metrics['torso_horizontal_score'] = round(float(torso_score), 3)
    if torso_score < PUSHUP_TORSO_HORIZONTAL_MIN:
        reasons.append('Torso mostly vertical')

    # 6) motion amplitude
    # Prefer arm trajectory for motion amplitude; fall back to generic motion amplitude
    arm_amp = _arm_motion_amplitude(segment_data)
    metrics['arm_motion_amplitude'] = round(float(arm_amp), 3) if arm_amp is not None else None
    motion_amp = arm_amp if arm_amp is not None else _motion_amplitude(segment_data)
    metrics['motion_amplitude'] = round(float(motion_amp), 3)
    if motion_amp < 0.05:
        reasons.append('Very low motion amplitude')

    # 7) valid frame ratio threshold
    if valid_frame_ratio < MIN_VALID_FRAME_RATIO:
        reasons.append('Too few valid frames')

    # If keypoints are missing, be conservative: mark review or reject depending on confidence
    keypoints_available = (kpts_arr.ndim == 3 and kpts_arr.size > 0)
    metrics['keypoints_available'] = bool(keypoints_available)
    if not keypoints_available:
        reasons.append('Keypoints missing')
        if avg_req_conf < PUSHUP_KEYPOINTS_MISSING_REJECT_CONF:
            # very low confidence and no kpts -> reject
            reasons.append('Low confidence and no keypoints')
            # set a flag here; final decision logic will convert to rejected
            metrics['force_reject_no_kpts'] = True

    # Compose a simple combined score in [0,1]
    # We weight: confidences, alignment, horizontal stability, motion amplitude
    conf_score = avg_req_conf
    align_score = body_alignment_score
    stability = horiz_score
    amp = motion_amp

    # weighted geometric mean-like aggregator (keep defensively clamped)
    weights = {'conf': 0.35, 'align': 0.25, 'stab': 0.2, 'amp': 0.2}
    combined = (conf_score * weights['conf'] + align_score * weights['align'] +
                stability * weights['stab'] + amp * weights['amp'])
    score = float(max(0.0, min(1.0, combined)))

    # Decision logic (conservative defaults)
    status = 'review'
    # explicit immediate rejects
    if zero_conf_ratio > MAX_ZERO_CONF_FRAMES_RATIO:
        status = 'rejected'
        reasons.append('Too many zero-confidence frames')
    if avg_req_conf < MIN_AVG_JOINT_CONF:
        status = 'rejected'
        reasons.append('Average joint confidence below threshold')
    if metrics.get('elbow_angle_range') is not None and metrics.get('elbow_angle_range') < MIN_ELBOW_RANGE_DEGREES:
        status = 'rejected'
        reasons.append('Elbow range below threshold')
    if body_alignment_score < MIN_BODY_ALIGNMENT:
        # if body alignment very poor -> reject, if borderline -> review
        if body_alignment_score < (MIN_BODY_ALIGNMENT - REVIEW_BODY_ALIGNMENT_MARGIN):
            status = 'rejected'
            reasons.append('Body alignment indicates non-push-up posture')
        else:
            reasons.append('Borderline body alignment')
    # torso too vertical -> reject
    if metrics.get('torso_horizontal_score') is not None and metrics.get('torso_horizontal_score') < PUSHUP_TORSO_HORIZONTAL_MIN:
        status = 'rejected'
        reasons.append('Torso mostly vertical')

    # missing keypoints with low confidence -> force reject
    if metrics.get('force_reject_no_kpts'):
        status = 'rejected'
        reasons.append('Rejected: missing keypoints & low confidence')

    # No reliable push-up motion: prefer elbow range or arm trajectory amplitude
    elbow_range = metrics.get('elbow_angle_range') if metrics.get('elbow_angle_range') is not None else 0.0
    arm_motion_val = metrics.get('arm_motion_amplitude')
    if (elbow_range < MIN_ELBOW_RANGE_DEGREES) and (arm_motion_val is None or float(arm_motion_val) < PUSHUP_ARM_MOTION_MIN_AMPLITUDE):
        # lack of push-up motion evidence
        status = 'rejected'
        reasons.append('No reliable push-up motion')

    # weak motion -> mark review
    if motion_amp < MOTION_AMPLITUDE_REVIEW and status != 'rejected':
        reasons.append('Low motion amplitude — review')

    # Accept only if combined score is clearly strong and no reject reasons
    if status != 'rejected':
        if score >= getattr(cfg, 'PUSHUP_ACCEPT_SCORE', 0.75) and avg_req_conf >= MIN_AVG_JOINT_CONF and body_alignment_score >= (MIN_BODY_ALIGNMENT + 0.05) and metrics.get('elbow_angle_range', 0) >= MIN_ELBOW_RANGE_DEGREES:
            # ensure no fatal reasons
            fatal = any('Too many zero-confidence' in r or 'below threshold' in r for r in reasons)
            if not fatal:
                status = 'accepted'
        else:
            status = 'review'

    # ensure reasons are unique and meaningful
    reasons = list(dict.fromkeys(reasons))

    return {
        'status': status,
        'score': round(float(score), 3),
        'reasons': reasons,
        'metrics': metrics,
    }


if __name__ == '__main__':
    # quick smoke test placeholder
    sample = {}
    print(score_pushup_segment(sample))
