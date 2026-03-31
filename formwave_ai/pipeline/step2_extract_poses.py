#!/usr/bin/env python3
"""
FormWave — Step 2: Extract Poses with MMPose / MediaPipe Tasks API
===================================================================
For every clip_XXXX.mp4 in data/{exercise}/{camera}/{split}/videos/,
this script:
  1. Runs MMDet to detect the person bounding box per frame
  2. Runs MMPose to extract 17 COCO keypoints per frame
  3. Filters low-confidence keypoints
  4. Scale-normalises keypoints to body proportions (torso = 1.0)
  5. Extracts body-segment trajectories:
       arm_Trajectory   — wrist + elbow positions over time
       core_            — shoulder + hip midpoint (torso axis)
       legs_trajectory  — knee + ankle positions over time
       trajectory       — overall hip midpoint vertical displacement
  6. Saves clip_XXXX.json into data/{exercise}/{camera}/{split}/annotations/

Output clip_XXXX.json schema:
{
  "video_id":        str,
  "exercise":        str,
  "CAMERA_POSITION": str,
  "expert":          false,
  "video":           str,
  "fps":             float,
  "n_frames":        int,
  "keypoints":       [[T, 17, 2]],      ← raw (use for 3D lift later)
  "confidence":      [[T, 17]],
  "arm_Trajectory":  [T],               ← 1D normalised signal
  "core_":           [T],
  "legs_trajectory": [T],
  "trajectory":      [T],
  "error_rate":      [],                ← filled by Gemini step
  "LANGUAGE":        ""                 ← filled by Gemini step
}

Usage
-----
  # Install MMPose (see README — requires mmcv)
  python pipeline/step2_extract_poses.py
  python pipeline/step2_extract_poses.py --data_dir ./data --skip_existing
"""

import argparse
import json
import sys
from pathlib import Path
import types
from datetime import datetime
import logging
import warnings

# ── MAC SILICON MMCV PATCH ──────────────────────────────────────────────────
# MMCV extensions (_ext) are often missing on Mac Silicon. 
# We monkey-patch them to allow MMPoseInferencer to load for standard models.
try:
    import mmcv
    try:
        from mmcv import _ext
    except ImportError:
        # Professional mock for MMCV extensions
        class MockModule(types.ModuleType):
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
            def __reduce__(self):
                return (MockModule, (self.__name__,))

        mock_ext = MockModule("mmcv._ext")
        mock_ext.__path__ = []
        mock_ext.__file__ = "mock_ext.py"
        mock_ext.__loader__ = None
        mock_ext.__spec__ = None
        
        sys.modules["mmcv._ext"] = mock_ext
        
        # Patch loader
        import mmcv.utils.ext_loader as ext_loader
        ext_loader.load_ext = lambda name, funcs: mock_ext
        
        import mmcv.ops
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def is_single_person_video(video_path, sample_frames=10):
    detect_count = 0
    valid_frames = 0
    multi_person_suspect = 0
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return False

    step = max(total_frames // sample_frames, 1)

    valid_frames = 0
    multi_person_suspect = 0

    # prepare additional lightweight detectors (HOG person detector + face cascade)
    face_cascade = None
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        face_cascade = None

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    with mp_pose.Pose(static_image_mode=True) as pose:
        for i in range(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                continue

            # resize for faster detection
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # pose landmarks check
            results = pose.process(small)
            if results.pose_landmarks:
                valid_frames += 1
                # heuristic: low overall landmark visibility may indicate multiple people or occlusion
                try:
                    if results.pose_landmarks.landmark[0].visibility < 0.3:
                        multi_person_suspect += 1
                except Exception:
                    pass

            # face detection: multiple faces -> multi-person
            try:
                if face_cascade is not None:
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    if len(faces) >= 2:
                        multi_person_suspect += 1
            except Exception:
                pass

            # HOG person detector as fallback: detects upright people
            try:
                rects, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
                if rects is not None and len(rects) >= 2:
                    multi_person_suspect += 1
            except Exception:
                pass

    cap.release()

    if valid_frames < 5:
        return False

    if multi_person_suspect > 3:
        return False

    return True


import cv2
import numpy as np
from scipy.signal import savgol_filter

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

logger = logging.getLogger(__name__)

# MediaPipe pose landmarker model (Tasks API, mediapipe >= 0.10)
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ── COCO 17-keypoint indices ──────────────────────────────────────────────────
#  0:nose   1:l_eye   2:r_eye   3:l_ear   4:r_ear
#  5:l_sho  6:r_sho   7:l_elb   8:r_elb   9:l_wri  10:r_wri
# 11:l_hip  12:r_hip  13:l_kne  14:r_kne  15:l_ank  16:r_ank

KP = {
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow":    7,  "r_elbow":    8,
    "l_wrist":    9,  "r_wrist":   10,
    "l_hip":     11,  "r_hip":     12,
    "l_knee":    13,  "r_knee":    14,
    "l_ankle":   15,  "r_ankle":   16,
}

CONF_THRESHOLD = 0.3          # discard joints below this confidence
SG_WINDOW      = 7           # Savitzky-Golay smoothing window (frames)
SG_POLYORDER   = 3


# ─────────────────────────────────────────────────────────────────────────────
# MMPose inference (with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

# ── RTMPose ONNX Backend (Mac Silicon Optimized) ───────────────────────────
class RTMPoseONNXInferencer:
    """
    Stand-in for MMPoseInferencer that uses ONNX Runtime.
    Provides identical keypoints to MMPose RTMPose-M without MMCV.
    Uses MediaPipe for fast person detection.
    """
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        # 1. RTMPose Setup
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_size = (192, 256) # W, H
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        # 2. Detector Setup (MediaPipe)
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        print(f"[OK] MMPose-ONNX backend loaded (RTMPose-M) via {ort.get_device()}")

    def detect_person_bbox(self, img_rgb):
        # Use MediaPipe to get a person bbox
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = self.detector.detect(mp_image)
        
        if not res.pose_landmarks:
            return None
        
        # Calculate bbox from landmarks
        h, w = img_rgb.shape[:2]
        lms = res.pose_landmarks[0]
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        return [min(xs), min(ys), max(xs), max(ys)]

    def infer_pose(self, img_rgb, bbox):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        center = np.array([x1 + bw/2, y1 + bh/2], dtype=np.float32)
        scale = np.array([bw * 1.25, bh * 1.25], dtype=np.float32)
        
        # Aspect ratio correction
        aspect_ratio = self.input_size[0] / self.input_size[1]
        if scale[0] > scale[1] * aspect_ratio:
            scale[1] = scale[0] / aspect_ratio
        else:
            scale[0] = scale[1] * aspect_ratio
            
        shift = scale / 2
        crop_x1, crop_y1 = int(center[0] - shift[0]), int(center[1] - shift[1])
        crop_x2, crop_y2 = int(center[0] + shift[0]), int(center[1] + shift[1])
        
        ih, iw = img_rgb.shape[:2]
        pad_l = max(0, -crop_x1); pad_t = max(0, -crop_y1)
        pad_r = max(0, crop_x2 - iw); pad_b = max(0, crop_y2 - ih)
        
        if any([pad_l, pad_t, pad_r, pad_b]):
            img_padded = cv2.copyMakeBorder(img_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            crop_x1 += pad_l; crop_x2 += pad_l
            crop_y1 += pad_t; crop_y2 += pad_t
        else:
            img_padded = img_rgb
            
        crop = img_padded[max(0, crop_y1):crop_y2, max(0, crop_x1):crop_x2]
        if crop.size == 0: return np.zeros((17, 2)), np.zeros(17)
        
        resized = cv2.resize(crop, self.input_size)
        inp = (resized.astype(np.float32) - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)[None, ...]
        
        # Inference
        outputs = self.session.run(None, {"input": inp})
        simcc_x, simcc_y = outputs[0], outputs[1]
        
        # Decode SimCC
        x_idx = np.argmax(simcc_x, axis=2)[0]
        y_idx = np.argmax(simcc_y, axis=2)[0]
        scores = (np.max(simcc_x, axis=2)[0] + np.max(simcc_y, axis=2)[0]) / 2.0
        
        # Map back
        kpts = np.zeros((17, 2), dtype=np.float32)
        kpts[:, 0] = (x_idx / 2.0 / self.input_size[0] - 0.5) * scale[0] + center[0]
        kpts[:, 1] = (y_idx / 2.0 / self.input_size[1] - 0.5) * scale[1] + center[1]
        
        return kpts, scores

def _build_inferencer():
    """
    Build the best available pose backend.
    RTMPose-ONNX (MMPose equiv) > MediaPipe native.
    """
    # 1. Check for ONNX weights (Best on Mac)
    onnx_path = Path(__file__).parent / "models" / "rtmpose_m_onnx" / "20230831" / "rtmpose_onnx" / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504" / "end2end.onnx"
    
    if onnx_path.exists():
        try:
            return "rtmpose_onnx", RTMPoseONNXInferencer(onnx_path)
        except Exception as e:
            print(f"[WARN] Failed to load RTMPose-ONNX: {e}")

    # 2. Try real MMPose (if available)
    try:
        from mmpose.apis import MMPoseInferencer
        inferencer = MMPoseInferencer(
            pose2d="rtmpose-m_8xb256-420e_coco-256x192",
            det_model="rtmdet-nano_8xb32-100e_coco-obj365-person-235e",
            device="cpu",
        )
        print("[OK] MMPose inferencer loaded (RTMPose-M)")
        return "mmpose", inferencer
    except Exception:
        pass

    # 3. Fallback to MediaPipe
    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading MediaPipe model...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        print("[OK] MediaPipe PoseLandmarker loaded (IMAGE mode)")
        return "mediapipe", landmarker
    except Exception as e:
        print(f"[ERROR] All pose backends failed: {e}")
        return None, None


def extract_keypoints_onnx(inferencer, video_path: Path, frame_skip: int = 2) -> tuple:
    """Extract keypoints using RTMPose ONNX implementation."""
    logger.debug("Starting ONNX keypoint extraction: %s (frame_skip=%s)", video_path.name, frame_skip)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_interval = max(1, total_frames // 10) if total_frames > 0 else max(1, frame_skip * 100)
    all_kpts, all_confs = [], []

    frame_idx = 0

    bbox = None
    prev_frame = None
    skipped_since_log = 0
    detect_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            skipped_since_log += 1
            frame_idx += 1
            continue
        frame_idx += 1

        # reduce resolution for speed (smaller but adequate)
        frame = cv2.resize(frame, (480, 270))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Reduce detection frequency: detect every (frame_skip * 5) frames
        # e.g. frame_skip=4 -> detect every 20 frames (huge speedup with minor quality loss)
        if frame_idx % (frame_skip * 5) == 0 or bbox is None:
            bbox = inferencer.detect_person_bbox(rgb)
            detect_count += 1
            # Log detect occasionally (one per N detects) to avoid flood
            if detect_count % 5 == 0:
                logger.debug("Frame %d: detect called — landmarks=%s skipped_since_last_detect=%d detect_count=%d", frame_idx, bool(bbox), skipped_since_log, detect_count)
            skipped_since_log = 0
        
        if bbox is not None:
            kpts, confs = inferencer.infer_pose(rgb, bbox)
        else:
            kpts, confs = np.zeros((17, 2)), np.zeros(17)

        all_kpts.append(kpts)
        all_confs.append(confs)
        # Progress: log at 10% intervals (compact)
        if total_frames > 0 and frame_idx % progress_interval == 0:
            pct = int(frame_idx / total_frames * 100)
            logger.info("Progress %d%% — frame %d/%d", pct, frame_idx, total_frames)

    cap.release()
    if not all_kpts: return np.zeros((1, 17, 2)), np.zeros((1, 17)), fps
    return np.stack(all_kpts), np.stack(all_confs), fps


def extract_keypoints_mmpose(inferencer, video_path: Path, frame_skip: int = 2) -> tuple:
    """Extract keypoints from video using MMPose. Returns (kpts, confs, fps)."""
    logger.debug("Starting MMPose extraction: %s (frame_skip=%s)", video_path.name, frame_skip)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_interval = max(1, total_frames // 10) if total_frames > 0 else max(1, frame_skip * 100)
    cap.release()

    result_gen = inferencer(str(video_path), show=False, progress=False)

    all_kpts  = []
    all_confs = []
    frame_idx = 0
    skipped_since_log = 0
    last_result = None
    detect_count = 0
    for frame_result in result_gen:
        if frame_idx % frame_skip != 0:
            skipped_since_log += 1
            frame_idx += 1
            continue
        preds = frame_result.get("predictions", [[]])
        if preds and preds[0]:
            person = preds[0][0]  # first detected person
            kpts   = np.array(person["keypoints"],   dtype=np.float32)   # [17, 2]
            confs  = np.array(person["keypoint_scores"], dtype=np.float32)  # [17]
        else:
            kpts  = np.zeros((17, 2), dtype=np.float32)
            confs = np.zeros(17, dtype=np.float32)

        all_kpts.append(kpts)
        all_confs.append(confs)
        # Progress: log at 10% intervals
        if total_frames > 0 and frame_idx % progress_interval == 0:
            pct = int(frame_idx / total_frames * 100)
            logger.info("Progress %d%% — frame %d/%d", pct, frame_idx, total_frames)
            skipped_since_log = 0
        frame_idx += 1

    return np.stack(all_kpts), np.stack(all_confs), fps


def _tasks_landmarks_to_coco(result, h: int, w: int) -> tuple:
    """
    Convert MediaPipe PoseLandmarker result to COCO 17-keypoint format.
    MediaPipe 33 landmarks → COCO 17 subset.
    Returns (kpts [17,2], conf [17]).
    """
    # Mapping: COCO index → MediaPipe landmark index
    mp_map = {
        0:  0,   # nose
        1:  2,   # l_eye
        2:  5,   # r_eye
        3:  7,   # l_ear
        4:  8,   # r_ear
        5:  11,  # l_shoulder
        6:  12,  # r_shoulder
        7:  13,  # l_elbow
        8:  14,  # r_elbow
        9:  15,  # l_wrist
        10: 16,  # r_wrist
        11: 23,  # l_hip
        12: 24,  # r_hip
        13: 25,  # l_knee
        14: 26,  # r_knee
        15: 27,  # l_ankle
        16: 28,  # r_ankle
    }
    kpts  = np.zeros((17, 2), dtype=np.float32)
    confs = np.zeros(17,      dtype=np.float32)

    if result.pose_landmarks:
        lms = result.pose_landmarks[0]   # first detected person
        for coco_idx, mp_idx in mp_map.items():
            lm = lms[mp_idx]
            kpts[coco_idx]  = [lm.x * w, lm.y * h]
            confs[coco_idx] = lm.visibility if hasattr(lm, 'visibility') else 0.9

    return kpts, confs


def extract_keypoints_mediapipe(landmarker, video_path: Path, frame_skip: int = 2) -> tuple:
    """Extract keypoints using MediaPipe PoseLandmarker Tasks API."""
    import mediapipe as mp
    logger.debug("Starting MediaPipe extraction: %s (frame_skip=%s)", video_path.name, frame_skip)

    cap = cv2.VideoCapture(str(video_path))
    detect_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_interval = max(1, total_frames // 10) if total_frames > 0 else max(1, frame_skip * 100)

    all_kpts, all_confs = [], []
    frame_idx = 0
    skipped_since_log = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            skipped_since_log += 1
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        # Use detect() for IMAGE mode (no timestamp needed)
        # Reduce detection frequency: reuse last_result between detects
        if frame_idx % (frame_skip * 5) == 0 or last_result is None:
            result = landmarker.detect(mp_image)
            last_result = result
            detect_count += 1
            # Log detect occasionally (one per N detects) to avoid flood
            if detect_count % 5 == 0:
                logger.debug("MediaPipe detect frame %d — landmarks=%s skipped_since_last_detect=%d detect_count=%d", frame_idx, bool(result.pose_landmarks), skipped_since_log, detect_count)
            skipped_since_log = 0
        else:
            result = last_result

        kpts, confs = _tasks_landmarks_to_coco(result, h, w)
        all_kpts.append(kpts)
        all_confs.append(confs)
        # Progress: log at 10% intervals
        if total_frames > 0 and frame_idx % progress_interval == 0:
            pct = int(frame_idx / total_frames * 100)
            logger.info("Progress %d%% — frame %d/%d", pct, frame_idx, total_frames)
        frame_idx += 1

    cap.release()

    if not all_kpts:
        return np.zeros((1, 17, 2)), np.zeros((1, 17)), fps

    return np.stack(all_kpts), np.stack(all_confs), fps


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def filter_low_confidence(kpts: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """Set low-confidence keypoints to interpolated values."""
    T, J, _ = kpts.shape
    out = kpts.copy()
    for j in range(J):
        mask = confs[:, j] < CONF_THRESHOLD
        if mask.all():
            continue   # all frames low-confidence — leave as zeros
        if mask.any():
            # linear interpolation for missing frames
            good_t  = np.where(~mask)[0]
            for c in range(2):
                out[:, j, c] = np.interp(
                    np.arange(T), good_t, out[good_t, j, c]
                )
    return out


def scale_normalise(kpts: np.ndarray) -> np.ndarray:
    """
    Normalise keypoints so torso length = 1.0 unit.
    Torso = midpoint(shoulders) → midpoint(hips).
    Also centre x,y around hip midpoint.
    """
    out = kpts.copy().astype(np.float32)
    T   = out.shape[0]

    for t in range(T):
        l_sho = out[t, KP["l_shoulder"]]
        r_sho = out[t, KP["r_shoulder"]]
        l_hip = out[t, KP["l_hip"]]
        r_hip = out[t, KP["r_hip"]]

        sho_mid  = (l_sho + r_sho) / 2.0
        hip_mid  = (l_hip + r_hip) / 2.0
        torso_len = np.linalg.norm(sho_mid - hip_mid)

        if torso_len < 1e-6:
            torso_len = 1.0   # protect div-by-zero on missed frames

        out[t] = (out[t] - hip_mid) / torso_len

    return out.astype(np.float32)


def smooth(sig: np.ndarray) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along time axis."""
    T = sig.shape[0]
    win = min(SG_WINDOW, T - (1 if T % 2 == 0 else 0))
    if win < SG_POLYORDER + 2:
        return sig
    if win % 2 == 0:
        win -= 1
    return savgol_filter(sig, window_length=win, polyorder=SG_POLYORDER, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Biomechanical Analysis Utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute interior angle at joint B given positions A, B, C.
    A, B, C are [T, 2] arrays. Returns [T] array in degrees.
    """
    BA = A - B
    BC = C - B
    
    # Normalise vectors
    BA_norm = BA / (np.linalg.norm(BA, axis=1, keepdims=True) + 1e-8)
    BC_norm = BC / (np.linalg.norm(BC, axis=1, keepdims=True) + 1e-8)
    
    # Dot product + arccos
    dot = np.sum(BA_norm * BC_norm, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _midpoint_y(kpts: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    """Return the mean Y-coordinate between two joints over time. Shape: [T]"""
    return (kpts[:, idx_a, 1] + kpts[:, idx_b, 1]) / 2.0


def _midpoint_dist(kpts: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    """Return Euclidean distance between two joints over time. Shape: [T]"""
    diff = kpts[:, idx_a, :] - kpts[:, idx_b, :]
    return np.linalg.norm(diff, axis=1)


def extract_trajectories(kpts: np.ndarray, exercise: str, camera: str) -> dict:
    """
    Extract exercise-specific 1D signals and angles from keypoints.
    Follows the rules in system-prompt-to-understand-type-of-exercise-and-camera-angle.md
    """
    kp_sm = smooth(kpts)  # [T, 17, 2]
    T = kp_sm.shape[0]
    
    # Default outputs
    results = {
        "trajectory": _midpoint_y(kp_sm, KP["l_hip"], KP["r_hip"]).tolist(),
    }
    
    ex = exercise.lower()
    cam = camera.upper()

    # Shared helper: get joint sequence
    def j(name): return kp_sm[:, KP[name]]

    # 1. SQUAT Logic
    if "squat" in ex:
        if cam == "SIDE":
            # Primary: Hip Y, Knee Y, Shoulder Y
            results["legs_trajectory"] = _midpoint_y(kp_sm, KP["l_knee"], KP["r_knee"]).tolist()
            results["shoulder_trajectory"] = _midpoint_y(kp_sm, KP["l_shoulder"], KP["r_shoulder"]).tolist()
            
            # Critical Angles
            results["back_trajectory"] = compute_angle(j("l_shoulder"), j("l_hip"), j("l_knee")).tolist()
            results["knee_angle_trajectory"] = compute_angle(j("l_hip"), j("l_knee"), j("l_ankle")).tolist()
            
        elif cam == "FRONT":
            # Bilateral Knee X for cave detection
            results["knee_l_x"] = kp_sm[:, KP["l_knee"], 0].tolist()
            results["knee_r_x"] = kp_sm[:, KP["r_knee"], 0].tolist()
            results["legs_trajectory"] = _midpoint_y(kp_sm, KP["l_knee"], KP["r_knee"]).tolist()

    # 2. DEADLIFT Logic
    elif "deadlift" in ex:
        if cam == "SIDE":
            results["legs_trajectory"] = _midpoint_y(kp_sm, KP["l_knee"], KP["r_knee"]).tolist()
            results["bar_path_trajectory"] = _midpoint_y(kp_sm, KP["l_wrist"], KP["r_wrist"]).tolist()
            results["back_trajectory"] = compute_angle(j("l_shoulder"), j("l_hip"), j("l_knee")).tolist()
            
    # 3. PUSHUP Logic
    elif "pushup" in ex:
        if cam == "SIDE":
            results["legs_trajectory"] = _midpoint_y(kp_sm, KP["l_hip"], KP["r_hip"]).tolist()
            results["core_trajectory"] = compute_angle(j("l_shoulder"), j("l_hip"), j("l_ankle")).tolist()
            results["arm_trajectory"] = compute_angle(j("l_shoulder"), j("l_elbow"), j("l_wrist")).tolist()

    # 4. PULLUP Logic
    elif "pullup" in ex or "chinup" in ex:
        if cam == "FRONT":
            results["arm_trajectory"] = compute_angle(j("l_shoulder"), j("l_elbow"), j("l_wrist")).tolist()
            results["chin_clearance_trajectory"] = kp_sm[:, 0, 1].tolist() # Nose Y
        elif cam == "SIDE":
            results["arm_trajectory"] = compute_angle(j("l_shoulder"), j("l_elbow"), j("l_wrist")).tolist()

    # 5. OVERHEAD PRESS Logic
    elif any(x in ex for x in ["overhead", "press", "ohp"]):
        if cam == "FRONT":
            # Primary: Bar path (wrists), Elbow drive
            results["bar_path_trajectory"] = _midpoint_y(kp_sm, KP["l_wrist"], KP["r_wrist"]).tolist()
            results["arm_trajectory"] = compute_angle(j("l_shoulder"), j("l_elbow"), j("l_wrist")).tolist()
            results["shoulder_symmetry"] = (kp_sm[:, KP["l_shoulder"], 1] - kp_sm[:, KP["r_shoulder"], 1]).tolist()
        elif cam == "SIDE":
            results["bar_path_trajectory"] = _midpoint_y(kp_sm, KP["l_wrist"], KP["r_wrist"]).tolist()
            results["back_trajectory"] = compute_angle(j("l_shoulder"), j("l_hip"), j("l_ankle")).tolist()
            results["arm_trajectory"] = compute_angle(j("l_shoulder"), j("l_elbow"), j("l_wrist")).tolist()

    # Fallback/Legacy support for the wave analysis engine
    if "legs_trajectory" not in results:
        results["legs_trajectory"] = _midpoint_y(kp_sm, KP["l_knee"], KP["r_knee"]).tolist()
    if "arm_Trajectory" not in results:
        results["arm_Trajectory"] = _midpoint_y(kp_sm, KP["l_wrist"], KP["r_wrist"]).tolist()
    if "core_" not in results:
        results["core_"] = (_midpoint_y(kp_sm, KP["l_shoulder"], KP["r_shoulder"]) - 
                            _midpoint_y(kp_sm, KP["l_hip"], KP["r_hip"])).tolist()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def process_clip(
    video_path: Path,
    exercise:   str,
    camera:     str,
    split:      str,
    backend:    str,
    inferencer,
    frame_skip: int = 2,
) -> dict:
    """Full processing for one video clip. Returns clip annotation dict."""

    # Extract raw keypoints
    logger.info("Processing clip %s with backend=%s frame_skip=%s", video_path.name, backend, frame_skip)
    start_t = datetime.now()

    if backend == "mmpose":
        kpts, confs, fps = extract_keypoints_mmpose(inferencer, video_path, frame_skip=frame_skip)
    elif backend == "rtmpose_onnx":
        kpts, confs, fps = extract_keypoints_onnx(inferencer, video_path, frame_skip=frame_skip)
    else:
        kpts, confs, fps = extract_keypoints_mediapipe(inferencer, video_path, frame_skip=frame_skip)

    T, J, _ = kpts.shape
    duration = (datetime.now() - start_t).total_seconds()
    logger.info("Finished extraction for %s — frames=%d fps=%.2f duration=%.2fs", video_path.name, T, fps, duration)

    # Filter low-confidence + normalise
    kpts_filtered  = filter_low_confidence(kpts, confs)
    kpts_norm      = scale_normalise(kpts_filtered)

    # Extract body-segment trajectories
    trajectories = extract_trajectories(kpts_norm, exercise, camera)

    return {
        "video_id":        video_path.stem,
        "exercise":        exercise,
        "CAMERA_POSITION": camera.upper(),
        "expert":          False,              # manually reviewed later
        "video":           video_path.name,
        "fps":             round(fps, 2),
        "n_frames":        T,
        # Keypoints stored as nested list [T, 17, 2]
        "keypoints":       kpts_norm.tolist(),
        "confidence":      confs.tolist(),
        # Body-segment 1D signals
        **trajectories,
        # Filled in by Step 3 (Gemini)
        "error_rate": [],
        "LANGUAGE":   "",
    }


def run(args):
    data_root = Path(args.data_dir)
    # Configure logging
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    # Reduce noisy logs from third-party libs when not verbose
    if not getattr(args, "verbose", False):
        logging.getLogger('absl').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)

    # Suppress known protobuf deprecation warning (harmless)
    warnings.filterwarnings("ignore", message=".*GetPrototype.*deprecated.*", category=UserWarning)

    backend, inferencer = _build_inferencer()

    # Find all full videos under data/raw/videos
    video_dir = Path(args.data_dir) / "raw" / "videos"
    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"[ERROR] No mp4 videos found under {video_dir}")
        print("  Run Step 1 first: python pipeline/step1_download_ytd.py")
        sys.exit(1)

    print(f"\n[OK] Found {len(video_files)} videos in {video_dir}")
    print(f"[OK] Using backend: {backend}")
    print("=" * 60)

    processed = skipped = errors = 0
    # Tracking (lazy import to avoid top-level dependency issues)
    try:
        from pipeline.tracking import (
            load_tracking, save_tracking, is_clip_processed, mark_clip_status, reset_tracking,
        )
    except Exception:
        from tracking import (
            load_tracking, save_tracking, is_clip_processed, mark_clip_status, reset_tracking,
        )

    if getattr(args, "reset_tracking", False):
        reset_tracking(data_root)
    tracking = load_tracking(data_root)


    for i, vid_path in enumerate(video_files):
        # For full-video processing, we don't rely on nested clip folders.
        exercise, camera, split = "unknown", "RAW", "raw"
        ann_dir = Path(args.data_dir) / "annotations"
        ann_path = ann_dir / f"{vid_path.stem}.json"

        print(f"\n[{i+1}/{len(video_files)}] {vid_path.stem}  (full video)")

        clip_id = vid_path.stem
        if is_clip_processed(tracking, clip_id) and not getattr(args, "force_reprocess", False):
            print(f"   [VIDEO SKIP] {clip_id} already processed")
            skipped += 1
            continue

        # quick validation: ensure video has sufficient frames to process
        try:
            cap_check = cv2.VideoCapture(str(vid_path))
            if not cap_check.isOpened():
                print(f"   [VIDEO SKIPPED] {clip_id} unreadable by OpenCV")
                mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="unreadable")
                save_tracking(tracking, data_root)
                cap_check.release()
                skipped += 1
                continue
            n_frames_est = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap_check.release()
            if n_frames_est < 30:
                print(f"   [VIDEO SKIPPED] {clip_id} too few frames ({n_frames_est})")
                mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="too_few_frames")
                save_tracking(tracking, data_root)
                skipped += 1
                continue
        except Exception:
            pass

        # Quick single-person heuristic before heavy pose extraction
        try:
            if not is_single_person_video(vid_path):
                print(f"   ❌ Skipping (multi-person or bad pose): {vid_path}")
                mark_clip_status(tracking, clip_id, processed=False, status="skipped", reason="multi_person_or_bad_pose")
                save_tracking(tracking, data_root)
                skipped += 1
                continue
        except Exception:
            # If the quick check fails for any reason, proceed with normal flow
            pass

        if args.skip_existing and ann_path.exists():
            print("   → Already processed (annotation exists). Skipping.")
            skipped += 1
            mark_clip_status(tracking, clip_id, processed=True, status="ok", annotation_path=str(ann_path))
            save_tracking(tracking, data_root)
            continue

        try:
            clip_data = process_clip(
                vid_path, exercise, camera, split, backend, inferencer,
                frame_skip=getattr(args, "frame_skip", 2),
            )

            # 🚨 Skip bad videos (safety): ensure sufficient frames
            if clip_data is None or clip_data.get("n_frames", 0) < 30:
                nfr = clip_data.get('n_frames', 0) if clip_data else 0
                print(f"   [VIDEO SKIPPED] Too few frames (frames={nfr})")
                mark_clip_status(
                    tracking, clip_id,
                    video_id=clip_data.get('video_id') if clip_data else clip_id,
                    exercise=exercise, camera=camera, split=split,
                    processed=False, n_frames=nfr, annotation_path="",
                    status="skipped", reason="too_few_frames",
                )
                save_tracking(tracking, data_root)
                skipped += 1
                continue

            ann_dir.mkdir(parents=True, exist_ok=True)
            # Log pose extraction
            print(f"   [POSE EXTRACTED] {clip_id} -> {ann_path.name}")
            # Save full `keypoints` by default for raw/full-video processing
            # (these files are used by curated push-up workflows). Users may
            # still opt-out with `--save_keypoints=False` if desired.
            save_keypoints = getattr(args, "save_keypoints", False) or (split == 'raw')

            if save_keypoints:
                save_data = clip_data
            else:
                save_data = {k: v for k, v in clip_data.items() if k != "keypoints"}

            with open(ann_path, "w") as f:
                json.dump(save_data, f, indent=2)

            T = clip_data["n_frames"]
            print(f"   → {T} frames | fps={clip_data['fps']} | saved: {ann_path.name}")
            processed += 1

            # Mark clip processed
            mark_clip_status(
                tracking, clip_id,
                video_id=clip_data.get("video_id"),
                exercise=exercise, camera=camera, split=split,
                processed=True, n_frames=T, annotation_path=str(ann_path),
                status="ok",
            )
            save_tracking(tracking, data_root)

        except Exception as e:
            import traceback
            msg = str(e)
            print(f"   [CLIP FAILED] Processing {vid_path.name}: {msg}")
            if args.verbose:
                traceback.print_exc()
            # mark failure
            reason = "decode_error" if "av1" in msg.lower() else msg
            mark_clip_status(
                tracking, clip_id,
                video_id=vid_path.stem, exercise=exercise, camera=camera, split=split,
                processed=False, n_frames=0, annotation_path="",
                status="failed", reason=reason,
            )
            save_tracking(tracking, data_root)
            errors += 1

    print("\n" + "=" * 60)
    print("  Step 2 Complete — Pose Extraction Summary")
    print("=" * 60)
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"\n  Run Step 3 next:")
    print(f"  export GEMINI_API_KEY='your-key'")
    print(f"  python annotate_with_gemini.py --data_dir {data_root}")


def parse_args():
    p = argparse.ArgumentParser(description="FormWave Step 2: Extract poses from video clips")
    p.add_argument("--data_dir",       default="./data",
                   help="Root data directory containing clip videos")
    p.add_argument("--skip_existing",  action="store_true",
                   help="Skip clips that already have annotation JSONs")
    p.add_argument("--save_keypoints", action="store_true",
                   help="Save full [T,17,2] keypoints in annotation (large files)")
    p.add_argument("--frame_skip", type=int, default=2,
                   help="Skip frames when extracting keypoints to speed up processing (default: 2)")
    p.add_argument("--verbose",        action="store_true",
                   help="Print full tracebacks on errors")
    p.add_argument("--force_reprocess", action="store_true",
                   help="Ignore tracking and force reprocess clips")
    p.add_argument("--reset_tracking", action="store_true",
                   help="Delete tracking.json and start fresh")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
