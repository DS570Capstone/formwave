#!/usr/bin/env python3
"""
FormWave — Gemini-Powered Annotation Pipeline
==============================================
Reads clip annotation JSONs from the dataset folder structure:

    data/{exercise}/{camera}/{split}/annotations/clip_XXXX.json

For each clip:
  1. Loads trajectory data (arm, core, legs, overall)
  2. Runs BiomechanicalWaveExtractor (wave physics analysis)
  3. Sends metrics to Gemini API for natural-language annotation
  4. Saves enriched annotations + Alpaca-format LLM training dataset

Output
------
  ./outputs/annotated/          — enriched clip JSONs with LANGUAGE field
  ./outputs/train_wave_alpaca.json
  ./outputs/val_wave_alpaca.json
  ./outputs/test_wave_alpaca.json
  ./outputs/annotation_log.jsonl — per-clip Gemini responses for audit

Usage
-----
  export GEMINI_API_KEY="your-key-here"
  python annotate_with_gemini.py --data_dir ./data
  python annotate_with_gemini.py --data_dir ./data --dry_run  # no API calls
  python annotate_with_gemini.py --data_dir ./data --max_clips 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import numpy as np

# ── MMPose / wave physics ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from wave_physics_network import BiomechanicalWaveExtractor, WavePhysicsConfig
except ImportError as e:
    print(f"[ERROR] Cannot import wave_physics_network: {e}")
    sys.exit(1)

# ── Gemini ────────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARN] google-genai not installed. Run: pip install google-genai")
    print("       Continuing in dry-run mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_DISPLAY = {
    "push": "Push-Up",
    "push up": "Push-Up",
    "squat": "Squat",
    "deadlift": "Deadlift",
    "ohp": "Overhead Press",
    "overhead press": "Overhead Press",
    "barbell_row": "Barbell Row",
    "curl": "Bicep Curl",
    "bench": "Bench Press",
    "bench press": "Bench Press",
    "lunge": "Lunge",
    "plank": "Plank",
}

CAMERA_DISPLAY = {
    "front": "front-facing camera",
    "back": "rear-facing camera",
    "side": "side-profile camera",
}

# ── GEMINI system prompt (read from file) ────────────────────────────────────
SYSTEM_PROMPT_PATH = SCRIPT_DIR / "system-prompt-to-understand-type-of-exercise-and-camera-angle.md"
if SYSTEM_PROMPT_PATH.exists():
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        GEMINI_SYSTEM_PROMPT = f.read()
else:
    GEMINI_SYSTEM_PROMPT = """You are an expert biomechanics coach. Analyze the provided \
wave physics metrics and provide specific, concise feedback on exercise form."""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load clip annotation
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(json_path: Path) -> dict:
    """Load a clip annotation JSON. Returns None if unreadable."""
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot load {json_path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract primary trajectory signal
# ─────────────────────────────────────────────────────────────────────────────

def get_primary_signal(clip: dict, fps: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the primary vertical displacement signal from clip data.
    Prioritizes exercise-specific signals from the new system.
    """
    # Priority list of possible trajectory keys — favor exercise-specific signals
    keys = [
        "bar_path_trajectory", "back_trajectory", "arm_trajectory", "arm_Trajectory",
        "knee_angle_trajectory", "core_trajectory", "legs_trajectory",
        "trajectory", "core_"
    ]

    for key in keys:
        data = clip.get(key)
        if data is not None and len(data) > 10:
            arr = np.array(data, dtype=np.float32).flatten()
            if arr.size > 10 and not np.all(arr == 0):
                return arr, key

    # Fallback: raw keypoints → hip midpoint Y
    kpts = clip.get("keypoints")
    if kpts is not None:
        kpts_arr = np.array(kpts, dtype=np.float32)
        if kpts_arr.ndim == 3 and kpts_arr.shape[1] >= 13:
            # COCO: 11=l_hip, 12=r_hip
            hip_y = (kpts_arr[:, 11, 1] + kpts_arr[:, 12, 1]) / 2.0
            return hip_y, "keypoints_hip_y"

    return np.zeros(60, dtype=np.float32), "fallback_zeros"


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] range."""
    mn, mx = sig.min(), sig.max()
    if mx - mn > 1e-6:
        return 2.0 * (sig - mn) / (mx - mn) - 1.0
    return sig - sig.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Wave physics analysis
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    """Return default for NaN/Inf/None."""
    try:
        if val is None:
            return default
        f = float(val)
        return default if (f != f or abs(f) == float("inf")) else f
    except Exception:
        return default


def run_wave_physics(signal: np.ndarray, fps: int = 30) -> dict:
    """Run BiomechanicalWaveExtractor and return sanitized feature dict."""
    config  = WavePhysicsConfig(fps=fps, dt=1.0 / fps)
    ex      = BiomechanicalWaveExtractor(config)
    norm    = normalize_signal(signal)
    vel     = ex._derivative(norm, order=1)

    raw = ex.extract_wave_features(norm, vel)

    # ── sanitize ─────────────────────────────────────────────────────────────
    q = raw.get("quality",   {})
    e = raw.get("energy",    {})
    d = raw.get("damping",   {})
    f = raw.get("frequency", {})
    h = raw.get("harmonic",  {})
    w = raw.get("waves",     [])

    bp = f.get("band_power", {})

    features = {
        "quality": {
            "grade":             q.get("grade", "C"),
            "overall":           _safe(q.get("overall_quality"), 0.5),
            "smoothness":        _safe(q.get("smoothness_score"), 0.5),
            "control":           _safe(q.get("control_score"),    0.5),
            "efficiency":        min(1.0, max(0.0, _safe(q.get("efficiency_score"), 0.4))),
            "consistency":       _safe(q.get("consistency_score"), 0.5),
        },
        "energy": {
            "work_positive":     abs(_safe(e.get("work_positive"), 10.0)),
            "work_negative":     abs(_safe(e.get("work_negative"),  8.0)),
            "efficiency_pct":    round(min(100.0, max(0.0, _safe(e.get("mechanical_efficiency"), 0.5) * 100)), 1),
            "peak_power_w":      max(0.0, _safe(e.get("peak_power"), 20.0)),
        },
        "damping": {
            "ratio":             _safe(d.get("damping_ratio"), 0.5),
            "control_quality":   d.get("control_quality", "fair"),
            "is_underdamped":    bool(d.get("is_underdamped", False)),
            "is_overdamped":     bool(d.get("is_overdamped", False)),
            "is_critically":     bool(d.get("is_critically_damped", False)),
        },
        "frequency": {
            "dominant_hz":       max(0.01, _safe(f.get("dominant_frequency"), 0.5)),
            "band_power":        {k: max(0.0, _safe(bp.get(k), 0.0)) for k in ["slow","medium","fast","harmonic"]},
            "spectral_entropy":  _safe(f.get("spectral_entropy"), 2.0),
        },
        "harmonic": {
            "oscillation_count": int(_safe(h.get("oscillation_count"), 2)),
            "is_harmonic":       bool(h.get("is_harmonic", False)),
        },
        "waves": [
            {
                "type":          wv.get("type", "transition"),
                "duration_sec":  _safe(wv.get("duration_sec"), 1.0),
                "mean_velocity": _safe(wv.get("mean_velocity"), 0.1),
                "smoothness":    _safe(wv.get("smoothness"),    0.5),
            }
            for wv in w[:8]   # cap at 8 waves
        ],
        "wave_count": len(w),
    }
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Gemini prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_gemini_prompt(clip: dict, features: dict, signal_name: str) -> str:
    """Construct the structured biomechanics prompt for Gemini."""

    exercise    = EXERCISE_DISPLAY.get(
        str(clip.get("exercise", "unknown")).lower(), 
        str(clip.get("exercise", "Unknown"))
    )
    camera      = str(clip.get("CAMERA_POSITION", "SIDE")).upper()
    is_expert   = clip.get("expert", False)
    error_rate  = clip.get("error_rate", [])
    has_error   = bool(error_rate)

    q  = features["quality"]
    e  = features["energy"]
    d  = features["damping"]
    f  = features["frequency"]
    h  = features["harmonic"]
    ws = features["waves"]

    dominant_band = max(
        features["frequency"]["band_power"],
        key=features["frequency"]["band_power"].get
    )

    wave_summary = ""
    if ws:
        wave_lines = [
            f"  - {w['type'].capitalize()}: {w['duration_sec']:.2f}s, "
            f"velocity={w['mean_velocity']:.3f}, smoothness={w['smoothness']:.2f}"
            for w in ws[:4]
        ]
        wave_summary = "\n".join(wave_lines)
    else:
        wave_summary = "  - No distinct wave segments detected"

    # Context about other available signals
    other_signals = [k for k in ["back_trajectory", "knee_angle_trajectory", "arm_trajectory", "core_trajectory"] 
                     if k in clip and k != signal_name]
    
    prompt = f"""Exercise: {exercise}
Camera position: {camera}
Performer level: {"Expert" if is_expert else "Non-expert"}
Primary signal analyzed: {signal_name}
Other recorded metrics: {', '.join(other_signals) if other_signals else 'none'}

--- WAVE PHYSICS METRICS ---
(Based on primary signal: {signal_name})

Movement quality:
  Grade: {q['grade']}  |  Overall score: {q['overall']:.2f}
  Smoothness:   {q['smoothness']:.2f}
  Control:      {q['control']:.2f}
  Efficiency:   {q['efficiency']:.2f}
  Consistency:  {q['consistency']:.2f}

Energy analysis:
  Positive work: {e['work_positive']:.1f} J
  Negative work: {e['work_negative']:.1f} J
  Mechanical efficiency: {e['efficiency_pct']:.1f}%

Damping / movement control:
  Damping ratio ζ = {d['ratio']:.3f}
  Control quality: {d['control_quality']}

Detected wave segments:
{wave_summary}

--- TASK ---
Using your expert knowledge of {exercise} biomechanics from the {camera} view, \
analyze these metrics. Write 1–3 concise sentences. \
{"If you see errors in these numbers, name the specific joint or phase affected." if has_error else "If the form is correct, explain why these metrics confirm it."}"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gemini API call
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    retries: int = 3,
    delay: float = 2.0,
) -> str | None:
    """Call Gemini API and return the annotation string."""
    for attempt in range(retries):
        try:
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_PROMPT,
                    temperature=0.3,
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    [Gemini retry {attempt+1}/{retries}] {e}")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    [Gemini ERROR] {e}")
                return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build Alpaca training example
# ─────────────────────────────────────────────────────────────────────────────

def make_alpaca_examples(
    clip: dict,
    features: dict,
    annotation: str,
    video_id: str,
    split: str,
    signal_name: str,
) -> list[dict]:
    """Generate Alpaca instruction-tuning examples for one clip."""

    exercise   = EXERCISE_DISPLAY.get(
        str(clip.get("exercise", "unknown")).lower(),
        str(clip.get("exercise", "Unknown"))
    )
    camera     = str(clip.get("CAMERA_POSITION", "FRONT")).upper()
    is_expert  = clip.get("expert", False)
    error_rate = clip.get("error_rate", [])
    has_error  = bool(error_rate)

    q = features["quality"]
    e = features["energy"]
    d = features["damping"]
    f = features["frequency"]

    dominant_band = max(
        features["frequency"]["band_power"],
        key=features["frequency"]["band_power"].get
    )

    system_msg = (
        "You are FormWave, an expert AI biomechanics coach. You analyze exercise "
        "form by interpreting temporal waveform signals from joint trajectories. "
        "Provide concise, specific, actionable feedback grounded in motion physics."
    )

    base_input = (
        f"Video: {video_id} | Exercise: {exercise} | "
        f"Camera: {camera} | Performer: {'Expert' if is_expert else 'Non-expert'}"
    )

    examples = []

    # ── Example 1: Primary form annotation (Gemini output as target) ──────────
    examples.append({
        "instruction": (
            f"Analyze the exercise form in this {exercise} clip and describe "
            f"what the person is doing {'incorrectly' if has_error else 'correctly'}."
        ),
        "input": (
            f"{base_input}\n"
            f"Wave quality grade: {q['grade']} | Overall score: {q['overall']:.2f}\n"
            f"Error timestamps: {error_rate if has_error else 'none'}"
        ),
        "output": annotation,
        "system": system_msg,
    })

    # ── Example 2: Energy efficiency ──────────────────────────────────────────
    eff_word = ("excellent" if e['efficiency_pct'] > 80
                else "good" if e['efficiency_pct'] > 65
                else "needs improvement")
    examples.append({
        "instruction": f"What does the energy analysis reveal about this {exercise}?",
        "input": (
            f"{base_input}\n"
            f"Concentric work: {e['work_positive']:.1f} J | "
            f"Eccentric work: {e['work_negative']:.1f} J\n"
            f"Mechanical efficiency: {e['efficiency_pct']:.1f}% | "
            f"Peak power: {e['peak_power_w']:.1f} W"
        ),
        "output": (
            f"Mechanical efficiency of {e['efficiency_pct']:.1f}% is {eff_word}. "
            f"The athlete generated {e['work_positive']:.1f} J concentrically and "
            f"absorbed {e['work_negative']:.1f} J eccentrically, peaking at "
            f"{e['peak_power_w']:.1f} W. "
            + (f"{annotation}" if has_error and e['efficiency_pct'] < 70
               else "Power output is appropriate for the movement pattern.")
        ),
        "system": system_msg,
    })

    # ── Example 3: Tempo and rhythm ───────────────────────────────────────────
    examples.append({
        "instruction": f"Describe the tempo and rhythm of this {exercise} based on wave analysis.",
        "input": (
            f"{base_input}\n"
            f"Dominant frequency: {f['dominant_hz']:.3f} Hz ({dominant_band} tempo)\n"
            f"Oscillations: {features['harmonic']['oscillation_count']} | "
            f"Wave count: {features['wave_count']}"
        ),
        "output": (
            f"The {exercise} runs at {f['dominant_hz']:.3f} Hz ({dominant_band} tempo). "
            f"{features['wave_count']} movement phases were detected across "
            f"{features['harmonic']['oscillation_count']} rep cycles. "
            + (
                "This fast cadence risks losing eccentric control — slow down for safer form."
                if dominant_band == "fast" else
                "Controlled medium-tempo cadence — ideal for muscle activation and joint loading."
                if dominant_band == "medium" else
                "Very slow tempo indicates maximum time-under-tension training."
                if dominant_band == "slow" else
                "High harmonic content — reduce bounce/momentum for cleaner reps."
            )
        ),
        "system": system_msg,
    })

    # ── Example 4: Movement control / damping ─────────────────────────────────
    damping_class = (
        "critically damped (optimal neuromuscular control)"
        if d["is_critically"] else
        "underdamped (slight oscillation — reduce momentum)"
        if d["is_underdamped"] else
        "overdamped (overly restricted — may indicate fatigue or bracing issues)"
    )
    examples.append({
        "instruction": f"Assess the movement control quality in this {exercise}.",
        "input": (
            f"{base_input}\n"
            f"Damping ratio ζ: {d['ratio']:.3f} | Control quality: {d['control_quality']}\n"
            f"Control score: {q['control']:.2f} | Smoothness: {q['smoothness']:.2f}"
        ),
        "output": (
            f"Movement control is {damping_class} (ζ = {d['ratio']:.3f}). "
            f"Control score: {q['control']:.2f}/1.00. Smoothness: {q['smoothness']:.2f}/1.00. "
            + ("Form is mechanically sound — maintain under progressive load."
               if q['control'] > 0.75 else
               "Recommend tempo training (3-1-3 cadence) to improve eccentric control.")
        ),
        "system": system_msg,
    })

    # ── Example 5: Comprehensive report ──────────────────────────────────────
    examples.append({
        "instruction": f"Give a full biomechanical form report for this {exercise} clip.",
        "input": f"{base_input}\nCamera: {camera}",
        "output": (
            f"## {exercise} Form Report — Camera: {camera}\n\n"
            f"**Overall Grade: {q['grade']}** | Score: {q['overall']:.2f}/1.00\n\n"
            f"| Metric | Score |\n|---|---|\n"
            f"| Smoothness | {q['smoothness']:.2f} |\n"
            f"| Control    | {q['control']:.2f} |\n"
            f"| Efficiency | {q['efficiency']:.2f} |\n"
            f"| Consistency| {q['consistency']:.2f} |\n\n"
            f"**Tempo:** {f['dominant_hz']:.3f} Hz ({dominant_band})\n"
            f"**Efficiency:** {e['efficiency_pct']:.1f}% | "
            f"**Peak Power:** {e['peak_power_w']:.1f} W\n\n"
            f"**Assessment:** {annotation}"
        ),
        "system": system_msg,
    })

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# 7. Walk the data directory
# ─────────────────────────────────────────────────────────────────────────────

def walk_dataset(data_dir: Path) -> list[dict]:
    """
    Walk data/{exercise}/{camera}/{split}/annotations/*.json
    and yield records with metadata.
    """
    records = []
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return records

    for exercise_dir in sorted(data_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name

        for camera_dir in sorted(exercise_dir.iterdir()):
            if not camera_dir.is_dir():
                continue
            camera = camera_dir.name

            for split_dir in sorted(camera_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                split = split_dir.name.lower()   # train / val / test

                ann_dir = split_dir / "annotations"
                if not ann_dir.exists():
                    # also check directly in split_dir
                    ann_dir = split_dir

                json_files = sorted(ann_dir.glob("*.json"))
                for jf in json_files:
                    records.append({
                        "json_path": jf,
                        "exercise":  exercise,
                        "camera":    camera,
                        "split":     split,
                        "video_id":  jf.stem,
                    })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    data_dir    = Path(args.data_dir)
    output_dir  = Path(args.output_dir)
    ann_dir_out = output_dir / "annotated"

    output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir_out.mkdir(parents=True, exist_ok=True)

    # ── Setup Gemini ──────────────────────────────────────────────────────────
    dry_run = args.dry_run
    if not dry_run:
        if not GEMINI_AVAILABLE:
            print("[WARN] Gemini unavailable — switching to dry-run mode.")
            dry_run = True
        else:
            api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("[ERROR] No GEMINI_API_KEY found. Set env var or pass --api_key.")
                print("        Running in dry-run mode (no API calls).")
                dry_run = True
            else:
                global _gemini_client
                _gemini_client = genai.Client(api_key=api_key)
                print(f"[OK] Gemini configured — model: {args.model}")

    # ── Walk dataset ──────────────────────────────────────────────────────────
    records = walk_dataset(data_dir)
    if not records:
        print(f"\n[INFO] No annotation JSONs found in {data_dir}")
        print("       Expected structure: data/{exercise}/{camera}/{split}/annotations/*.json")
        print("\n[INFO] Running in DEMO MODE with synthetic sample clips...")
        records = _make_demo_records(output_dir)

    if args.max_clips:
        records = records[:args.max_clips]

    print(f"\n[INFO] Found {len(records)} clips to process")

    # ── Split buckets ─────────────────────────────────────────────────────────
    split_examples: dict[str, list] = {"train": [], "val": [], "test": []}
    log_path = output_dir / "annotation_log.jsonl"
    
    stats = {
        "processed": 0, "skipped": 0,
        "gemini_success": 0, "gemini_failed": 0,
        "dry_run": 0,
    }

    print("=" * 65)
    print("  FormWave — Gemini Annotation Pipeline")
    print("=" * 65)

    with open(log_path, "w") as log_f:
        for i, rec in enumerate(records):
            json_path = rec["json_path"]
            video_id  = rec["video_id"]
            split     = rec.get("split", "train")
            if split not in split_examples:
                split = "train"

            print(f"\n[{i+1}/{len(records)}] {video_id}  ({rec['exercise']} / {rec['camera']} / {split})")

            # Load clip
            clip = load_clip(json_path)
            if clip is None:
                stats["skipped"] += 1
                continue

            # Override exercise/camera from folder structure if not in JSON
            clip.setdefault("exercise",        rec["exercise"])
            clip.setdefault("CAMERA_POSITION", rec["camera"])

            try:
                # Wave physics
                signal, sig_name = get_primary_signal(clip)
                features = run_wave_physics(signal)
                print(f"   Wave: grade={features['quality']['grade']}  "
                      f"eff={features['energy']['efficiency_pct']:.1f}%  "
                      f"waves={features['wave_count']}")

                # Gemini annotation
                if dry_run:
                    has_error = bool(clip.get("error_rate"))
                    annotation = (
                        clip.get("LANGUAGE") or
                        (f"The person demonstrates {'incorrect' if has_error else 'correct'} "
                         f"{EXERCISE_DISPLAY.get(str(clip.get('exercise','')).lower(), 'exercise')} "
                         f"form (grade {features['quality']['grade']}, "
                         f"efficiency {features['energy']['efficiency_pct']:.1f}%).")
                    )
                    stats["dry_run"] += 1
                    print(f"   [DRY RUN] annotation: {annotation[:80]}...")
                else:
                    prompt     = build_gemini_prompt(clip, features, sig_name)
                    annotation = call_gemini(prompt, model_name=args.model)
                    if annotation:
                        stats["gemini_success"] += 1
                        print(f"   [Gemini] {annotation[:100]}...")
                    else:
                        annotation = clip.get("LANGUAGE", "Form analysis unavailable.")
                        stats["gemini_failed"] += 1
                        print(f"   [Gemini FAILED] using fallback annotation")

                # Build training examples
                examples = make_alpaca_examples(
                    clip, features, annotation, video_id, split, sig_name
                )
                split_examples[split].extend(examples)

                # Save enriched annotation JSON
                enriched = {
                    **clip,
                    "LANGUAGE":    annotation,
                    "wave_features": features,
                    "signal_source": sig_name,
                }
                out_path = ann_dir_out / f"{video_id}.json"
                with open(out_path, "w") as f:
                    # Remove large array fields before saving
                    save_clip = {k: v for k, v in enriched.items()
                                 if k not in ("keypoints", "confidence")}
                    json.dump(save_clip, f, indent=2)

                # Audit log
                log_entry = {
                    "video_id":   video_id,
                    "exercise":   rec["exercise"],
                    "camera":     rec["camera"],
                    "split":      split,
                    "annotation": annotation,
                    "grade":      features["quality"]["grade"],
                    "efficiency": features["energy"]["efficiency_pct"],
                }
                log_f.write(json.dumps(log_entry) + "\n")

                stats["processed"] += 1

            except Exception as exc:
                import traceback
                print(f"   [ERROR] {exc}")
                if args.verbose:
                    traceback.print_exc()
                stats["skipped"] += 1
                continue

            # Rate limiting for Gemini
            if not dry_run and i < len(records) - 1:
                time.sleep(args.rate_limit_s)

    # ── Save datasets ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Saving datasets...")

    split_files = {}
    for split_name, examples in split_examples.items():
        if not examples:
            continue
        out = output_dir / f"{split_name}_wave_alpaca.json"
        with open(out, "w") as f:
            json.dump(examples, f, indent=2)
        split_files[split_name] = str(out)
        print(f"  {split_name:<6}: {len(examples):>5} examples → {out.name}")

    # Summary JSON
    summary = {
        **stats,
        "total_examples": sum(len(v) for v in split_examples.values()),
        "split_files": split_files,
        "model": args.model,
        "dry_run": dry_run,
    }
    with open(output_dir / "annotation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  FormWave Annotation Pipeline Complete!")
    print("=" * 65)
    print(f"  Processed : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}")
    if not dry_run:
        print(f"  Gemini OK : {stats['gemini_success']}")
        print(f"  Gemini ❌ : {stats['gemini_failed']}")
    else:
        print(f"  Dry-run   : {stats['dry_run']} (no API calls made)")
    print(f"  Total examples: {summary['total_examples']}")
    print(f"  Audit log : {log_path}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Demo mode — synthetic clips when no data dir exists yet
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_records(output_dir: Path) -> list[dict]:
    """Create synthetic sample clips for pipeline testing."""
    demo_dir = output_dir / "demo_data"
    demo_dir.mkdir(exist_ok=True)

    exercises = [
        ("Push up",  "FRONT", "train",  True,  [], "THE PERSON IS DOING THE PUSH UP CORRECTLY"),
        ("Push up",  "SIDE",  "val",    False, [[1.2, 2.4]], ""),
        ("Squat",    "FRONT", "train",  False, [[0.8, 1.9]], ""),
        ("Squat",    "SIDE",  "train",  True,  [], "THE PERSON IS SQUATTING WITH CORRECT FORM"),
        ("Deadlift", "SIDE",  "train",  False, [[2.0, 3.5]], ""),
        ("OHP",      "FRONT", "val",    False, [[1.5, 2.8]], ""),
    ]

    records = []
    fps = 30

    for i, (exercise, camera, split, expert, error_rate, lang) in enumerate(exercises):
        vid = f"demo_clip_{i+1:04d}"
        t   = np.linspace(0, 8, 8 * fps)
        n   = len(t)
        sig = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(n)

        clip_data = {
            "video_id":        vid,
            "exercise":        exercise,
            "CAMERA_POSITION": camera,
            "expert":          expert,
            "video":           f"{vid}.mp4",
            "trajectory":      sig.tolist(),
            "legs_trajectory": (sig * 0.8 + 0.05 * np.random.randn(n)).tolist(),
            "arm_Trajectory":  (sig * 0.6 + 0.05 * np.random.randn(n)).tolist(),
            "core_":           (sig * 0.4 + 0.03 * np.random.randn(n)).tolist(),
            "error_rate":      error_rate,
            "LANGUAGE":        lang,
        }

        path = demo_dir / f"{vid}.json"
        with open(path, "w") as f:
            json.dump(clip_data, f, indent=2)

        records.append({
            "json_path": path,
            "exercise":  exercise,
            "camera":    camera,
            "split":     split,
            "video_id":  vid,
        })

    print(f"[DEMO] Created {len(records)} synthetic clips in {demo_dir}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FormWave: Gemini-powered biomechanical annotation pipeline"
    )
    p.add_argument("--data_dir",    default="./data",
                   help="Root data directory (default: ./data)")
    p.add_argument("--output_dir",  default="./outputs",
                   help="Output directory (default: ./outputs)")
    p.add_argument("--api_key",     default=None,
                   help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--model",       default="gemini-2.5-flash",
                   help="Gemini model name (default: gemini-2.5-flash)")
    p.add_argument("--max_clips",   type=int, default=None,
                   help="Maximum clips to process (for testing)")
    p.add_argument("--rate_limit_s",type=float, default=0.5,
                   help="Seconds between Gemini calls (default: 0.5)")
    p.add_argument("--dry_run",     action="store_true",
                   help="Skip Gemini API calls, use fallback annotations")
    p.add_argument("--verbose",     action="store_true",
                   help="Print full tracebacks on errors")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
