import json
import os
import tempfile
from pathlib import Path
from datetime import datetime


TRACKING_DIRNAME = "metadata"
TRACKING_FILENAME = "tracking.json"


def _tracking_path(data_dir: Path) -> Path:
    meta = Path(data_dir) / TRACKING_DIRNAME
    meta.mkdir(parents=True, exist_ok=True)
    return meta / TRACKING_FILENAME


def load_tracking(data_dir: str | Path) -> dict:
    p = _tracking_path(Path(data_dir))
    if not p.exists():
        return {"videos": {}, "clips": {}}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        # Corrupted file fallback
        return {"videos": {}, "clips": {}}


def save_tracking(tracking: dict, data_dir: str | Path) -> None:
    p = _tracking_path(Path(data_dir))
    # atomic write: write to temp file then replace
    dirp = p.parent
    fd, tmp = tempfile.mkstemp(dir=dirp)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(tracking, f, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def is_video_downloaded(tracking: dict, video_id: str) -> bool:
    v = tracking.get("videos", {}).get(video_id)
    return bool(v and v.get("downloaded", False))


def mark_video_downloaded(tracking: dict, video_id: str, url: str, file_path: str) -> None:
    videos = tracking.setdefault("videos", {})
    videos[video_id] = {
        "url": url,
        "downloaded": True,
        "file_path": file_path,
        "processed": False,
        "clips_generated": False,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def is_clip_processed(tracking: dict, clip_id: str) -> bool:
    c = tracking.get("clips", {}).get(clip_id)
    return bool(c and c.get("processed", False))


def mark_clip_status(
    tracking: dict,
    clip_id: str,
    *,
    video_id: str | None = None,
    exercise: str | None = None,
    camera: str | None = None,
    split: str | None = None,
    processed: bool | None = None,
    n_frames: int | None = None,
    annotation_path: str | None = None,
    status: str | None = None,
    reason: str | None = None,
) -> None:
    clips = tracking.setdefault("clips", {})
    entry = clips.setdefault(clip_id, {})
    if video_id is not None:
        entry["video_id"] = video_id
    if exercise is not None:
        entry["exercise"] = exercise
    if camera is not None:
        entry["camera"] = camera
    if split is not None:
        entry["split"] = split
    if processed is not None:
        entry["processed"] = bool(processed)
    if n_frames is not None:
        entry["n_frames"] = int(n_frames)
    if annotation_path is not None:
        entry["annotation_path"] = annotation_path
    if status is not None:
        entry["status"] = status
    if reason is not None:
        entry["reason"] = reason
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")


def mark_video_clips_generated(tracking: dict, video_id: str) -> None:
    v = tracking.setdefault("videos", {}).setdefault(video_id, {})
    v["clips_generated"] = True
    v.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")


def reset_tracking(data_dir: str | Path) -> None:
    p = _tracking_path(Path(data_dir))
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass
