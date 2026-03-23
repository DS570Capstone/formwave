"""Download high-res segments based on detection results.

Refactored from data/c_download_highres_segment.py
"""
from pathlib import Path
import json
import subprocess
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def download_highres_segments(meta_csv: Path, segments_json: Path, out_dir: Path, format_spec: str = "best[height<=720]") -> int:
    """Download high-res segments for videos present in segments_json.
    Returns number of downloads attempted.
    """
    meta_csv = Path(meta_csv)
    segments_json = Path(segments_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not segments_json.exists():
        logger.warning("Segments file not found: %s", segments_json)
        return 0

    segments = json.load(open(segments_json))
    df = pd.read_csv(meta_csv)
    count = 0

    for _, row in df.iterrows():
        vid_id = str(row["id"])
        url = str(row["url"])

        if vid_id not in segments:
            continue

        start = segments[vid_id]["start"]
        end = segments[vid_id]["end"]

        output = out_dir / f"{vid_id}_slice.mp4"
        if output.exists():
            logger.info("Exists: %s", output.name)
            continue

        cmd = [
            "yt-dlp",
            "-f", format_spec,
            "--download-sections", f"*{start}-{end}",
            "-o", str(output),
            url,
        ]

        logger.info("Downloading high-res slice: %s", vid_id)
        subprocess.run(cmd)
        count += 1

    logger.info("High-res segment download complete.")
    return count
