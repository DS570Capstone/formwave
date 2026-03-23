"""Download helpers for low-res videos (refactored from data/a_download_lowres.py)."""
from pathlib import Path
import subprocess
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def download_lowres(meta_csv: Path, out_dir: Path, format_spec: str = "worst[height<=360]") -> int:
    meta_csv = Path(meta_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)
    n = 0

    for _, row in df.iterrows():
        vid_id = str(row["id"])
        url = str(row["url"])
        output = out_dir / f"{vid_id}.mp4"

        if output.exists():
            logger.info("Exists: %s", vid_id)
            continue

        cmd = [
            "yt-dlp",
            "--js-runtimes", "node",  # 🔥 FIX
            "-f", format_spec,
            "-o", str(output),
            url,
        ]

        logger.info("Downloading low-res: %s", vid_id)
        subprocess.run(cmd)
        n += 1

    return n