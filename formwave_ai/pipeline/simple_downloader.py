"""Compatibility wrapper for the original step1 downloader.

This file exposes a `run(args)` entry that delegates to the existing
`step1_download_ytd.py` implementation so callers can import a stable name.
"""
from pathlib import Path
import argparse
import step1_download_ytd as legacy


def run(args: argparse.Namespace):
    # Delegate to legacy implementation
    return legacy.run(args)


def parse_args():
    return legacy.parse_args()


if __name__ == "__main__":
    run(parse_args())
