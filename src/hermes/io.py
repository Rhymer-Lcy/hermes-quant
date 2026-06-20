"""Atomic on-disk writes: write to a temp sibling, then os.replace() onto the final path.

os.replace() is an atomic same-volume rename on NTFS and POSIX, so a crash, scheduler timeout,
or machine sleep mid-write can never leave a truncated file that wedges every later read. This
matters for the unattended daily driver, whose ~660-name pull is a multi-minute write window.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def atomic_to_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write `df` to `path` as parquet atomically (temp file + os.replace)."""
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str | Path, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (temp file + os.replace)."""
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)
