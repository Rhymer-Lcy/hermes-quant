"""Read the parquet data lake into analysis-ready panels."""
from __future__ import annotations

import pandas as pd

from ..paths import PARQUET_DIR


def load_close_panel(codes: list[str] | None = None, field: str = "close",
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Wide panel of `field` indexed by date, one column per code.

    `codes=None` loads every symbol in data/parquet/daily/. Missing days stay NaN
    (a symbol is untradable on a day it has no bar — suspended or not yet listed).
    """
    daily = PARQUET_DIR / "daily"
    if codes is None:
        files = sorted(daily.glob("*.parquet"))
    else:
        files = [daily / f"{c.replace('.', '_')}.parquet" for c in codes]

    series: dict[str, pd.Series] = {}
    for f in files:
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["date", "code", field])
        if df.empty:
            continue
        code = str(df["code"].iloc[0])
        series[code] = df.set_index("date")[field]

    panel = pd.DataFrame(series).sort_index()
    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    return panel
