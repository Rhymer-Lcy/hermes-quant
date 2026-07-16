"""Shenwan sector index closes, for the conviction-sector creed study (issue #8).

One small lake: ``sw_indices.parquet`` -- daily closes, one column per index, pulled from the
exchange-published Shenwan series via AKShare, plus the HS300 index close (BaoStock) as the
study's market-alternative leg. Price indices (no dividends) -- the study credits a frozen
+2%/yr on top as the deliberately generous variant.
"""
from __future__ import annotations

import time

import pandas as pd

from ..paths import PARQUET_DIR

SW_INDICES_PARQUET = PARQUET_DIR / "sw_indices.parquet"

# The 31 Shenwan level-1 sector indices (the issue #8 robustness set).
SW1_CODES = [
    "801010", "801030", "801040", "801050", "801080", "801110", "801120", "801130",
    "801140", "801150", "801160", "801170", "801180", "801200", "801210", "801230",
    "801710", "801720", "801730", "801740", "801750", "801760", "801770", "801780",
    "801790", "801880", "801890", "801950", "801960", "801970", "801980",
]
# The friend's named exposures (issue #8 primary set): semiconductors (SW2),
# defense/aerospace, nonferrous metals.
PRIMARY_CODES = ["801081", "801740", "801050"]
HS300 = "sh.000300"


def pull_sw_indices(pause: float = 0.5) -> pd.DataFrame:
    """Pull every SW1 index plus the SW2 semiconductor index and the HS300 close ->
    sw_indices.parquet. Wide frame: date index, one column per code."""
    import akshare as ak

    from ..io import atomic_to_parquet
    from .sources import baostock_source as bss

    series: dict[str, pd.Series] = {}
    for code in sorted(set(SW1_CODES + PRIMARY_CODES)):
        df = ak.index_hist_sw(symbol=code, period="day")
        df.columns = ["code", "date", "close", "open", "high", "low", "volume", "amount"]
        s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(),
                      index=pd.to_datetime(df["date"]), name=code)
        series[code] = s[~s.index.duplicated(keep="last")].sort_index()
        time.sleep(pause)
    with bss.session():
        hs = bss.index_close(HS300, "2000-01-01",
                             pd.Timestamp.now().strftime("%Y-%m-%d"))
    series[HS300] = hs
    panel = pd.DataFrame(series).sort_index()
    atomic_to_parquet(panel, SW_INDICES_PARQUET)
    return panel


def load_sw_indices() -> pd.DataFrame:
    return pd.read_parquet(SW_INDICES_PARQUET)
