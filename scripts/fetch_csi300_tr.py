"""Fetch the official CSI 300 Total Return Index (H00300) and price index (000300).

Source: China Securities Index Co. (中证指数有限公司) via akshare's `stock_zh_index_hist_csindex`,
which reads the index provider's own published series -- not a third-party reconstruction.

H00300 is the benchmark Hermes should be judged against: Hermes trades a PIT HS300 universe on a
forward-adjusted (total-return) price lake, so it collects dividends, and the 000300 PRICE index
does not. Over a window containing the June-July ex-dividend season that difference is material,
which is precisely the window this review covers.

Cached under results/ (gitignored) so the review is reproducible without re-hitting the network.

    conda activate hermes
    python scripts/fetch_csi300_tr.py
"""
from __future__ import annotations

import sys

import pandas as pd

from hermes.io import atomic_to_parquet
from hermes.paths import RESULTS_DIR, ensure_dirs

OUT_DIR = RESULTS_DIR / "benchmarks"
START, END = "20150101", None          # END=None -> today
SYMBOLS = {"H00300": "csi300_total_return", "000300": "csi300_price"}


def fetch(symbol: str, start: str, end: str) -> pd.Series:
    import akshare as ak

    df = ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=start, end_date=end)
    if df is None or df.empty:
        raise SystemExit(f"ABORT: {symbol} returned no rows")
    # Columns are Chinese; select by name rather than position so a schema change fails loudly.
    date_col = next(c for c in df.columns if "日期" in c)
    close_col = next(c for c in df.columns if c.strip() == "收盘")
    s = pd.Series(pd.to_numeric(df[close_col]).to_numpy(),
                  index=pd.to_datetime(df[date_col]), name=symbol).sort_index()
    s = s[s > 0].dropna()
    if s.empty:
        raise SystemExit(f"ABORT: {symbol} had no usable closes")
    return s


def main() -> None:
    ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    end = END or pd.Timestamp.now().strftime("%Y%m%d")
    out = {}
    for sym, label in SYMBOLS.items():
        s = fetch(sym, START, end)
        out[label] = s
        print(f"  {label:22} ({sym}): {len(s):>5} bars  {s.index.min().date()} -> "
              f"{s.index.max().date()}  last={s.iloc[-1]:,.2f}")
    frame = pd.DataFrame(out).sort_index()
    # A total-return index must never fall below its own price index cumulatively from a common
    # start -- if it does, the two series were mismatched or mis-parsed.
    common = frame.dropna()
    if len(common) > 20:
        tr = common["csi300_total_return"].iloc[-1] / common["csi300_total_return"].iloc[0]
        px = common["csi300_price"].iloc[-1] / common["csi300_price"].iloc[0]
        print(f"\n  sanity: since {common.index[0].date()}, TR x{tr:.3f} vs price x{px:.3f} "
              f"-> {'PASS' if tr > px else 'FAIL'}")
        if tr <= px:
            raise SystemExit("ABORT: the total-return series does not dominate the price series")
    atomic_to_parquet(frame, OUT_DIR / "csi300.parquet")
    print(f"\nwrote {OUT_DIR / 'csi300.parquet'}  shape={frame.shape}")


if __name__ == "__main__":
    sys.exit(main())
