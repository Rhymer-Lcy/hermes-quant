"""Build the STAR Market IPO lake (issue #15).

Two pieces: the subscription metadata (Eastmoney via akshare -- issue price, listing date,
proceeds) and, per listed name, daily bars from the listing day (BaoStock, raw AND
adjusted: the allottee return needs the raw day-1 open over the issue price, then the
adjusted total-return factor). Resumable; ends with the hard completeness gate.

    conda activate hermes
    python scripts/build_ipo_star_lake.py
"""
from __future__ import annotations

import time

import pandas as pd

from hermes.data.fundamentals import _done_path
from hermes.data.sources import baostock_source as bss
from hermes.io import atomic_to_parquet, atomic_write_text
from hermes.paths import PARQUET_DIR, ensure_dirs

IPO_META_PARQUET = PARQUET_DIR / "ipo_star_meta.parquet"
IPO_BARS_PARQUET = PARQUET_DIR / "ipo_star_bars.parquet"


def pull_meta() -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_xgsglb_em(symbol="全部股票")
    star = raw[raw["板块"] == "科创板"].copy()
    meta = pd.DataFrame({
        "code": "sh." + star["股票代码"].astype(str),
        "name": star["股票简称"],
        "issue_price": pd.to_numeric(star["发行价格"], errors="coerce"),
        "list_date": pd.to_datetime(star["上市日期"], errors="coerce"),
        "shares": pd.to_numeric(star["发行总数"], errors="coerce"),
    }).dropna(subset=["issue_price", "list_date"])
    meta["proceeds"] = meta["issue_price"] * meta["shares"] * 1e4   # 发行总数 arrives in 万股
    meta = (meta.drop_duplicates("code").sort_values("list_date").reset_index(drop=True))
    atomic_to_parquet(meta, IPO_META_PARQUET, index=False)
    return meta


def pull_bars(meta: pd.DataFrame, pause: float = 0.2) -> pd.DataFrame:
    done_path = _done_path("ipo_bars")
    done: set[str] = (set(done_path.read_text(encoding="utf-8").split())
                      if done_path.exists() else set())
    table = (pd.read_parquet(IPO_BARS_PARQUET) if IPO_BARS_PARQUET.exists()
             else pd.DataFrame(columns=["code", "date", "open_raw", "close_raw",
                                        "open_hfq", "close_hfq"]))
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    listed = meta[meta["list_date"] <= end]
    todo = [(c, d) for c, d in zip(listed["code"], listed["list_date"]) if c not in done]
    if done:
        print(f"  resuming: {len(done)} codes already complete, {len(todo)} to go")

    def checkpoint(frames: list[pd.DataFrame]) -> pd.DataFrame:
        objs = [f for f in [table, *frames] if not f.empty]
        merged = pd.concat(objs, ignore_index=True) if objs else table
        merged = (merged.drop_duplicates(["code", "date"], keep="last")
                        .sort_values(["code", "date"]).reset_index(drop=True))
        atomic_to_parquet(merged, IPO_BARS_PARQUET, index=False)
        atomic_write_text("\n".join(sorted(done)), done_path)
        return merged

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    n = len(todo)
    with bss.session():
        for i, (code, ld) in enumerate(todo, 1):
            start = ld.strftime("%Y-%m-%d")
            try:
                for attempt in range(4):
                    try:
                        raw = bss.daily_bars(code, start, end, adjustflag="3")
                        hfq = bss.daily_bars(code, start, end, adjustflag="1")
                        break
                    except Exception as exc:  # noqa: BLE001 -- retry recoverable families
                        msg = str(exc)
                        ok = bss.is_session_error(msg) or bss.is_transport_error(msg)
                        if attempt + 1 >= 4 or not ok:
                            raise
                        time.sleep(2.0 ** attempt)
                        bss.relogin()
                if not raw.empty:
                    r = raw.set_index("date")
                    h = hfq.set_index("date")
                    frames.append(pd.DataFrame({
                        "code": code, "date": r.index,
                        "open_raw": r["open"].to_numpy(),
                        "close_raw": r["close"].to_numpy(),
                        "open_hfq": h["open"].reindex(r.index).to_numpy(),
                        "close_hfq": h["close"].reindex(r.index).to_numpy(),
                    }))
                done.add(code)
            except Exception as exc:  # noqa: BLE001 -- leave the code for the next run
                failed.append(f"{code}: {exc}")
            time.sleep(pause)
            if i % 25 == 0 or i == n:
                table = checkpoint(frames)
                frames = []
                print(f"  ...{i}/{n} names ({len(failed)} failed)")
    table = checkpoint(frames)
    if failed:
        print(f"  {len(failed)} codes failed; first few: {failed[:5]} -- re-run to retry")
    return table


def main() -> None:
    ensure_dirs()
    meta = pull_meta()
    print(f"STAR meta: {len(meta)} listings, {meta['list_date'].min().date()} -> "
          f"{meta['list_date'].max().date()}")

    bars = pull_bars(meta)
    print(f"bars: {len(bars)} rows, {bars['code'].nunique()} names")

    done = set(_done_path("ipo_bars").read_text(encoding="utf-8").split())
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    missing = [c for c in meta.loc[meta["list_date"] <= end, "code"] if c not in done]
    if missing:
        raise SystemExit(f"ABORT: ipo_bars: {len(missing)} codes incomplete "
                         f"(first: {missing[:5]}); re-run to resume")


if __name__ == "__main__":
    main()
