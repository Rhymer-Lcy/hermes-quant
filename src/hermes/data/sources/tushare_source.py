"""Tushare adapter for fields BaoStock lacks -- chiefly market cap (size factor).

daily_basic by ts_code works on this token tier (by trade_date is rate-limited to
1/min; fina_indicator is not permitted). These market-ratio fields are daily as-of
values, so they are point-in-time without announcement-lag concerns.

RETIRED size path: the size factor was tested and REJECTED (A4 -- a size tilt deepens the
drawdown; see docs/risk_control.md), and free-float cap now reconstructs from the BaoStock daily
lake via fl.float_cap with NO Tushare pull. This adapter is therefore unused by the deployed
pipeline and kept only as a reference for a possible higher Tushare tier; there is no size-pull script.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import tushare as ts

from ...config import tushare_token
from ...paths import PARQUET_DIR, ensure_dirs

_PRO = None


def _api() -> Any:
    global _PRO
    if _PRO is None:
        ts.set_token(tushare_token())
        _PRO = ts.pro_api()
    return _PRO


def to_ts_code(bao_code: str) -> str:
    """'sh.600000' -> '600000.SH'; 'sz.000001' -> '000001.SZ'."""
    mkt, num = bao_code.split(".")
    return f"{num}.{'SH' if mkt == 'sh' else 'SZ'}"


def pull_daily_basic(codes, start: str = "20150101", end: str = "20251231",
                     throttle: float = 0.4) -> pd.DataFrame:
    """Per ts_code -> data/parquet/daily_basic/<bao_code>.parquet (total_mv etc.).

    Stores the original BaoStock-style `code` and a parsed `date` so it joins the
    rest of the lake. Records per-symbol status; throttles to respect rate limits.
    """
    ensure_dirs()
    out = PARQUET_DIR / "daily_basic"
    out.mkdir(parents=True, exist_ok=True)
    pro = _api()
    results = []
    n = len(codes)
    for i, bao in enumerate(codes, 1):
        try:
            df = pro.daily_basic(ts_code=to_ts_code(bao), start_date=start, end_date=end,
                                 fields="trade_date,total_mv,circ_mv,pe_ttm,pb")
            if not df.empty:
                df["code"] = bao
                df["date"] = pd.to_datetime(df["trade_date"])
                df.to_parquet(out / f"{bao.replace('.', '_')}.parquet", index=False)
            results.append({"code": bao, "rows": len(df), "status": "ok"})
        except Exception as exc:  # noqa: BLE001 -- record and continue the batch
            results.append({"code": bao, "rows": 0, "status": f"error: {exc}"})
        time.sleep(throttle)
        if i % 50 == 0 or i == n:
            print(f"  ...{i}/{n}")
    return pd.DataFrame(results)
