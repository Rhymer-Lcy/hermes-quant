"""Probe Tushare fundamental endpoints needed for factors: daily_basic (PB/PE/mv)
and fina_indicator (ROE, with ann_date for point-in-time alignment).

Establishes permission + rate-limit ground truth before building the factor layer.
Value factors (pe/pb/ps) are already in the BaoStock lake; Tushare adds size
(total_mv) and quality (ROE).

    python scripts/probes/probe_tushare_fundamentals.py
"""
import time

import tushare as ts

from hermes.config import tushare_token

ts.set_token(tushare_token())
pro = ts.pro_api()


def show(name, fn):
    try:
        df = fn()
        print(f"[OK ] {name}: {len(df)} rows | cols={list(df.columns)}")
        if len(df):
            print(df.head(2).to_string(index=False))
    except Exception as exc:
        print(f"[ERR] {name}: {exc}")
    print()


# daily_basic by ts_code (fewer calls than by trade_date for a full-history pull)
show("daily_basic 600000.SH (by ts_code)",
     lambda: pro.daily_basic(ts_code="600000.SH", start_date="20240101", end_date="20240131",
                             fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,turnover_rate,total_mv,circ_mv"))
time.sleep(1.5)

# daily_basic by trade_date (whole cross-section in one call)
show("daily_basic 20240102 (by trade_date)",
     lambda: pro.daily_basic(trade_date="20240102", fields="ts_code,trade_date,pe_ttm,pb,total_mv"))
time.sleep(1.5)

# fina_indicator: ROE + ann_date (announcement date = the PIT key)
show("fina_indicator 600000.SH (ROE + ann_date)",
     lambda: pro.fina_indicator(ts_code="600000.SH", start_date="20200101", end_date="20241231",
                                fields="ts_code,ann_date,end_date,roe,roe_dt,grossprofit_margin,debt_to_assets"))
time.sleep(1.5)

# pro_bar qfq (set_token now done globally, fixing the earlier api-init error)
show("pro_bar 600000.SH qfq",
     lambda: ts.pro_bar(ts_code="600000.SH", adj="qfq", start_date="20240102", end_date="20240110"))
