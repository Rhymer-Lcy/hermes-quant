"""Probe which Tushare Pro endpoints this token can call (points/permissions vary).

Establishes ground truth before we build the point-in-time data layer on top of it.

    python scripts/probe_tushare.py
"""
import tushare as ts

from hermes.config import tushare_token

pro = ts.pro_api(tushare_token())


def show(name, fn):
    try:
        df = fn()
        cols = list(df.columns)
        print(f"[OK ] {name}: {len(df)} rows | cols={cols}")
        if len(df):
            print(df.head(3).to_string(index=False))
    except Exception as exc:
        print(f"[ERR] {name}: {exc}")
    print()


# 1) HS300 point-in-time membership (index_weight is monthly).
show("index_weight 000300.SH 2020-01",
     lambda: pro.index_weight(index_code="000300.SH", start_date="20200101", end_date="20200201"))

# 2) Delisted universe (survivorship-bias guard).
show("stock_basic list_status=D (delisted)",
     lambda: pro.stock_basic(exchange="", list_status="D",
                             fields="ts_code,name,list_date,delist_date"))

# 3) Listed + paused counts for the full universe.
show("stock_basic list_status=L (listed)",
     lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,list_date"))
show("stock_basic list_status=P (paused)",
     lambda: pro.stock_basic(exchange="", list_status="P", fields="ts_code,name,list_date"))
