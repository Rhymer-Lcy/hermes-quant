"""Probe BaoStock for point-in-time HS300 membership + delisted-name history.

If query_hs300_stocks(date) returns historical constituents and we can pull bars
for names later removed/delisted, then BaoStock alone kills survivorship bias for
free (no Tushare index_weight permission needed).

    python scripts/probes/probe_baostock_pit.py
"""
import baostock as bs
import pandas as pd


def rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


lg = bs.login()
print("login:", lg.error_code, lg.error_msg)

for d in ["2015-06-30", "2018-06-29", "2021-06-30", "2024-06-28"]:
    rs = bs.query_hs300_stocks(date=d)
    df = rs_to_df(rs)
    print(f"hs300 @ {d}: err={rs.error_code} n={len(df)} cols={list(df.columns)}")
    if len(df):
        print("   sample:", df.iloc[0].to_dict())

# A name delisted in 2024 (Tushare: 000005.SZ delisted 2024-04-26) -> baostock sz.000005
rs = bs.query_history_k_data_plus(
    "sz.000005", "date,code,close,tradestatus",
    start_date="2023-06-01", end_date="2024-05-10", frequency="d", adjustflag="2")
df = rs_to_df(rs)
print(f"\nsz.000005 (delisted ~2024-04) bars: err={rs.error_code} n={len(df)}")
if len(df):
    print("   first:", df.iloc[0].to_dict())
    print("   last :", df.iloc[-1].to_dict())

bs.logout()
