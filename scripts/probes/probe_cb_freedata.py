"""Probe free convertible-bond (CB) data for a survivorship-free double-low study.

Kill-first gate for the CB lake (docs/cb_lake.md): the pre-registered question --
"did double-low survive the 2022-08 exchange trading rules and the 2023 credit
events, net of costs, on a survivorship-free universe?" -- is only worth building
a lake for if FREE sources can reconstruct, point-in-time:

  A. the full universe incl. matured/forced-called/defaulted bonds, with the
     listing date and the underlying-stock code (Eastmoney listing table);
  B. daily bars for bonds that no longer trade (Sina serves the whole life);
  C. the conversion-price trail, either as Eastmoney's dated conversion-value /
     premium series (conversion price implied point-in-time) or as JSL's
     explicit downward-revision log (no login).

Any hard failure kills the lake before it is built. Names probed: two 2023
credit-default delistings (Soute 128100, Landun 123015), one forced exit
(Huifeng 128012), one classic forced call (Taijing 113503), plus the largest
recently-listed bond as the still-alive freshness check.

    python scripts/probes/probe_cb_freedata.py
"""
import akshare as ak
import pandas as pd

DEAD = {"搜特": None, "蓝盾": None, "辉丰": None, "泰晶": None}


def sina_symbol(code: str) -> str:
    return ("sh" if code.startswith("11") else "sz") + code


def main() -> None:
    # --- Gate A: survivorship-free universe -------------------------------------
    cov = ak.bond_zh_cov()
    cov["上市时间"] = pd.to_datetime(cov["上市时间"], errors="coerce")
    listed = cov[cov["上市时间"].notna()].copy()
    by18 = int((listed["上市时间"] <= "2018-12-31").sum())
    by21 = int((listed["上市时间"] <= "2021-12-31").sum())
    print(f"A universe: {len(listed)} listed bonds, earliest {listed['上市时间'].min().date()}, "
          f"{by18} by end-2018, {by21} by end-2021")
    for nm in DEAD:
        hit = listed[listed["债券简称"].str.contains(nm, na=False)
                     & ~listed["债券代码"].str.startswith("404")]
        DEAD[nm] = hit.iloc[0]["债券代码"] if len(hit) else None
        print(f"   dead-name lookup {nm}: {DEAD[nm]}")
    print(f"   fields: {list(cov.columns)}")
    print(f"   GATE A {'PASS' if all(DEAD.values()) and by21 > 300 else 'FAIL'}\n")

    # --- Gate B: full daily bars for bonds that no longer trade ------------------
    ok_b = True
    for nm, code in DEAD.items():
        df = ak.bond_zh_hs_cov_daily(symbol=sina_symbol(code))
        ok_b &= len(df) > 200
        print(f"B {nm} {code}: {len(df)} bars {df['date'].min()} -> {df['date'].max()} "
              f"cols={list(df.columns)}")
    # Post-delisting segment (404xxx): a fact-check for the terminal-value convention.
    try:
        df404 = ak.bond_zh_hs_cov_daily(symbol="sz404002")
        print(f"   404002 (Soute, delisted segment): {len(df404)} bars "
              f"{df404['date'].min()} -> {df404['date'].max()}")
    except Exception as exc:
        print(f"   404002 (Soute, delisted segment): unavailable ({type(exc).__name__})")
    print(f"   GATE B {'PASS' if ok_b else 'FAIL'}\n")

    # --- Gate C2: Eastmoney dated conversion-value / premium series --------------
    recent = listed[(listed["上市时间"] < pd.Timestamp.today() - pd.Timedelta(days=30))
                    & (listed["上市时间"] > pd.Timestamp.today() - pd.Timedelta(days=400))]
    live_code = recent.sort_values("发行规模", ascending=False).iloc[0]["债券代码"]
    ok_c2, fresh = True, None
    for tag, code in list(DEAD.items()) + [("live", live_code)]:
        df = ak.bond_zh_cov_value_analysis(symbol=code)
        ok_c2 &= len(df) > 100 and "转股溢价率" in df.columns
        if tag == "live":
            fresh = (pd.Timestamp.today() - pd.to_datetime(df["日期"].max())).days
        print(f"C2 {tag} {code}: {len(df)} rows {df['日期'].min()} -> {df['日期'].max()}")
    ok_c2 &= fresh is not None and fresh < 10
    print(f"   live-bond staleness: {fresh} days")
    print(f"   GATE C2 {'PASS' if ok_c2 else 'FAIL'}\n")

    # --- Gate C1: JSL downward-revision log, no login -----------------------------
    ok_c1 = False
    for nm in ["蓝盾", "搜特", "辉丰"]:
        try:
            df = ak.bond_cb_adj_logs_jsl(symbol=DEAD[nm])
            ok_c1 |= len(df) >= 1
            print(f"C1 {nm}: {len(df)} revisions, cols={list(df.columns)}")
        except Exception as exc:
            print(f"C1 {nm}: unavailable ({type(exc).__name__}: {exc})")
    print(f"   GATE C1 {'PASS' if ok_c1 else 'FAIL'}\n")

    go = all(DEAD.values()) and ok_b and (ok_c2 or ok_c1)
    print(f"VERDICT: {'GO -- build the lake' if go else 'KILL -- record and stop'}")


if __name__ == "__main__":
    main()
