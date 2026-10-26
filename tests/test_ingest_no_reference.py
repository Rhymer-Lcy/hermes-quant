"""The incremental refresh no longer takes the market's latest date from one reference stock: each
code is checked on its own, so a stale or suspended sh.600000 cannot freeze the universe, and a
current code is still re-checked for a late adjustment factor."""
from contextlib import nullcontext

import pandas as pd

from hermes.data import ingest

D = pd.bdate_range("2026-06-01", periods=12)
RAW = [10.0 + 0.1 * i for i in range(12)]


def _bars(closes, dates, volume=1.0):
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "close": closes,
                         "volume": volume})


def _lake(monkeypatch, tmp_path, stored: dict, vendor: dict, broken=()):
    daily = tmp_path / "daily"
    daily.mkdir()
    for code, df in stored.items():
        df.to_parquet(daily / f"{code.replace('.', '_')}.parquet", index=False)

    def pull(code, start, end):
        if code in broken:
            raise RuntimeError("BaoStock query failed: 10002007 network receive error")
        v = vendor[code]
        d = pd.to_datetime(v["date"])
        return v[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)

    monkeypatch.setattr(ingest, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(ingest, "ensure_dirs", lambda: None)
    monkeypatch.setattr(ingest.bss, "session", nullcontext)
    monkeypatch.setattr(ingest, "_daily_bars_reloginning", pull)
    return {c: daily / f"{c.replace('.', '_')}.parquet" for c in stored}


def _last(path):
    return pd.read_parquet(path)["date"].max()


def test_reference_stale_but_market_current(tmp_path, monkeypatch):
    stored = {c: _bars(RAW[:11], D[:11]) for c in ("sh.600000", "sh.600036")}
    vendor = {"sh.600000": _bars(RAW[:11], D[:11]), "sh.600036": _bars(RAW, D)}   # reference stops
    paths = _lake(monkeypatch, tmp_path, stored, vendor)
    s = ingest.pull_universe_incremental(["sh.600000", "sh.600036"], "2026-06-01", "2026-06-30")
    assert dict(zip(s["code"], s["mode"])) == {"sh.600000": "append:+0", "sh.600036": "append:+1"}
    assert _last(paths["sh.600036"]) == D[11].strftime("%Y-%m-%d")


def test_reference_suspended_rows_do_not_matter(tmp_path, monkeypatch):
    """BaoStock serves a suspended day as a row with the previous close and no volume."""
    stored = {c: _bars(RAW[:11], D[:11]) for c in ("sh.600000", "sh.600036")}
    suspended = pd.concat([_bars(RAW[:11], D[:11]), _bars([RAW[10]], D[11:], volume=float("nan"))],
                          ignore_index=True)
    vendor = {"sh.600000": suspended, "sh.600036": _bars(RAW, D)}
    paths = _lake(monkeypatch, tmp_path, stored, vendor)
    s = ingest.pull_universe_incremental(["sh.600000", "sh.600036"], "2026-06-01", "2026-06-30")
    assert (s["mode"] == "append:+1").all()
    assert all(_last(p) == D[11].strftime("%Y-%m-%d") for p in paths.values())


def test_source_temporarily_unavailable_for_one_code(tmp_path, monkeypatch):
    stored = {c: _bars(RAW[:11], D[:11]) for c in ("sh.600000", "sh.600036")}
    vendor = {c: _bars(RAW, D) for c in stored}
    paths = _lake(monkeypatch, tmp_path, stored, vendor, broken={"sh.600036"})
    before = paths["sh.600036"].read_bytes()
    s = ingest.pull_universe_incremental(["sh.600000", "sh.600036"], "2026-06-01", "2026-06-30")
    status = dict(zip(s["code"], s["status"]))
    assert status["sh.600000"] == "ok" and status["sh.600036"].startswith("error: BaoStock query failed")
    assert paths["sh.600036"].read_bytes() == before


def test_current_lake_with_a_changed_adjustment_is_rebased(tmp_path, monkeypatch):
    """Previously skipped as 'already current'; now the window catches the late factor."""
    stored = {"sh.600036": _bars(RAW, D)}
    vendor = {"sh.600036": _bars([c * 0.97 if i < 9 else c for i, c in enumerate(RAW)], D)}
    paths = _lake(monkeypatch, tmp_path, stored, vendor)
    s = ingest.pull_universe_incremental(["sh.600036"], "2026-06-01", "2026-06-30")
    assert s["mode"].tolist() == ["full:rebased"]
    pd.testing.assert_frame_equal(pd.read_parquet(paths["sh.600036"]), vendor["sh.600036"])


def test_same_day_rerun_is_idempotent(tmp_path, monkeypatch):
    stored = {c: _bars(RAW, D) for c in ("sh.600000", "sh.600036")}
    paths = _lake(monkeypatch, tmp_path, stored, {c: _bars(RAW, D) for c in stored})
    before = {c: p.read_bytes() for c, p in paths.items()}
    for _ in range(2):
        s = ingest.pull_universe_incremental(list(stored), "2026-06-01", "2026-06-30")
        assert (s["mode"] == "append:+0").all() and (s["status"] == "ok").all()
    assert {c: p.read_bytes() for c, p in paths.items()} == before


def test_weekend_or_holiday_run_writes_nothing(tmp_path, monkeypatch):
    """No session since the last stored bar: the vendor serves nothing newer."""
    stored = {c: _bars(RAW, D) for c in ("sh.600000", "sh.600036")}
    paths = _lake(monkeypatch, tmp_path, stored, {c: _bars(RAW, D) for c in stored})
    before = {c: p.read_bytes() for c, p in paths.items()}
    s = ingest.pull_universe_incremental(list(stored), "2026-06-01", "2026-06-20")   # Saturday
    assert (s["mode"] == "append:+0").all()
    assert {c: p.read_bytes() for c, p in paths.items()} == before
