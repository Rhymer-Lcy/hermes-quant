"""No lake write may drop a bar the lake already holds: full pulls with an old `end` (the research
builders default to BACKTEST_END), a vendor gap, or an incremental refresh with a past `end` must
leave the stored file byte-for-byte unchanged and report the code as an error."""
from contextlib import nullcontext

import pandas as pd

from hermes.data import ingest

D = pd.bdate_range("2026-06-01", periods=12)
RAW = [10.0 + 0.1 * i for i in range(12)]


def _bars(closes, dates):
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "close": closes,
                         "volume": 1.0})


def _lake(monkeypatch, tmp_path, stored, vendor):
    daily = tmp_path / "daily"
    daily.mkdir()
    path = daily / "sh_600036.parquet"
    stored.to_parquet(path, index=False)

    def pull(code, start, end):
        d = pd.to_datetime(vendor["date"])
        return vendor[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)

    monkeypatch.setattr(ingest, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(ingest, "ensure_dirs", lambda: None)
    monkeypatch.setattr(ingest.bss, "session", nullcontext)
    monkeypatch.setattr(ingest, "_daily_bars_reloginning", pull)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    return path


def test_full_pull_with_an_old_end_refuses_and_keeps_the_file(tmp_path, monkeypatch):
    path = _lake(monkeypatch, tmp_path, _bars(RAW, D), _bars(RAW, D))
    before = path.read_bytes()
    s = ingest.pull_universe(["sh.600036"], "2026-06-01", D[5].strftime("%Y-%m-%d"))
    assert s["status"].iloc[0].startswith("error: refused: the pulled series lacks 6 stored bar(s)")
    assert path.read_bytes() == before


def test_full_pull_with_a_vendor_gap_refuses(tmp_path, monkeypatch):
    gap = _bars(RAW, D).drop(index=4).reset_index(drop=True)
    path = _lake(monkeypatch, tmp_path, _bars(RAW, D), gap)
    before = path.read_bytes()
    s = ingest.pull_universe(["sh.600036"], "2026-06-01", "2026-06-30")
    assert "lacks 1 stored bar(s)" in s["status"].iloc[0] and path.read_bytes() == before


def test_full_pull_that_extends_the_series_is_written(tmp_path, monkeypatch):
    path = _lake(monkeypatch, tmp_path, _bars(RAW[:10], D[:10]), _bars(RAW, D))
    s = ingest.pull_universe(["sh.600036"], "2026-06-01", "2026-06-30")
    assert s["status"].tolist() == ["ok"]
    pd.testing.assert_frame_equal(pd.read_parquet(path), _bars(RAW, D))


def test_incremental_refresh_with_a_past_end_never_truncates(tmp_path, monkeypatch):
    """paper_live --as-of without --no-refresh: stored bars after `end` are out of scope."""
    path = _lake(monkeypatch, tmp_path, _bars(RAW, D), _bars(RAW, D))
    before = path.read_bytes()
    s = ingest.pull_universe_incremental(["sh.600036"], "2026-06-01", D[7].strftime("%Y-%m-%d"))
    assert s["mode"].tolist() == ["append:+0"] and (s["status"] == "ok").all()
    assert path.read_bytes() == before


def test_a_real_rebase_with_a_past_end_refuses_rather_than_truncating(tmp_path, monkeypatch):
    rebased = _bars([c * 0.97 if i < 5 else c for i, c in enumerate(RAW)], D)
    path = _lake(monkeypatch, tmp_path, _bars(RAW, D), rebased)
    before = path.read_bytes()
    s = ingest.pull_universe_incremental(["sh.600036"], "2026-06-01", D[7].strftime("%Y-%m-%d"))
    assert s["status"].iloc[0].startswith("error: refused") and path.read_bytes() == before
