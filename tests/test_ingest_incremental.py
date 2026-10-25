"""Incremental lake refresh: a forward-adjustment re-base must be detected even when the vendor
applies the dividend factor to the earlier history only after the ex-date bar was stored."""
from contextlib import nullcontext

import pandas as pd
import pytest

from hermes.data import ingest

DATES = pd.bdate_range("2026-06-01", periods=12)


def _bars(closes, dates):
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "close": closes,
                         "volume": 1.0})


def _lake(monkeypatch, tmp_path, vendor: pd.DataFrame, stored: pd.DataFrame):
    """Point the refresh at tmp_path with `stored` on disk and a fake vendor serving `vendor`."""
    daily = tmp_path / "daily"
    daily.mkdir()
    stored.to_parquet(daily / "sh_600000.parquet", index=False)
    calls = []

    def fake_pull(code, start, end):
        calls.append(start)
        d = pd.to_datetime(vendor["date"])
        return vendor[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)

    monkeypatch.setattr(ingest, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(ingest, "ensure_dirs", lambda: None)
    monkeypatch.setattr(ingest.bss, "session", nullcontext)
    monkeypatch.setattr(ingest, "_latest_published_date", lambda end: pd.to_datetime(vendor["date"]).max())
    monkeypatch.setattr(ingest, "_daily_bars_reloginning", fake_pull)
    return daily / "sh_600000.parquet", calls


def test_late_dividend_factor_triggers_full_repull(tmp_path, monkeypatch):
    raw = [10.0 + 0.1 * i for i in range(12)]
    # Stored on the OLD basis through day 10: days 8-10 were appended after the ex-date (day 8)
    # while the vendor had not yet rescaled days 1-7.
    stored = _bars(raw[:10], DATES[:10])
    # The vendor has since applied the 0.97 factor to every bar before the ex-date; day 10, the
    # bar a one-bar overlap check compares, is untouched.
    vendor = _bars([c * 0.97 if i < 7 else c for i, c in enumerate(raw)], DATES)
    path, calls = _lake(monkeypatch, tmp_path, vendor, stored)
    summary = ingest.pull_universe_incremental(["sh.600000"], "2026-06-01", "2026-06-30")
    assert summary["mode"].tolist() == ["full:rebased"]
    assert calls[-1] == "2026-06-01"                         # the re-pull starts at `start`
    pd.testing.assert_frame_equal(pd.read_parquet(path), vendor)


def test_unchanged_window_appends_only_new_bars(tmp_path, monkeypatch):
    raw = [10.0 + 0.1 * i for i in range(12)]
    stored = _bars(raw[:10], DATES[:10])
    path, calls = _lake(monkeypatch, tmp_path, _bars(raw, DATES), stored)
    summary = ingest.pull_universe_incremental(["sh.600000"], "2026-06-01", "2026-06-30")
    assert summary["mode"].tolist() == ["append:+2"]
    assert calls == ["2026-06-01"]                           # window = all 10 stored bars (< 60)
    pd.testing.assert_frame_equal(pd.read_parquet(path), _bars(raw, DATES))


def test_window_starts_at_the_nth_last_stored_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REBASE_WINDOW_BARS", 3)
    raw = [10.0 + 0.1 * i for i in range(12)]
    stored = _bars(raw[:10], DATES[:10])
    old_only = _bars([c * 0.97 if i < 7 else c for i, c in enumerate(raw)], DATES)  # days 1-7
    path, calls = _lake(monkeypatch, tmp_path, old_only, stored)
    summary = ingest.pull_universe_incremental(["sh.600000"], "2026-06-01", "2026-06-30")
    assert calls == [DATES[7].strftime("%Y-%m-%d")]          # 3rd-last stored bar = day 8
    assert summary["mode"].tolist() == ["append:+2"]         # ex-date bar stored W-1 bars ago: missed


def test_factor_landing_w_minus_2_bars_late_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REBASE_WINDOW_BARS", 3)
    raw = [10.0 + 0.1 * i for i in range(12)]
    stored = _bars(raw[:10], DATES[:10])
    ex_day9 = _bars([c * 0.97 if i < 8 else c for i, c in enumerate(raw)], DATES)  # ex-date day 9
    path, calls = _lake(monkeypatch, tmp_path, ex_day9, stored)
    summary = ingest.pull_universe_incremental(["sh.600000"], "2026-06-01", "2026-06-30")
    assert summary["mode"].tolist() == ["full:rebased"]      # stored W-2 = 1 bar past the ex-date


def test_window_rebased_rules():
    since = DATES[0]
    s = pd.DataFrame({"date": DATES[:4], "close": [10.0, 10.1, float("nan"), 10.3]})
    same = s.copy()
    assert ingest._window_rebased(s, same, since) is False   # NaN == NaN, the rest identical
    gone = same.drop(index=1)
    assert ingest._window_rebased(s, gone, since) is True    # a stored bar no longer served
    gone_nan = same.drop(index=2)
    assert ingest._window_rebased(s, gone_nan, since) is True   # ...even one whose close was NaN
    filled = same.assign(close=[10.0, 10.1, 10.2, 10.3])
    assert ingest._window_rebased(s, filled, since) is True  # NaN against a number
    noise = same.assign(close=same["close"] + 1e-9)
    assert ingest._window_rebased(s, noise, since) is False  # float noise is not a re-base
    before = same.assign(close=[9.7, 10.1, float("nan"), 10.3])
    assert ingest._window_rebased(s, before, DATES[1]) is False   # a change before `since` is out


@pytest.mark.parametrize("n_stored", [1, 2])
def test_short_history_window_covers_every_stored_bar(tmp_path, monkeypatch, n_stored):
    raw = [10.0 + 0.1 * i for i in range(12)]
    stored = _bars(raw[:n_stored], DATES[:n_stored])
    path, calls = _lake(monkeypatch, tmp_path, _bars(raw, DATES), stored)
    summary = ingest.pull_universe_incremental(["sh.600000"], "2026-06-01", "2026-06-30")
    assert calls == ["2026-06-01"]
    assert summary["mode"].tolist() == [f"append:+{12 - n_stored}"]
