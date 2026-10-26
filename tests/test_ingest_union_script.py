"""scripts/ingest_union.py must fail closed: no implicit end date, no write that drops stored bars,
and a nonzero exit whenever any code is not written. Runs against a tmp lake and a fake vendor."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from hermes.data import ingest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_union.py"
D = pd.bdate_range("2025-12-24", periods=12)     # straddles the BACKTEST_END year-end
RAW = [10.0 + 0.1 * i for i in range(12)]


def _bars(closes, dates):
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "close": closes,
                         "volume": 1.0})


@pytest.fixture
def lake(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ingest_union_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.Series(["sh.600036"], name="code").to_csv(raw_dir / "hs300_union.csv", index=False)
    daily = tmp_path / "daily"
    daily.mkdir()
    path = daily / "sh_600036.parquet"
    _bars(RAW, D).to_parquet(path, index=False)
    vendor = _bars(RAW, D)
    pulls = []

    def pull(code, start, end):
        pulls.append((start, end))
        d = pd.to_datetime(vendor["date"])
        return vendor[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)

    monkeypatch.setattr(mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ingest, "RAW_DIR", raw_dir)
    monkeypatch.setattr(ingest, "PARQUET_DIR", tmp_path)
    monkeypatch.setattr(ingest, "ensure_dirs", lambda: None)
    monkeypatch.setattr(ingest.bss, "session", nullcontext)
    monkeypatch.setattr(ingest, "_daily_bars_reloginning", pull)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    return mod, path, pulls


@pytest.mark.parametrize("argv", [[], ["--end"], ["--end", "2026-13-01"], ["--end", "20261026"],
                                  ["--end", "2026-01-01", "--start", "2026-01-02"]])
def test_missing_or_invalid_arguments_exit_2_and_touch_nothing(lake, argv):
    mod, path, pulls = lake
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    assert pulls == [] and path.read_bytes() == before


def test_an_end_before_the_stored_data_is_refused(lake):
    mod, path, pulls = lake
    before = path.read_bytes()
    assert mod.main(["--end", "2025-12-31"]) == 1             # the old implicit default, now explicit
    assert pulls == [("2015-01-01", "2025-12-31")] and path.read_bytes() == before


def test_a_current_end_rewrites_and_exits_0(lake):
    mod, path, pulls = lake
    assert mod.main(["--end", "2026-01-31"]) == 0
    assert pulls == [("2015-01-01", "2026-01-31")]
    pd.testing.assert_frame_equal(pd.read_parquet(path), _bars(RAW, D))


def test_an_empty_pull_of_a_stored_code_is_a_failure(lake, monkeypatch):
    mod, path, pulls = lake
    monkeypatch.setattr(ingest, "_daily_bars_reloginning", lambda code, start, end: _bars([], []))
    before = path.read_bytes()
    assert mod.main(["--end", "2026-01-31"]) == 1
    assert path.read_bytes() == before
