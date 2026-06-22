"""Operational hardening: atomic writes, the degraded-pull gate, and the publication-completeness guard."""
import pandas as pd
import pytest

from hermes.io import atomic_to_parquet, atomic_write_text
from hermes.live.feed import assert_pull_healthy, latest_coverage


def test_atomic_to_parquet_roundtrip_no_temp_left(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "x.parquet"
    atomic_to_parquet(df, path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
    assert not (tmp_path / "x.parquet.tmp").exists()        # temp swapped away, nothing left behind


def test_atomic_write_text_roundtrip(tmp_path):
    path = tmp_path / "r.json"
    atomic_write_text('{"ok": true}', path)
    assert path.read_text(encoding="utf-8") == '{"ok": true}'
    assert not (tmp_path / "r.json.tmp").exists()


def test_pull_gate_passes_when_healthy():
    summary = pd.DataFrame({"status": ["ok"] * 99 + ["error: timeout"]})
    assert assert_pull_healthy(summary, n_union=100, min_ok_fraction=0.98) == 0.99   # 99% >= 98%


def test_pull_gate_raises_on_degraded_pull():
    summary = pd.DataFrame({"status": ["ok"] * 90 + ["error: timeout"] * 10})        # 90% < 98%
    with pytest.raises(RuntimeError, match="degraded BaoStock pull"):
        assert_pull_healthy(summary, n_union=100, min_ok_fraction=0.98)


def test_pull_gate_raises_on_total_outage():
    summary = pd.DataFrame({"status": ["error: login failed"] * 5})
    with pytest.raises(RuntimeError):
        assert_pull_healthy(summary, n_union=5)


_MEMBERS = ["sh.600000", "sh.600015", "sz.000001"]


def test_latest_coverage_full_publication():
    panel = pd.DataFrame([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
                         index=pd.to_datetime(["2026-06-18", "2026-06-22"]), columns=_MEMBERS)
    latest, cov = latest_coverage(panel, _MEMBERS)
    assert latest == pd.Timestamp("2026-06-22") and cov == 1.0   # all posted -> guard passes


def test_latest_coverage_partial_publication():
    # the latest date is posted for only 1 of 3 current members (today's mis-liquidation scenario)
    panel = pd.DataFrame([[1.0, 2.0, 3.0], [float("nan"), float("nan"), 3.1]],
                         index=pd.to_datetime(["2026-06-18", "2026-06-22"]), columns=_MEMBERS)
    latest, cov = latest_coverage(panel, _MEMBERS)
    assert latest == pd.Timestamp("2026-06-22")
    assert cov == pytest.approx(1 / 3)                            # < 0.90 -> refresh would raise
