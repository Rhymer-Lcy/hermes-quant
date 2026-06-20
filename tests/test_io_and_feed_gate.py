"""Operational hardening: atomic writes + the degraded-pull fail-loud gate."""
import pandas as pd
import pytest

from hermes.io import atomic_to_parquet, atomic_write_text
from hermes.live.feed import assert_pull_healthy


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
