"""CB lake pulls: the JS runtime is warmed on the calling thread before any worker starts."""
import threading

import pandas as pd
import pytest

from hermes.cb import data as cbdata


def _recording(monkeypatch):
    events = []
    monkeypatch.setattr(cbdata, "_SLEEP_SEC", 0.0)
    monkeypatch.setattr(cbdata, "_warm_js_runtime",
                        lambda: events.append(("warm", threading.get_ident())))

    def pull(code):
        events.append(("pull", threading.get_ident()))
        return pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "close": [100.0]})

    return events, pull


def test_warm_runs_on_caller_before_any_pull(tmp_path, monkeypatch):
    events, pull = _recording(monkeypatch)
    cbdata._build_per_bond(["A", "B", "C", "D", "E"], tmp_path, pull, "t", max_workers=4)
    assert [k for k, _ in events].count("warm") == 1
    assert events[0] == ("warm", threading.get_ident())
    assert [k for k, _ in events[1:]] == ["pull"] * 5


def test_no_warm_when_single_worker_or_nothing_to_do(tmp_path, monkeypatch):
    events, pull = _recording(monkeypatch)
    cbdata._build_per_bond(["A"], tmp_path, pull, "t", max_workers=1)
    cbdata._build_per_bond(["A"], tmp_path, pull, "t", max_workers=4)   # resumed: A on disk
    assert [k for k, _ in events] == ["pull"]


def test_warm_failure_propagates_before_any_pull(tmp_path, monkeypatch):
    events, pull = _recording(monkeypatch)

    def broken():
        raise RuntimeError("V8 unavailable")

    monkeypatch.setattr(cbdata, "_warm_js_runtime", broken)
    with pytest.raises(RuntimeError, match="V8 unavailable"):
        cbdata._build_per_bond(["A", "B"], tmp_path, pull, "t", max_workers=4)
    assert events == [] and not list(tmp_path.iterdir())


def test_warm_js_runtime_constructs_and_closes():
    pytest.importorskip("py_mini_racer")
    cbdata._warm_js_runtime()


@pytest.mark.parametrize("has_close", [True, False])
def test_warm_js_runtime_closes_only_where_supported(monkeypatch, has_close):
    py_mini_racer = pytest.importorskip("py_mini_racer")
    made = []

    class Runtime:                       # mini-racer >=0.12 has close(); py-mini-racer 0.6 does not
        def __init__(self):
            made.append(self)
            self.closed = False

    if has_close:
        Runtime.close = lambda self: setattr(self, "closed", True)
    monkeypatch.setattr(py_mini_racer, "MiniRacer", Runtime)
    cbdata._warm_js_runtime()
    assert len(made) == 1 and made[0].closed is has_close


def test_warm_js_runtime_propagates_construction_and_close_errors(monkeypatch):
    py_mini_racer = pytest.importorskip("py_mini_racer")

    class NoV8:
        def __init__(self):
            raise OSError("mini_racer.dll not loadable")

    class BadClose:
        def close(self):
            raise RuntimeError("isolate teardown failed")

    monkeypatch.setattr(py_mini_racer, "MiniRacer", NoV8)
    with pytest.raises(OSError, match="not loadable"):
        cbdata._warm_js_runtime()
    monkeypatch.setattr(py_mini_racer, "MiniRacer", BadClose)
    with pytest.raises(RuntimeError, match="teardown failed"):
        cbdata._warm_js_runtime()
