"""CB lake pulls: the JS runtime is warmed on the calling thread before any worker starts, and the
Eastmoney value-analysis response is mapped by field name, never by position."""
import threading

import pandas as pd
import pytest
import requests

from hermes.cb import data as cbdata


def _em_row(date, close, floor, conv, floor_prem, conv_prem):
    """One RPTA_WEB_KZZ_LS record in the endpoint's own key order (15 fields since 2026-10)."""
    return {"DATE": f"{date} 00:00:00", "ZCODE": "113527", "SECURITYVARIETYCODE": "1001746879",
            "SWAPVALUE": conv, "PUREBONDVALUE": floor, "PUREBONDOR": floor_prem,
            "SWAPOR": conv_prem, "FCLOSE": close, "SWAPPRICE": 9.39, "USWAPTVOL": None,
            "USWAPTRATE": None, "SYFE": None, "SECUCODE": "113527.SH",
            "FLOORPRICE_PREMIUM_RATIO_CB": -0.15, "BOND_NAME_ABBR": "x"}


def _serve(monkeypatch, payload=None, status=200):
    calls = []

    class Resp:
        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(f"{status} Server Error")

        def json(self):
            return payload

    def get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return Resp()

    monkeypatch.setattr(requests, "get", get)
    return calls


@pytest.mark.parametrize("key_order", ["endpoint", "reversed"])
def test_pull_premium_maps_fields_by_name(monkeypatch, key_order):
    rows = [_em_row("2025-01-23", 114.831, 114.99, 97.02, -0.14, 18.36),   # newest first
            _em_row("2025-01-22", 114.5, 114.98, 97.76, -0.13, 17.46)]
    rows[0]["SWAPOR"] = None                                                # a null field -> NaN
    if key_order == "reversed":                     # same fields, other order: no positional luck
        rows = [dict(reversed(list(r.items()))) for r in rows]
    calls = _serve(monkeypatch, {"result": {"data": rows}})
    df = cbdata._pull_premium("113527")
    expected = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-22", "2025-01-23"]),
        "close": [114.5, 114.831], "bond_floor": [114.98, 114.99], "conv_value": [97.76, 97.02],
        "bond_premium": [-0.13, -0.14], "conv_premium": [17.46, float("nan")]})
    pd.testing.assert_frame_equal(df, expected)
    assert calls[0]["params"]["filter"] == '(zcode="113527")' and calls[0]["timeout"] == 30


def test_pull_premium_missing_field_fails_loudly(monkeypatch):
    row = _em_row("2025-01-23", 114.831, 114.99, 97.02, -0.14, 18.36)
    del row["SWAPOR"]
    _serve(monkeypatch, {"result": {"data": [row]}})
    with pytest.raises(KeyError, match=r"value-analysis fields missing: \['SWAPOR'\]"):
        cbdata._pull_premium("113527")


@pytest.mark.parametrize("payload", [{"result": None}, {"result": {"data": []}}])
def test_pull_premium_without_rows_is_a_non_retried_miss(monkeypatch, payload):
    _serve(monkeypatch, payload)
    with pytest.raises(ValueError, match="no value-analysis rows"):
        cbdata._pull_premium("126006")


def test_pull_premium_http_error_stays_retryable(monkeypatch):
    _serve(monkeypatch, {}, status=502)
    with pytest.raises(requests.HTTPError):          # not KeyError/ValueError: _retry retries it
        cbdata._pull_premium("113527")


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
