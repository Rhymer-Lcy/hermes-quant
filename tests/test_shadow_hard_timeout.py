"""The D=5 shadow's delisting lookup runs outside the wrapper's idle watchdog, and a BaoStock socket
has no timeout of its own. This drives the REAL client into a real blocked recv() and checks the
PROCESS, not a Python exception: a stand-in server (its own process) accepts the login and never
answers; the shadow driver runs as a separate child process with the lookup deadline shortened to
2 s. The child must exit by itself within a hard wall-clock limit, with exit 75, no surviving
descendant, nothing published under the shadow directory and the canonical directory untouched;
the server must see the connection close only when the child process is gone."""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_shadow_d5.py"
DEADLINE_S = 2.0
HARD_LIMIT_S = 90.0         # interpreter start + imports + the deadline, with a wide margin

STUCK_SERVER = r'''
import socket, sys, threading
log = open(sys.argv[1], "a", buffering=1)
srv = socket.socket()
srv.bind(("127.0.0.1", 0))
srv.listen(8)
print(srv.getsockname()[1], flush=True)

def hold(conn):
    log.write("accepted\n")
    try:
        while conn.recv(65536):            # read the login request, never answer it
            log.write("request\n")
    except OSError:
        pass
    log.write("closed\n")

while True:
    c, _ = srv.accept()
    threading.Thread(target=hold, args=(c,), daemon=True).start()
'''

RUNNER = r'''
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import baostock.common.contants as cons
from hermes.data.sources import baostock_source as bss
from hermes.live import shadow as sh

script, port, shadow_dir, canon_dir, deadline = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("paper_shadow_d5_hang", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def pin():                                 # the real login, against the stand-in server
    cons.BAOSTOCK_SERVER_IP, cons.BAOSTOCK_SERVER_PORT = "127.0.0.1", int(port)
    return "127.0.0.1"

bss._pin_server_ip = pin
codes = [f"sh.6000{i:02d}" for i in range(12)]
dates = pd.bdate_range("2026-06-18", "2026-09-30")
rng = np.random.default_rng(0)
close = pd.DataFrame(10 + rng.random((len(dates), 12)).cumsum(0) * 0.01, index=dates, columns=codes)
signal = pd.DataFrame(np.tile(np.arange(12, 0, -1, dtype=float), (len(dates), 1)),
                      index=dates, columns=codes)
signal.loc[signal.index >= "2026-09-02", codes[10]] = 100.0    # held by the shadow only
close.loc[close.index[-1], codes[10]] = np.nan                  # then no bar: a lookup is needed
for name in ("gate_production_parity", "gate_no_lookahead", "gate_common_state"):
    setattr(mod, name, lambda *a, **k: None)
mod.canonical_at_fork = lambda tiers: {}
mod.build_panels = lambda as_of=None: (close, signal, lambda _d: set(codes))
mod.head_commit = lambda: "0" * 40
prov = {"runtime_code_sha": "0" * 40, "runtime_worktree_clean": True}
mod.runtime_provenance = sh.runtime_provenance = lambda: prov
mod.SHADOW_DIR = sh.SHADOW_DIR = Path(shadow_dir)
sh.PAPER_DIR = Path(canon_dir)             # the real fingerprint guard, on a stand-in tree
mod.TERMINAL_LOOKUP_TIMEOUT_S = float(deadline)
sys.exit(mod.run(["--tiers", "100000", "1000000"]))
'''


def _digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _descendants(pid: int) -> list[str]:
    """Processes the child left behind. POSIX: the child led its own session, so any member of that
    process group is a survivor. Windows: processes whose parent is the child."""
    if os.name == "posix":
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return []
        return [f"process group {pid} still has members"]
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | "
                          "ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.split()


def _lines(path: Path, until: str, wait_s: float = 10.0) -> list[str]:
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        lines = path.read_text().split() if path.exists() else []
        if until in lines:
            return lines
        time.sleep(0.1)
    return path.read_text().split() if path.exists() else []


def test_a_hung_baostock_socket_cannot_hold_the_shadow_process_open(tmp_path):
    log = tmp_path / "server.log"
    server = subprocess.Popen([sys.executable, "-c", STUCK_SERVER, str(log)],
                              stdout=subprocess.PIPE, text=True)
    try:
        port = int(server.stdout.readline())
        canon, shadow = tmp_path / "paper", tmp_path / "d5"
        canon.mkdir()
        (canon / "report_100000.json").write_text('{"equity": 1}', encoding="utf-8")
        before = _digest(canon)
        runner = tmp_path / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUNBUFFERED="1")
        group = ({"start_new_session": True} if os.name == "posix"
                 else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP})
        t0 = time.monotonic()
        child = subprocess.Popen([sys.executable, str(runner), str(SCRIPT), str(port), str(shadow),
                                  str(canon), str(DEADLINE_S)], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, **group)
        try:
            _out, err = child.communicate(timeout=HARD_LIMIT_S)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            pytest.fail(f"the shadow process was still running after {HARD_LIMIT_S} s")
        wall = time.monotonic() - t0

        assert child.returncode == 75, err
        assert f"did not answer within {DEADLINE_S} s" in err and "RETRYABLE" in err
        assert DEADLINE_S <= wall < HARD_LIMIT_S
        assert _descendants(child.pid) == []
        assert not shadow.exists() or not any(shadow.iterdir())     # nothing published
        assert _digest(canon) == before
        events = _lines(log, "closed")
        assert events[:2] == ["accepted", "request"], events      # the real login blocked in recv
        assert "closed" in events                                 # released only by process exit
    finally:
        server.kill()
        server.wait()
