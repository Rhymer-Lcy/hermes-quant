"""A canonical write killed by process death, not by a Python exception. A child process builds the
synthetic lake (test_paper_live_gate.Env), commits a 10-23 record, then starts the 10-26 write and is
killed with os._exit at a chosen boundary -- between two whole files, just before the manifest, or
inside an atomic write after its temp file exists but before the rename. The parent checks what the
commit marker says about the directory; a second process then runs normally and must report the
previous state, rewrite every file and commit a consistent manifest.

What this proves is DETECTION and RECOVERY. Seven tiers x three files are not one transaction:
between the first replaced file and the manifest, the directory holds a mix of generations, and a
reader that ignores the manifest can see it."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes.live import paper

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["--tiers", "100000", "1000000"]          # 2 tiers x 3 files = writes 1..6; manifest = 7

CHILD = r'''
import os, sys
from datetime import date
from pathlib import Path
import pytest
sys.path.insert(0, sys.argv[1])
import test_paper_live_gate as g
from hermes.live import paper

work, phase, kill_at, mode, paper_dir = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
mp = pytest.MonkeyPatch()
env = g.Env(work, mp)
if paper_dir != "-":                               # the recovery run reuses the killed run's record
    mp.setattr(paper, "PAPER_DIR", Path(paper_dir))
    mp.setattr(env.mod, "PAPER_DIR", Path(paper_dir))
if phase == "crash":
    env.lake_through(date(2026, 10, 23))
    env.clock(g.at(2026, 10, 23, 18, 30))
    assert env.mod.run(g.TIERS) == 0
    import shutil
    shutil.copytree(env.paper_dir, work / "committed")    # the 10-23 generation, for the parent
env.lake_through(date(2026, 10, 26))
env.clock(g.at(2026, 10, 26, 18, 30))
if phase == "crash":
    real_parquet, real_text, n = paper.atomic_to_parquet, paper.atomic_write_text, [0]

    def die_or(real, half):
        def write(obj, path, *a, **k):
            n[0] += 1
            if n[0] == kill_at:
                if mode == "inside":               # the temp sibling is on disk, the rename never runs
                    half(obj, Path(path), *a, **k)
                os._exit(137)
            return real(obj, path, *a, **k)
        return write

    def half_parquet(df, path, *a, **k):
        df.to_parquet(path.parent / (path.name + ".tmp"), **k)

    def half_text(text, path, *a, **k):
        (path.parent / (path.name + ".tmp")).write_text(text, encoding="utf-8")

    mp.setattr(paper, "atomic_to_parquet", die_or(real_parquet, half_parquet))
    mp.setattr(paper, "atomic_write_text", die_or(real_text, half_text))
sys.exit(env.mod.run(g.TIERS))
'''


def _child(tmp, name, phase, kill_at=0, mode="between", paper_dir="-"):
    work = tmp / name
    work.mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-c", CHILD, str(ROOT / "tests"), str(work), phase,
                           str(kill_at), mode, paper_dir], env=env, capture_output=True, text=True,
                          timeout=300), work / "paper"


@pytest.mark.parametrize("kill_at, mode", [(3, "between"), (4, "inside"), (7, "between"),
                                           (7, "inside")])
def test_a_killed_canonical_write_is_detected_and_the_next_run_heals_it(tmp_path, kill_at, mode):
    crash, paper_dir = _child(tmp_path, "crash", "crash", kill_at, mode)
    assert crash.returncode == 137, crash.stderr[-2000:]
    m = json.loads((paper_dir / paper.CANONICAL_MANIFEST).read_text("utf-8"))
    assert m["as_of"] == "2026-10-23"              # the marker still names the last commit
    state, problems = paper.verify_canonical_manifest(paper_dir)
    assert state == "incomplete" and problems
    committed = paper_dir.parent / "committed"
    order = [f for cap in (100000, 1000000) for f in paper._tier_files(cap)]   # the write order
    replaced, untouched = order[:kill_at - 1], order[kill_at - 1:]
    for name in untouched:                         # never reached: still the committed bytes
        assert (paper_dir / name).read_bytes() == (committed / name).read_bytes(), name
    changed = [n for n in replaced if (paper_dir / n).read_bytes() != (committed / n).read_bytes()]
    assert changed, "the killed run must have replaced at least one file with new content"
    assert problems == [f"{n}: changed since the commit" for n in changed]
    if mode == "inside":
        assert list(paper_dir.glob("*.tmp")), "the half-written temp sibling should remain"

    heal, _ = _child(tmp_path, "heal", "heal", paper_dir=str(paper_dir))
    assert heal.returncode == 0, heal.stderr[-2000:]
    assert "previous canonical write: incomplete" in heal.stdout
    assert paper.verify_canonical_manifest(paper_dir) == ("consistent", [])
    healed = json.loads((paper_dir / paper.CANONICAL_MANIFEST).read_text("utf-8"))
    assert healed["as_of"] == "2026-10-26"
    assert not list(paper_dir.glob("*.tmp"))       # the rewrite replaced every temp sibling
