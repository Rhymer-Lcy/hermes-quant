"""Regenerate external/versions.lock — the pinned commit of each editable fork.

We do NOT use git submodules (see docs: PC/cluster fork split + active fork edits).
Instead this lockfile records each fork's HEAD commit so the setup is reproducible.

    python scripts/lock_externals.py
"""
import subprocess

from hermes.paths import EXTERNAL_DIR

UPSTREAM = {
    "vnpy": "https://github.com/vnpy/vnpy.git",
    "vnpy_xt": "https://github.com/vnpy/vnpy_xt.git",
    "vnpy_paperaccount": "https://github.com/vnpy/vnpy_paperaccount.git",
    "rqalpha": "https://github.com/ricequant/rqalpha.git",
}


def _git(repo, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    lines = [
        "# external/versions.lock -- pinned commit of each editable fork in external/.",
        "# Reproduce: clone the upstream repo, `git checkout <commit>`, `pip install -e external/<name>`.",
        "# Regenerate with: python scripts/lock_externals.py",
        "",
    ]
    repos = sorted(p for p in EXTERNAL_DIR.iterdir() if (p / ".git").exists())
    for d in repos:
        name = d.name
        sha = _git(d, "rev-parse", "HEAD")
        try:
            origin = _git(d, "remote", "get-url", "origin")
        except subprocess.CalledProcessError:
            origin = "(none)"
        lines += [
            name,
            f"  origin   {origin}",
            f"  upstream {UPSTREAM.get(name, '(unknown)')}",
            f"  commit   {sha}",
            "",
        ]
    out = EXTERNAL_DIR / "versions.lock"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(repos)} forks)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
