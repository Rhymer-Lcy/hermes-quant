"""Secret/token loading: environment variable first, then .env.local (never committed).

Pattern borrowed from odds-pipeline (config.py). Keeps tokens out of source code
and out of version control.
"""
from __future__ import annotations

import os

from .paths import REPO_ROOT

ENV_LOCAL = REPO_ROOT / ".env.local"


def _load_env_local() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV_LOCAL.exists():
        for raw in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            out[key.strip()] = val.strip().strip('"').strip("'")
    return out


_FILE_ENV = _load_env_local()


def get(key: str, default: str | None = None) -> str | None:
    """Return a secret: real environment wins, then .env.local, then default."""
    return os.environ.get(key) or _FILE_ENV.get(key) or default


def require(key: str) -> str:
    val = get(key)
    if not val:
        raise RuntimeError(
            f"Missing required secret {key!r}. Set it in the environment or {ENV_LOCAL}."
        )
    return val


def tushare_token() -> str | None:
    """Optional — only needed for fields BaoStock lacks. None if unset."""
    return get("TUSHARE_TOKEN")
