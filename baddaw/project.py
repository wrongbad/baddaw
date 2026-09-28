"""Project root resolution: relative paths are relative to the notebook's directory."""

from __future__ import annotations

import os
from pathlib import Path

from .config import config


def _notebook_path() -> Path | None:
    # jupyter_server exposes the notebook path to the kernel
    p = os.environ.get("JPY_SESSION_NAME")
    if p and Path(p).is_absolute() and Path(p).exists():
        return Path(p)
    # VS Code injects this into the user namespace
    try:
        ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        p = ip.user_ns.get("__vsc_ipynb_file__")
        if p:
            return Path(p)
    except NameError:
        pass
    return None


def root() -> Path:
    if config.root is not None:
        return Path(config.root)
    nb = _notebook_path()
    return nb.parent if nb else Path.cwd()


def resolve(path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else root() / p
