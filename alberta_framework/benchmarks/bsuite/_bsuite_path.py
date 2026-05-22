"""Path bootstrap helpers for local bsuite checkouts.

DeepMind bsuite is not reliably installable from PyPI on Python 3.13, so the
benchmark scripts support a sibling or local source checkout. This module keeps
that workaround explicit and testable without importing bsuite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root_from_file() -> Path:
    """Return the alberta-framework repository root."""
    return Path(__file__).resolve().parents[2]


def candidate_bsuite_paths(project_root: Path | None = None) -> list[Path]:
    """Return candidate directories that may contain a bsuite checkout.

    Parameters
    ----------
    project_root : Path, optional
        Repository root. Defaults to the root inferred from this file.

    Returns
    -------
    list of Path
        Candidate checkout roots, ordered by preference.
    """
    root = project_root or project_root_from_file()
    candidates: list[Path] = []

    env_path = os.environ.get("BSUITE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            root / "bsuite",
            root / "external" / "bsuite",
            root / "third_party" / "bsuite",
            root.parent / "bsuite",
        ]
    )
    return candidates


def import_root_for_bsuite_checkout(path: Path) -> Path | None:
    """Return the sys.path entry for a candidate bsuite checkout, if valid."""
    candidate = path.expanduser().resolve()

    # Standard checkout layout: <checkout>/bsuite/__init__.py
    if (candidate / "bsuite" / "__init__.py").is_file():
        return candidate

    # Package directory layout: <parent>/bsuite/__init__.py
    if candidate.name == "bsuite" and (candidate / "__init__.py").is_file():
        return candidate.parent

    return None


def add_bsuite_to_path(project_root: Path | None = None) -> Path | None:
    """Prepend a local bsuite checkout to ``sys.path`` when one is present.

    The function performs no imports, so it is safe to call before any
    ``import bsuite`` statement and safe to test without bsuite installed.

    Returns
    -------
    Path or None
        The path inserted into ``sys.path``, or None if no checkout was found.
    """
    for candidate in candidate_bsuite_paths(project_root):
        import_root = import_root_for_bsuite_checkout(candidate)
        if import_root is None:
            continue

        path_str = str(import_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            return import_root
        return import_root

    return None


def bsuite_missing_message() -> str:
    """Return actionable install guidance for missing bsuite imports."""
    return (
        "bsuite is required for benchmarks/bsuite. Install the local extras "
        "with `pip install -e '.[bsuite]'` and place a bsuite source checkout "
        "at `../bsuite`, `./bsuite`, `./external/bsuite`, or set BSUITE_PATH."
    )
