#!/usr/bin/env python3
"""Audit availability of external rlsecd/chronos-sec evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = (
    Path("/Users/shawwalters/Desktop/nca_fun"),
    Path("/Users/shawwalters/Desktop"),
    Path("/Users/shawwalters/src"),
    Path("/Users/shawwalters/Code"),
    Path("/Users/shawwalters/projects"),
)
REQUIRED_REPOS = ("rlsecd", "chronos-sec")


def run_command(args: Sequence[str], timeout_s: float = 15.0) -> dict[str, Any]:
    """Run a command and capture a compact structured result."""
    try:
        proc = subprocess.run(  # noqa: S603
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        return {
            "command": list(args),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "available": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(args),
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {timeout_s}s",
            "available": False,
        }
    return {
        "command": list(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "available": proc.returncode == 0,
    }


def find_local_repos(roots: Sequence[Path]) -> dict[str, list[str]]:
    """Find likely local checkouts under a small set of known workspace roots."""
    found: dict[str, list[str]] = {name: [] for name in (*REQUIRED_REPOS, "security-gym")}
    for root in roots:
        if not root.exists():
            continue
        for name in found:
            for candidate in root.glob(f"**/{name}"):
                if candidate.is_dir():
                    found[name].append(str(candidate))
    return {name: sorted(set(paths)) for name, paths in found.items()}


def audit(roots: Sequence[Path] = DEFAULT_ROOTS) -> dict[str, Any]:
    """Return external availability status for Step 3/4 daemon evidence."""
    local = find_local_repos(roots)
    github_exact = {
        "shawwalters/rlsecd": run_command(
            ["gh", "repo", "view", "shawwalters/rlsecd", "--json", "nameWithOwner,url"]
        ),
        "shawwalters/chronos-sec": run_command(
            [
                "gh",
                "repo",
                "view",
                "shawwalters/chronos-sec",
                "--json",
                "nameWithOwner,url",
            ]
        ),
    }
    github_search = run_command(
        [
            "gh",
            "search",
            "repos",
            "rlsecd",
            "chronos-sec",
            "--json",
            "fullName,url,description",
            "--limit",
            "30",
        ],
        timeout_s=30.0,
    )
    missing = [name for name in REQUIRED_REPOS if not local.get(name)]
    return {
        "schema": "alberta.rlsecd_external_audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_scope": "external_active_defense_daemon_availability",
        "roots": [str(root) for root in roots],
        "local_checkouts": local,
        "github_exact": github_exact,
        "github_search": github_search,
        "required_repos": list(REQUIRED_REPOS),
        "missing_required_repos": missing,
        "security_gym_available": bool(local.get("security-gym")),
        "rlsecd_available": not missing,
        "passed": False,
        "boundary": (
            "security-gym is locally available, but required rlsecd/chronos-sec "
            "daemon repositories and rollout logs are not available to this "
            "framework checkout"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=None)
    parser.add_argument("--write-status", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and optionally write JSON status."""
    args = parse_args(argv)
    roots = tuple(args.root) if args.root is not None else DEFAULT_ROOTS
    status = audit(roots)
    rendered = json.dumps(status, indent=2, sort_keys=True)
    print(rendered)
    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(rendered + "\n", encoding="utf-8")
    return 0 if status["rlsecd_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
