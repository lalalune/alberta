#!/usr/bin/env python3
"""Aggregate accepted-scope completion gate for Alberta Plan Steps 3-5.

This gate intentionally separates the accepted project scope from broader
research boundaries:

* Step 3 is accepted for given-feature GVF/Horde prediction.
* Step 4 is accepted for the Step 4a SARSA local control path.
* Step 5 is accepted for the local average-reward research scope.

The detailed Step 3 and Step 4 gates still report their broader open
boundaries, including external rlsecd evidence and actor-critic promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step3_solution_gate import audit_step3
from step4_solution_gate import audit_step4
from step5_solution_gate import audit_step5

DEFAULT_ROOT = Path(".")


def audit_steps3_5(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Return the accepted-scope completion status for Steps 3-5."""
    step3 = audit_step3(root)
    step4 = audit_step4(root)
    step5 = audit_step5(root)

    step3_accepted = bool(step3.get("accepted_given_feature_step3"))
    step4_accepted = bool(
        step4.get("accepted_sarsa_step4a")
        and step4.get("accepted_step4_local_framework_scope")
    )
    step5_accepted = bool(step5.get("solved_step5_full_research_scope"))

    accepted_scope_complete = step3_accepted and step4_accepted and step5_accepted

    return {
        "schema": "alberta.steps3_5.accepted_completion_gate.v1",
        "claim_scope": (
            "accepted_project_scope: Step 3 given-feature GVF/Horde, "
            "Step 4a SARSA local control, Step 5 local average-reward scope"
        ),
        "accepted_scope_complete": accepted_scope_complete,
        "steps": {
            "step3": {
                "accepted": step3_accepted,
                "claim_scope": step3.get("claim_scope"),
                "full_research_scope_solved": step3.get(
                    "solved_step3_full_research_scope"
                ),
                "open_boundaries": step3.get("open_boundaries", []),
            },
            "step4": {
                "accepted": step4_accepted,
                "claim_scope": step4.get("claim_scope"),
                "full_actor_critic_scope_solved": step4.get(
                    "solved_step4_full_actor_critic_scope"
                ),
                "open_boundaries": step4.get("open_boundaries", []),
            },
            "step5": {
                "accepted": step5_accepted,
                "claim_scope": step5.get("claim_scope"),
                "full_research_scope_solved": step5.get(
                    "solved_step5_full_research_scope"
                ),
                "missing_full_evidence": step5.get("missing_full_evidence", []),
            },
        },
        "not_claimed": [
            "Step 3 external rlsecd/chronos-sec daemon closure",
            "Step 4b actor-critic promotion over SARSA on seeded bsuite/control evidence",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root containing evidence artifacts.",
    )
    parser.add_argument(
        "--write-status",
        type=Path,
        default=None,
        help="Optional path to write the JSON status artifact.",
    )
    args = parser.parse_args(argv)

    status = audit_steps3_5(args.root)
    text = json.dumps(status, indent=2, sort_keys=True)
    print(text)

    if args.write_status is not None:
        args.write_status.parent.mkdir(parents=True, exist_ok=True)
        args.write_status.write_text(text + "\n", encoding="utf-8")

    return 0 if status["accepted_scope_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
