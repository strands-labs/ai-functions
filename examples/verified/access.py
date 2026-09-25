"""Decide if a user has can export data based on a policy.

A company's data-export policy is written in Lean (`lean/Access.lean`): who may
export restricted data, and through which delegations of access. The agent looks
up the user, the dataset and the grants, decides whether the export is allowed,
and proves the decision from the policy, whichever way it goes.

The policy involves several dishomogeneous conditions. The agent has to prove it
explored all possibilities and corned cases fo the policy before returning the answer.

Demonstrates:
- A decision proved in both directions, allowed and denied
- Tools that return complete records, so a proof can rule possibilities out
- An agent deciding which facts to fetch as it goes
"""

import asyncio
from pathlib import Path

from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject(Path(__file__).parent / "lean", imports=["Access"])
Access = project.symbols.Access


# A tiny identity-and-access world. Days are absolute day numbers; the example
# asks on day 730. customer-churn-v3 is restricted and owned by dana:
#
#   dana (owner) ──▶ alex (exp 900) ──▶ riley  (exp 800)   2 hops: within policy
#                        └──────────▶ morgan (exp 850) ──▶ casey (exp 900)
#                                                          3 hops: over the limit
#   jordan ──▶ casey (exp 100)                             expired long ago
#
# So riley is allowed, and casey is denied, but only because casey's one live
# chain is a hop too long, a fact no single lookup reveals.
DATASET = "customer-churn-v3"
DIRECTORY: dict[str, tuple[str, bool]] = {
    "riley": ("engineer", True),
    "casey": ("analyst", True),
    "alex": ("engineer", True),
    "morgan": ("engineer", True),
    "jordan": ("engineer", False),
    "dana": ("manager", True),
}
GRANTS: dict[tuple[str, str], list[tuple[str, int]]] = {
    ("riley", DATASET): [("alex", 800)],
    ("alex", DATASET): [("dana", 900)],
    ("morgan", DATASET): [("alex", 850)],
    ("casey", DATASET): [("jordan", 100), ("morgan", 900)],
}
OWNER = {DATASET: "dana"}
CLASSIFICATION = {DATASET: 2}
TRAINING_EXPIRY = {"riley": 950, "casey": 900, "alex": 940, "morgan": 935, "dana": 980}


def known_user(user: str) -> str:
    if user not in DIRECTORY:
        raise ValueError(f"unknown user {user!r}")
    return user


def known_dataset(dataset: str) -> str:
    if dataset not in OWNER:
        raise ValueError(f"unknown dataset {dataset!r}")
    return dataset


@verified.tool(Access.employment)
def directory(user: str) -> tuple[str, bool]:
    """Look up a user's complete directory entry as `(role, active)`.

    `role` is the job role (e.g. "engineer", "contractor"); `active` is whether
    the person is currently employed. Unknown users are refused.
    """
    return DIRECTORY[known_user(user)]


@verified.tool(Access.grantsFor)
def grants_for(user: str, dataset: str) -> list[tuple[str, int]]:
    """List ALL grants `user` holds on `dataset`, as `(delegated_by, expires_day)`.

    The list is complete: a grant not in it does not exist. `expires_day` is the
    last day the grant is in force.
    """
    return GRANTS.get((known_user(user), known_dataset(dataset)), [])


@verified.tool(Access.ownerOf)
def owner_of(dataset: str) -> str:
    """Return the username of the dataset's owner. Owners hold access outright."""
    return OWNER[known_dataset(dataset)]


@verified.tool(Access.classification)
def classification_of(dataset: str) -> int:
    """Return the dataset's classification level: 0 public, 1 internal, 2 restricted."""
    return CLASSIFICATION[known_dataset(dataset)]


@verified.tool(Access.trainingExpiry)
def training_expiry(user: str) -> int:
    """Return the last day the user's data-handling training is valid (0 = never taken)."""
    return TRAINING_EXPIRY.get(known_user(user), 0)


# The contract `Access.DecisionCorrect` demands the decision be proved correct
# whichever way it goes: `decision = true ↔ MayExport user dataset today`. There
# is no strategy guidance: the policy and the chain lemmas are in the project
# source the agent is shown, and choosing which facts to fetch is the task.
@verified.ai_function(
    contract=Access.DecisionCorrect,
    tools=[directory, grants_for, owner_of, classification_of, training_expiry],
    model=models.large,
    max_attempts=12,
)
def may_export(user: str, dataset: str, today: int) -> bool:
    """Decide whether {user} may export dataset {dataset} on day {today}.

    Consult the fact sources, apply the export policy, and return the decision
    as a bool, with a proof that it is correct: an allowed export must exhibit
    why the policy grants it, a denial must show the policy cannot grant it.
    """


async def main() -> None:
    print("Preparing Lean and certifying export decisions...", flush=True)
    today = 730
    for user, expected in (("riley", True), ("casey", False)):
        async with scope(on_event=print_event):
            decision, certificate = await may_export.with_certificate(user, DATASET, today)
        print(
            f"may_export({user!r}, {DATASET!r}, day {today}) = {'ALLOW' if decision else 'DENY'} (expected {expected})"
        )
        certificate.summary()
        path = certificate.write(Path(__file__).parent / "data" / f"access_{user}.lean")
        print(f"Certificate: {path}")


if __name__ == "__main__":
    asyncio.run(main())
