"""Leave requests — approve or deny, with proof that the policy was followed.

A company's leave policy is written in Lean (`lean/HR.lean`). An employee asks
for time off in plain words. The agent reads what kind of leave is asked for
and for how many days, looks up the employee's records, and decides the request
with a proof that the decision follows the policy. Its reading of the message is
the one thing a proof cannot check, so the certificate lists it for review.

Demonstrates:
- `verified.ai_function` with a contract from a Lean project
- Tools whose results the proof can use (`verified.tool`)
- Judgments: interpretations the agent must state explicitly
- `with_certificate`: what each decision relies on
"""

import asyncio
from pathlib import Path

from example_helpers import models

from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject(Path(__file__).parent / "lean", imports=["HR"])
HR = project.symbols.HR

RECORDS = {
    "sam": {"tenure": 14, "vacation_left": 9, "sick_taken": 1},
    "jo": {"tenure": 30, "vacation_left": 6, "sick_taken": 8},
}


# Each tool reads one record. Tool outputs become trusted facts
# that can be used in the proof.
@verified.tool(HR.tenureMonths)
def tenure_months(employee: str) -> int:
    """How long the employee has worked here, in whole months."""
    return RECORDS[employee]["tenure"]


@verified.tool(HR.vacationDaysLeft)
def vacation_days_left(employee: str) -> int:
    """Vacation days the employee has left this year."""
    return RECORDS[employee]["vacation_left"]


@verified.tool(HR.sickDaysTaken)
def sick_days_taken(employee: str) -> int:
    """Sick days the employee has already taken this year."""
    return RECORDS[employee]["sick_taken"]


@verified.ai_function(
    contract=HR.DecisionCorrect,
    tools=[tenure_months, vacation_days_left, sick_days_taken],
    judgments=[HR.requestedLeave],
    model=models.large,
)
def decide(employee: str, message: str) -> bool:
    """Decide the leave request employee {employee} sent: "{message}"

    First judge the requested leave (its kind and number of working days) with
    lean_judge on HR.requestedLeave at H.message. Then decide it, with a proof.
    """


REQUESTS = [
    ("sam", "Could I take next Thursday and Friday off? We're planning a small family trip."),
    ("jo", "I have the flu and must stay home from today, Tuesday, through Friday."),
]


async def main() -> None:
    for employee, message in REQUESTS:
        print(f"\n{employee}: {message}")
        approved, certificate = await decide.with_certificate(employee, message)
        print(f"Decision: {'approved' if approved else 'denied'}")
        for fact in certificate.used.values():
            if fact.kind == "judgment":
                print(f"  request read as: {fact.value}  ({fact.origin})")
            else:
                print(f"  record: {fact.symbol} {' '.join(fact.arguments)} = {fact.value}")


if __name__ == "__main__":
    asyncio.run(main())
