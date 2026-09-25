"""Answer a question based on a policy, with proof of correctness.

A company's leave policy is written in Lean (`lean/LeavePolicy.lean`): when leave is
approved, how vacation days are earned, when leave is paid. Employees ask
questions in natural language. The agent makes a judgment on how to formalize
the question in the policy's language (and reports it as a trust boundary to review),
looks up the employee's records, and proves the answer from the rules.
"""

import asyncio
from pathlib import Path

from example_helpers import models

from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject(Path(__file__).parent / "lean", imports=["LeavePolicy"])
Policy = project.symbols.LeavePolicy

# Sam has worked here 22 months, 10 of them this year, and has taken 8 vacation days.
RECORDS = {"sam": {"tenure": 22, "months_this_year": 10, "vacation_taken": 8, "sick_taken": 1}}


# Each tool reads one record. Tool outputs become trusted facts
# that can be used in the proof.
@verified.tool(Policy.tenureMonths)
def tenure_months(employee: str) -> int:
    """How long the employee has worked here, in whole months."""
    return RECORDS[employee]["tenure"]


@verified.tool(Policy.monthsWorkedThisYear)
def months_worked_this_year(employee: str) -> int:
    """Whole months the employee has worked so far this year."""
    return RECORDS[employee]["months_this_year"]


@verified.tool(Policy.vacationDaysTaken)
def vacation_days_taken(employee: str) -> int:
    """Vacation days the employee has already taken this year."""
    return RECORDS[employee]["vacation_taken"]


@verified.tool(Policy.sickDaysTaken)
def sick_days_taken(employee: str) -> int:
    """Sick days the employee has already taken this year."""
    return RECORDS[employee]["sick_taken"]


@verified.ai_function(
    contract=Policy.Answered,
    tools=[tenure_months, months_worked_this_year, vacation_days_taken, sick_days_taken],
    judgments=[Policy.asks],
    model=models.large,
)
def answer(employee: str, message: str) -> bool:
    """Employee {employee} asks HR: "{message}"

    First judge what the question asks as a proposition stated with the HR policy.
    In the justification, explain how to read the proposition and explain the judgments
    made, for example:

    Proposition:
        areApproved "sam" m [.vacation 10] ∧ ∀ k < m, ¬ areApproved "sam" k [.vacation 10]
    Justification:
        Formalized as: "Sam is approved to take 10 vacation days <m> months from now, and at no time
        less than <m> Sam will be allowed to take 10 vacation days"

        It is classified as "vacation" because the employee said the leave is: "to go skiing"
        The employee asked for two full weeks, which are 10 work days.

    Then answer the formalized question, with a proof.
    """


QUESTIONS = [
    "Can I take Thursday and Friday off, and will I be paid?",
    "Do I have enough vacation left for a full week off?",
    "Can I take both a day off and a full week off with my current vacation balance?",
    "Could I take two weeks off right now?",
]


async def main() -> None:
    for question in QUESTIONS:
        print(f"\nSam: {question}")
        result, certificate = await answer.with_certificate("sam", question)
        print(f"Answer: {'yes' if result else 'no'}")
        for fact in certificate.used.values():
            if fact.kind == "judgment":
                print("")
                print(f"  Formalized question: {fact.value}")
                print(f"  Explanation: {fact.origin}")
            else:
                print(f"  record: {fact.symbol} {' '.join(fact.arguments)} = {fact.value}")


if __name__ == "__main__":
    asyncio.run(main())
