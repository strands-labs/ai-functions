"""Teams: a lead over live members, with a hook that reviews the answer.

An incident desk writes the customer status update for a payment outage. The
lead is an ordinary ``@ai_function``. It holds two members as typed tools: an
analyst's ``timeline`` method on a live thread (``MethodMember``) and a
skeptic function that checks one claim against the notes (``FunctionMember``).
Each member keeps its own conversation across the lead's questions.

``HouseStyle`` is a hook. Its ``on_answer`` reads the lead's draft and returns
``Accept()`` or ``Revise(feedback=...)``, which sends the lead back to work with
the feedback. The team runs under a budget in dollars, and the result carries
the answer, a transcript of every member call and revision, and what the run
cost.
"""

import asyncio
import re

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.experimental.teams import (
    Accept,
    FunctionMember,
    MemberCall,
    MethodMember,
    Revise,
    Revision,
    RevisionCap,
    Team,
    Workspace,
)
from ai_functions.runtime import Budget, Cost

NOTES = """\
09:02 UTC deploy 4172 rolls out to checkout: new payment client, retry limit changed from 5 to 0
09:05 UTC p99 latency on /pay climbs from 300 ms to 2.1 s
09:07 UTC error rate on /pay reaches 4%; the card processor reports no incidents on its side
09:12 UTC on-call rolls back deploy 4172
09:15 UTC latency and error rate on /pay are back to baseline
"""


class Finding(BaseModel):
    answer: str = Field(description="The answer, in one or two sentences.")
    evidence: list[str] = Field(description="The note lines the answer rests on, quoted exactly.")


class Analyst(MethodAgent):
    """The analyst for one service. Its notes live on ``self`` and reach every prompt."""

    model = models.small

    def __init__(self, service: str, notes: str) -> None:
        self.service = service
        self.notes = notes

    @ai_method(Finding, description="Answer a question about when things happened")
    def timeline(self, question: str) -> Finding:
        """You're the analyst for the {self.service} service. Answer from these notes only:

        {self.notes}
        Question: {question}
        """


# The body returns the prompt, so it can put the module's notes in front of the claim.
@ai_function[str](model=models.small)
def check_claim(claim: str) -> str:
    return (
        'Say whether these incident notes support this claim. Reply "supported" or "unsupported", '
        f"then quote the note line that decides it.\n\nNotes:\n{NOTES}\nClaim: {claim}"
    )


DESK = """\
You write customer status updates for incidents, in at most three sentences. Ask the analyst for
the facts before you write, and have the skeptic check your root-cause claim. Customers read the
update, so leave out deploy numbers and internal names."""


# The team's request lands in the lead's first parameter. A reviewer's feedback lands there too,
# on the same live thread, so the standing instructions live in the system prompt and the
# docstring is only the request: a revision then reads as a follow-up to the lead's own draft.
@ai_function(model=models.medium, system_prompt=DESK)
def status_update(incident: str) -> str:
    """{incident}"""


class HouseStyle:
    """Send the update back until it states the restore time the way the status page prints it."""

    RESOLVED = re.compile(r"Resolved at \d{2}:\d{2} UTC")

    def on_answer(self, work: Workspace, answer: str) -> Accept | Revise:
        if self.RESOLVED.search(str(answer)):
            return Accept()
        return Revise(
            feedback='End the update with the restore time in the form "Resolved at HH:MM UTC".',
            cap=2,
        )


def _short(text: str, limit: int = 110) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


async def main() -> None:
    analyst = Analyst("checkout", NOTES)
    team = Team(
        status_update,
        members=[
            # The lead sees this member as the `analyst_timeline` tool. Under a budget, each ask
            # reserves the member's estimate until the member has measured an ask of its own.
            MethodMember(analyst, "timeline", ask_estimate=Cost(dollars=0.01)),
            FunctionMember(check_claim, name="skeptic", ask_estimate=Cost(dollars=0.01)),
        ],
        hooks=[HouseStyle()],
    )

    rule("One team run under a $0.50 budget")
    run = await team.run(
        "Payment errors on checkout this morning; the issue is fixed.",
        budget=Budget(Cost(dollars=0.50)),
    )
    display("Status update", run.answer)

    rule("The transcript: written by the team, not reported by the model")
    for entry in run.transcript:
        if isinstance(entry, MemberCall):
            print(f"member  {entry.call_id}")
            print(f"    asked:    {_short(entry.request)}")
            print(f"    answered: {_short(entry.answer)}")
        elif isinstance(entry, Revision):
            print(f"revise  {entry.hook} round {entry.round}: {entry.feedback}")
        elif isinstance(entry, RevisionCap):
            print(f"revise_cap  {entry.hook} after {entry.rounds} rounds; the last answer passed through")
        else:
            print(f"{entry.kind}  {entry.data}")

    rule("What the run cost")
    dollars = "unpriced" if run.cost.dollars is None else f"${run.cost.dollars:.4f}"
    print(f"{run.cost.tokens} tokens, {dollars}")


if __name__ == "__main__":
    asyncio.run(main())
