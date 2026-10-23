"""Local trainable parameters: a prompt fragment that learns from feedback.

``reply`` drafts a support reply. Its ``guidelines`` parameter is marked
``Trainable``, so the default on the signature is the parameter's initial value,
and ``Bind`` keeps the value in a ``Local`` host in this process. No memory
backend is involved.

One traced call and one piece of feedback are enough for ``TextGradOptimizer``
to add a guideline. The host bumps the parameter's version, and the next call
reads the new text. At the end, the host's ``snapshot()`` is written as JSON and
loaded into a fresh binder with ``restore()``, the way a program keeps what it
learned between runs.
"""

import asyncio
import json
from typing import Annotated

from example_helpers import models
from example_helpers.utils import display, rule

from ai_functions import TextGradOptimizer, ai_function
from ai_functions.experimental.params import Bind, Param, Trainable

TICKET = """\
Subject: Charged twice
I was billed twice for my March invoice and want the duplicate refunded.
Order 4417, card ending 0921.
"""

# A house rule the model can't guess, so the first reply breaks it and the second one follows it.
FEEDBACK = (
    "The reply has no case reference. Every reply must end with the case reference CASE-<order number> "
    "on its own line, and sign off as 'The Billing Desk'."
)


# `Annotated[str, Trainable(...)]` marks `guidelines` as a local trainable value; the
# default is its initial text. `Trainable[str]` is the short form when no knob is needed.
# The parameter stays in the signature, so a caller can still pass guidelines by hand.
@ai_function(model=models.medium)
def reply(
    ticket: str,
    guidelines: Annotated[str, Trainable(description="Rules every support reply follows.")] = (
        "Thank the customer and keep the reply under 120 words."
    ),
) -> str:
    """Write the reply to this support ticket.

    Follow these guidelines:
    <guidelines>
    {guidelines}
    </guidelines>

    Ticket:
    {ticket}
    """


async def main() -> None:
    bound = Bind(reply)
    optimizer = TextGradOptimizer(model=models.small)

    rule("First reply")
    display("Guidelines, version 0", bound.local.load("guidelines")[0])
    # trace() fills `guidelines` from the host and records which version this call read.
    first = await bound.trace(ticket=TICKET)
    display("Reply", str(first))

    rule("Feedback")
    display("Feedback", FEEDBACK)
    # The host is the backend: the optimizer routes the feedback to `guidelines` and the
    # host's reducer appends the new guideline to the current text, with no model call.
    await optimizer.step(first, FEEDBACK, backends=[bound.local])
    guidelines, version = bound.local.load("guidelines")
    display(f"Guidelines, version {version}", guidelines)

    rule("Second reply, with the updated guidelines")
    second = await bound.trace(ticket=TICKET)
    display("Reply", str(second))

    rule("Keeping what was learned")
    # A snapshot is one frozen `Param` record per parameter, and each record round-trips as JSON.
    saved = json.dumps({name: param.model_dump(mode="json") for name, param in bound.local.snapshot().items()})
    fresh = Bind(reply)
    fresh.local.restore({name: Param.model_validate(record) for name, record in json.loads(saved).items()})
    guidelines, version = fresh.local.load("guidelines")
    display(f"Restored guidelines, version {version}", guidelines)


if __name__ == "__main__":
    asyncio.run(main())
