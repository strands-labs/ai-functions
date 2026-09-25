"""Audit to flag every wrong fee in a database, and prove none was missed.

What a correct audit means is written in Lean (`lean/Audit.lean`): every flagged
transaction has a wrong fee, every transaction with a wrong fee is flagged, and
the balance never goes negative. The rules say what a correct fee is, not how to
check one, so the agent writes the checks itself and proves they agree with the
rules.

Demonstrates:
- A contract that specifies the result without saying how to compute it
- A result that is complete as well as correct: no wrong fee escapes
- Paginated tools feeding one proof
"""

import asyncio
from pathlib import Path

from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.lean import LeanProject

project = LeanProject(Path(__file__).parent / "lean", imports=["Audit"])
Audit = project.symbols.Audit


# The processor's fee schedule in basis points per transaction kind. Kind 0 is a
# payout (debits amount + fee); kinds 1 (card) and 2 (transfer) are sales
# credited net of fee. A recorded fee is correct when it is exactly
# floor(amount * bps / 10000). Two rows violate that: 1007 was overcharged (1303
# instead of 1253) and 1019, a payout, had its fee omitted (0 instead of 175).
# The running balance stays nonnegative throughout, dipping to $145.91 right
# after payout 1009, so solvency is genuinely tight.
FEE_BPS = {0: 25, 1: 290, 2: 80}

# Rows are (id, kind, amount_cents, fee_cents).
PAGES: list[list[tuple[int, int, int, int]]] = [
    [
        (1001, 1, 125000, 3625),
        (1002, 1, 8450, 245),
        (1003, 2, 60000, 480),
        (1004, 1, 15275, 442),
        (1005, 0, 185000, 462),
        (1006, 1, 9900, 287),
        (1007, 1, 43210, 1303),  # wrong: the correct fee is 1253
        (1008, 2, 25000, 200),
    ],
    [
        (1009, 0, 80000, 200),
        (1010, 1, 5150, 149),
        (1011, 1, 30000, 870),
        (1012, 2, 12000, 96),
        (1013, 0, 26000, 65),
        (1014, 1, 7777, 225),
        (1015, 1, 64000, 1856),
        (1016, 0, 60000, 150),
    ],
    [
        (1017, 2, 40000, 320),
        (1018, 1, 22000, 638),
        (1019, 0, 70000, 0),  # wrong: the correct fee is 175
        (1020, 1, 4990, 144),
        (1021, 2, 18000, 144),
        (1022, 0, 26000, 65),
        (1023, 1, 12345, 358),
        (1024, 0, 15000, 37),
    ],
]
EXPECTED_FLAGS = {1007, 1019}


# Each tool records only its defining equation. The pages are complete (a row on
# no page does not exist), which is what makes the completeness half of the
# contract provable at all.
@verified.tool(Audit.numPages)
def page_count() -> int:
    """Return the number of pages in the settlement ledger."""
    return len(PAGES)


@verified.tool(Audit.page)
def get_page(index: int) -> list[tuple[int, int, int, int]]:
    """Fetch one page of the settlement ledger, in ledger order.

    Each row is `(id, kind, amount_cents, fee_cents)`. Kind 0 is a payout; other
    kinds are sales channels. The page holds every row of the ledger in its range.
    Pages run from 0 to page_count - 1.
    """
    if not 0 <= index < len(PAGES):
        raise ValueError(f"page {index} does not exist; pages run 0..{len(PAGES) - 1}")
    return PAGES[index]


@verified.tool(Audit.feeBps)
def fee_bps(kind: int) -> int:
    """Return the fee schedule entry for a transaction kind (0 payout, 1 card, 2 transfer), in basis points."""
    if kind not in FEE_BPS:
        raise ValueError(f"kind {kind} is not in the fee schedule")
    return FEE_BPS[kind]


# The contract is `Audit.AuditCorrect flags`: flags sound and complete against
# `FeeOk`, plus `Solvent`. There is no strategy guidance; inventing the checking
# programs, and the loop invariant the scan needs, is the task.
@verified.ai_function(
    contract=Audit.AuditCorrect,
    tools=[page_count, get_page, fee_bps],
    model=models.large,
    max_attempts=12,
)
def audit_ledger() -> list[int]:
    """Audit the merchant settlement ledger.

    Fetch the ledger pages and the fee schedule, then return the list of
    transaction ids whose recorded fee violates the schedule, flagging exactly
    the violating rows, and prove the audit correct, including that the running
    account balance never goes negative anywhere in the ledger.
    """


async def main() -> None:
    print("Preparing Lean and certifying the audit...", flush=True)
    async with scope(on_event=print_event):
        flags, certificate = await audit_ledger.with_certificate()
    print(f"audit_ledger() flagged {sorted(flags)} (expected {sorted(EXPECTED_FLAGS)})")
    certificate.summary()
    path = certificate.write(Path(__file__).parent / "data" / "audit.lean")
    print(f"Certificate: {path}")


if __name__ == "__main__":
    asyncio.run(main())
