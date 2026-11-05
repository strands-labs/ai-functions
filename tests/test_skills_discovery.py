"""``PromptMiner`` -- the core's default miner drops invalid drafts and writes only valid ones."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai_functions.experimental.skills.discovery import Draft, PromptMiner


def _miner(drafts: list[Draft]) -> PromptMiner:
    async def mine_fn(transcripts: str, max_drafts: int) -> list[Draft]:
        del transcripts, max_drafts
        return drafts

    return PromptMiner(mine_fn=mine_fn)


def test_a_draft_with_no_description_is_dropped_and_nothing_is_written(tmp_path: Path):
    out = tmp_path / "candidates"
    drafts = [Draft(name="blank-description", description="   ", body="Do the thing.")]

    saved = asyncio.run(_miner(drafts)([], out))

    assert saved == []
    assert not out.exists() or not any(out.iterdir())


def test_invalid_drafts_are_dropped_and_valid_ones_are_saved(tmp_path: Path):
    out = tmp_path / "candidates"
    drafts = [
        Draft(name="Check Units", description="when a measurement is reported", body="Convert first."),
        Draft(name="", description="no name", body="body"),
        Draft(name="no-body", description="d", body="  "),
        Draft(name="no-description", description="", body="body"),
        Draft(name="refusal", description="d", body="I'm sorry, I can't help with that."),
        Draft(name="check-units", description="duplicate", body="Again."),
    ]

    saved = asyncio.run(_miner(drafts)([], out))

    assert [s.name for s in saved] == ["check-units"]
    assert saved[0].description == "when a measurement is reported"
    assert sorted(p.name for p in out.iterdir()) == ["check-units"]


def test_a_draft_named_like_the_reserved_index_is_dropped(tmp_path: Path):
    drafts = [
        Draft(name="Skill Index", description="d", body="Read the index."),
        Draft(name="check-units", description="d", body="Convert first."),
    ]

    saved = asyncio.run(_miner(drafts)([], tmp_path / "candidates"))

    assert [s.name for s in saved] == ["check-units"]


def test_render_keeps_the_head_and_the_tail_of_a_long_call():
    from ai_functions.experimental.skills.discovery import _MAX_CALL_CHARS
    from ai_functions.experimental.skills.learner import RecordedCall
    from ai_functions.types import MessageAssistantCompleteEvent, MessageUserEvent, ThreadId

    tid = ThreadId("t")
    events = [
        MessageUserEvent(thread_id=tid, text="START " + "x" * (3 * _MAX_CALL_CHARS)),
        MessageAssistantCompleteEvent(thread_id=tid, content=[{"text": "the run failed because THE-END"}]),
    ]
    call = RecordedCall(call_id="c1", outcome=0.0, passed=False, events=events)

    text = PromptMiner.render([call])

    assert text.startswith("### call c1 -- FAILED\nuser: START x")
    assert text.endswith("assistant: the run failed because THE-END")
    assert "\n…\n" in text
    assert len(text) <= _MAX_CALL_CHARS + 3


def test_a_wholesale_structured_output_failure_yields_no_drafts(tmp_path: Path, caplog):  # noqa: ANN001
    from ai_functions.ai_thread.errors import AIFunctionError

    async def mine_fn(transcripts: str, max_drafts: int) -> list[Draft]:
        raise AIFunctionError("structured output did not validate", "mine")

    saved = asyncio.run(PromptMiner(mine_fn=mine_fn)([], tmp_path / "candidates"))

    assert saved == []
    assert "structured output did not validate" in caplog.text
