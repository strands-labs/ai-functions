"""What a round asks of a miner: candidate skills from recorded calls.

A :class:`Miner` receives the window's scored calls -- each with its event log
and whether it passed -- and a directory to write into, and returns the skills it
wrote there. Pass your own to ``@skillify(miner=...)`` to change where candidate
skills come from; every candidate it returns is still gated before it is published.

:class:`PromptMiner` is the library's small default: one model call per round.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from ai_functions.ai_thread import ai_function
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.ai_thread.reconstruction import reconstruct_messages

from .observation import skill_event
from .skill_library import INDEX, Skill, _param_name, load_skill, save_skill

if TYPE_CHECKING:
    from strands.models import Model

    from .learner import RecordedCall

__all__ = ["Draft", "MineFn", "Miner", "PromptMiner"]

_logger = logging.getLogger(__name__)


class Miner(Protocol):
    """Proposes candidate skills from recorded calls."""

    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]:
        """Candidate skills, each saved under ``out_dir``."""
        ...


class Draft(BaseModel):
    """One skill the model proposes."""

    name: str = Field(description="kebab-case skill name")
    description: str = Field(description="one line: when this skill applies")
    body: str = Field(description="the procedure, as Markdown instructions")


class _Drafts(BaseModel):
    skills: list[Draft]


MineFn = Callable[[str, int], Awaitable[list[Draft]]]
"""``(transcripts, max_drafts) -> drafts``, async. See :class:`PromptMiner`."""

_REFUSAL = re.compile(r"^\s*(i'?m sorry|i can'?t|i cannot|i am unable|as an ai)\b", re.IGNORECASE)
_SLUG = re.compile(r"[^a-z0-9-]+")
_MAX_CALL_CHARS = 4_000


class PromptMiner:
    """The core's default miner: one model call per round.

    Reads the round's calls as compact transcripts (failed ones marked; a long call
    keeps its head and its tail) and asks for at most ``max_drafts`` reusable skills.
    A draft with no name, no description or no body, a refusal-shaped body, a name
    already accepted or the name reserved for the skill index is dropped and logged;
    only the drafts that remain are saved. When the model's structured output fails
    altogether (``AIFunctionError``), the failure is logged and nothing is mined.

    ``model=None`` uses the library's default model; ``@skillify`` builds its default
    ``PromptMiner`` on the decorated function's own model.
    """

    def __init__(self, model: str | Model | None = None, max_drafts: int = 5, mine_fn: MineFn | None = None) -> None:
        self.model, self.max_drafts = model, max_drafts
        self._mine_fn = mine_fn or self._default_mine_fn

    @staticmethod
    def render(calls: Sequence[RecordedCall]) -> str:
        """Compact transcripts of ``calls``, failed ones marked, for the mining prompt."""
        blocks = []
        for call in calls:
            lines = [f"### call {call.call_id} -- {'PASSED' if call.passed else 'FAILED'}"]
            for message in reconstruct_messages(list(call.events)):
                for block in message["content"]:
                    if "text" in block:
                        lines.append(f"{message['role']}: {block['text']}")
                    elif "toolUse" in block:
                        lines.append(
                            f"tool {block['toolUse']['name']}({json.dumps(block['toolUse'].get('input', {}))})"
                        )
                    elif "toolResult" in block:
                        result = " ".join(c.get("text", "") for c in block["toolResult"].get("content", []))
                        lines.append(f"  -> {result}")
            text = "\n".join(lines)
            if len(text) > _MAX_CALL_CHARS:  # the end of a failed run is where it failed
                half = _MAX_CALL_CHARS // 2
                text = f"{text[:half]}\n…\n{text[-half:]}"
            blocks.append(text)
        return "\n\n".join(blocks)

    async def _default_mine_fn(self, transcripts: str, max_drafts: int) -> list[Draft]:
        @ai_function[_Drafts](model=self.model, structured_output=True)
        def mine(transcripts: str, max_drafts: int) -> str:
            """You are reading transcripts of an AI agent doing similar tasks. Some calls FAILED.

            Write at most {max_drafts} reusable skills: written procedures that would have
            turned the failures into passes, or made the passes cheaper. Each skill must
            generalise across calls rather than restate one transcript, name the situation
            it applies to, and give concrete steps. Write nothing if no lesson generalises.

            {transcripts}
            """

        return (await mine(transcripts=transcripts, max_drafts=max_drafts)).skills

    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]:
        """Mine ``calls``; each valid draft is saved under ``out_dir`` as a skill."""
        try:
            drafts = await self._mine_fn(self.render(calls), self.max_drafts)
        except AIFunctionError as exc:
            _logger.warning("PromptMiner: no drafts, the model's output did not validate: %s", exc)
            return []
        saved: list[Skill] = []
        seen: set[str] = set()
        for draft in drafts:
            slug = _SLUG.sub("-", draft.name.strip().lower()).strip("-")
            body = draft.body.strip()
            description = draft.description.strip()
            reason = (
                "no name"
                if not slug
                else "no description"
                if not description
                else "no body"
                if not body
                else "refusal-shaped body"
                if _REFUSAL.match(body)
                else "duplicate name"
                if slug in seen
                else "reserved name"
                if _param_name(slug) == INDEX
                else "over max_drafts"
                if len(saved) >= self.max_drafts
                else None
            )
            if reason:
                skill_event("draft.rejected", draft=draft, reason=reason, stage="draft")
                _logger.info("PromptMiner: dropped draft %r (%s)", draft.name, reason)
                continue
            skill_event("draft.created", draft=draft)
            seen.add(slug)
            skill = Skill(
                path=out_dir / slug,
                frontmatter={"name": slug, "description": description},
                body=body + "\n",
                raw="",
            )
            saved.append(load_skill(save_skill(skill, out_dir, dirname=slug)))
        return saved
