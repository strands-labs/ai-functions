"""Skill model, library backend, and evaluation for SKILL.md-based skills.

A *skill* is an ability: an inference function that improves with experience
because it depends on memory experience can change. An AI Function is one;
this package improves it by growing its *procedural memory*, a library of
reusable procedures. Each procedure is stored as a directory containing a
``SKILL.md`` file -- YAML frontmatter (name, description, and friends) above a
Markdown body of instructions -- and loaded on demand through a ``Skill``
tool, which is why the API calls a stored procedure a skill and the library a
*skill library*. This package makes that library an optimizable object rather
than a static folder.

Covers ``Skill`` parsing and loading of SKILL.md files, ``SkillLibraryBackend``
— the ``MemoryBackend`` that exposes a skill library for recall, search,
save, delete, and consolidation — and ``skillify``, which wraps an
``@ai_function`` so its skill library grows from its own calls: each call is
recorded, and a background round mines candidates, keeps the ones that
improve on the installed library when recorded calls are replayed, and
publishes a new version.

Mining and gating are the two plug points: pass a ``Miner`` as ``miner=``
and a ``Gate`` as ``gate=``. The library ships a small default for each --
``PromptMiner`` (one model call per round) and ``CompletionGate`` (replay
trial calls with and without the candidate, judged on completion) -- but
either can be replaced with a caller's own pipeline. See ``docs/skills.md``.
"""

from __future__ import annotations

from .discovery import Miner, PromptMiner
from .evaluation import Candidate, CompletionGate, Gate, Replay, Verdict
from .learner import Environment, Episode, RecordedCall, RoundReport
from .observation import observe_skills, skill_event
from .skill_library import Skill, SkillLibraryBackend, load_skill, save_skill
from .skillify import SkillifiedFunction, SkillifyConfig, skillify

__all__ = [
    "Candidate",
    "CompletionGate",
    "Environment",
    "Episode",
    "Gate",
    "Miner",
    "PromptMiner",
    "RecordedCall",
    "Replay",
    "RoundReport",
    "Skill",
    "SkillLibraryBackend",
    "SkillifiedFunction",
    "SkillifyConfig",
    "Verdict",
    "load_skill",
    "observe_skills",
    "save_skill",
    "skillify",
    "skill_event",
]
