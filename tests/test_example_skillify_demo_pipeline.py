"""Tests for the demo's pipeline: formatting, ``discover_skills``, and the two plug-ins.

``FullDiscoveryMiner`` (a ``Miner``) and ``PrefilteredGate`` (a ``Gate``) are
what ``main.py`` hands ``@skillify``; ``discovery.py``/``discovery.py`` moved with
them from the library's own discovery tests. Every LLM-backed phase gets a fake
matching its real call shape (``segment_fn``, ``extract_fn``, ``group_fn``,
``merge_fn``, ``format_fn``) and the Tier-1 prefilter a stub judge, so no
LLM/network is ever constructed or reached. All disk I/O uses ``tmp_path``.

Async functions are driven via ``asyncio.run()`` inside plain ``def
test_...`` functions. No LLM/network.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from _eval_stubs import StubJudge

from ai_functions.experimental.skills.evaluation import Candidate, Verdict
from ai_functions.experimental.skills.learner import RecordedCall
from ai_functions.experimental.skills.skill_library import load_skill, load_skill_library
from ai_functions.types.events import MessageAssistantCompleteEvent, MessageUserEvent, ThreadId

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"))

from full_pipeline import FullDiscoveryMiner, PrefilteredGate  # noqa: E402
from full_pipeline.discovery import (  # noqa: E402
    DEFAULT_PRESET,
    DiscoveryPreset,
    FormatAlignmentError,
    FormattedSkillCandidate,
    _check_formatted_candidate,
    _render_draft,
    _slugify_name,
    discover_skills,
    format_skill,
)
from full_pipeline.mining import (  # noqa: E402
    CutPoints,
    ExtractedSkill,
    GateResult,
    LLMThematicGrouper,
    MergedSkillCandidate,
    Message,
    MetricGate,
    Provenance,
    SkillDraft,
    Subtask,
    ThemeGroups,
    ToolCall,
    Trajectory,
)
from full_pipeline.quality import (  # noqa: E402
    FormatAlignmentMetric,
    Level1Input,
    _CompletenessVerdict,
    _DistinctVerdict,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _traj(traj_id: str, n: int) -> Trajectory:
    # Content embeds traj_id so two trajectories never render identical
    # episode text (which would otherwise collide on the same slug when
    # _fake_extract_fn derives a draft name from the first rendered line).
    # Every assistant turn carries a tool call so Subtask's compact summary
    # is non-empty -- otherwise _episodes_from_cuts short-circuits to a
    # single whole-trajectory episode without ever calling segment_fn (see
    # mining.py), which would make every segment_fn fake below a no-op.
    messages: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        tool_calls = [ToolCall(id=f"{traj_id}-c{i}", name="do_thing", arguments={"path": f"{traj_id}/{i}"})]
        messages.append(
            Message(role=role, content=f"{traj_id} msg {i}", tool_calls=tool_calls if role == "assistant" else [])
        )
    return Trajectory(id=traj_id, messages=messages)


def _draft(name: str = "d", strategy: str = "preserve", body: str = "do the thing carefully") -> SkillDraft:
    return SkillDraft(
        name=name,
        description=f"{name} description",
        body=body,
        strategy=strategy,
        provenance=Provenance(trajectory_ids=[f"t-{name}"], constituent_ids=[f"t-{name}:0-1"], strategy=strategy),
    )


_GOOD_INSTRUCTIONS = " ".join(f"step{i}" for i in range(45))


def _good_candidate(name: str = "my-skill") -> FormattedSkillCandidate:
    return FormattedSkillCandidate(
        name=name,
        description="A well formed description with enough words.",
        overview="A short overview of the skill.",
        when_to_use="Use this when the described situation arises.",
        instructions=_GOOD_INSTRUCTIONS,
    )


def _thin_candidate(name: str = "thin-skill") -> FormattedSkillCandidate:
    # Passes _check_formatted_candidate's blank-field check but fails
    # FormatAlignmentMetric's body_min_words floor -- same shape as
    # test_format_skill_raises_format_alignment_error_for_violating_candidate.
    return FormattedSkillCandidate(
        name=name,
        description="short",
        overview="x",
        when_to_use="y",
        instructions="z",
    )


class _FakeFormatter:
    """Records calls; returns a canned/scripted sequence of candidates."""

    def __init__(self, candidates: list[FormattedSkillCandidate]) -> None:
        self._candidates = candidates
        self.calls: list[str] = []

    async def __call__(self, draft_text: str) -> FormattedSkillCandidate:
        self.calls.append(draft_text)
        idx = min(len(self.calls) - 1, len(self._candidates) - 1)
        return self._candidates[idx]


# ---------------------------------------------------------------------------
# discovery.py -- _check_formatted_candidate
# ---------------------------------------------------------------------------


def test_check_formatted_candidate_passes_for_full_fields():
    result = _check_formatted_candidate(_good_candidate())
    assert result is None or result.passed is True


def test_check_formatted_candidate_rejects_blank_instructions():
    bad = FormattedSkillCandidate(name="n", description="d", overview="o", when_to_use="w", instructions="")
    result = _check_formatted_candidate(bad)
    assert result is not None
    assert result.passed is False
    assert "instructions" in result.message


def test_check_formatted_candidate_rejects_all_blank_required_fields():
    bad = FormattedSkillCandidate(name="", description="", overview="", when_to_use="", instructions="")
    result = _check_formatted_candidate(bad)
    assert result is not None
    assert result.passed is False
    for field_name in ("name", "description", "overview", "when_to_use", "instructions"):
        assert field_name in result.message


# ---------------------------------------------------------------------------
# discovery.py -- _slugify_name / _render_draft
# ---------------------------------------------------------------------------


def test_slugify_name_lowercases_and_replaces_invalid_chars():
    assert _slugify_name("My Cool Skill!!") == "my-cool-skill"


def test_slugify_name_falls_back_to_skill_for_empty_result():
    assert _slugify_name("###") == "skill"


def test_slugify_name_caps_at_the_frontmatter_name_pattern_length():
    slug = _slugify_name("Recover A Failed Deployment " * 5)
    assert len(slug) <= 64
    assert not slug.endswith("-")


def test_format_skill_long_name_yields_a_slug_passing_the_real_format_alignment_metric():
    # Review fix: the frontmatter name_pattern caps a slug at 64 chars, so a
    # verbose model-chosen name must be truncated (and any boundary hyphen
    # trimmed) -- otherwise FormatAlignmentMetric's frontmatter check fails
    # and format_skill raises. Scored with the real Tier-1 metric, not a stub.
    verbose = "Recover A Failed Deployment " * 5
    fake = _FakeFormatter([_good_candidate(verbose)])

    skill = asyncio.run(format_skill(_draft(name="verbose-draft"), format_fn=fake))

    assert len(skill.name) <= 64
    assert not skill.name.endswith("-")
    result = FormatAlignmentMetric().compute(Level1Input(skill_spec=skill))
    assert result.passed is True
    assert not any("name" in violation for violation in result.evidence)


# ---------------------------------------------------------------------------
# discovery.py -- the optional formatter guidance string (README divergence #2)
# ---------------------------------------------------------------------------


def test_format_skill_appends_guidance_to_the_formatter_input():
    draft = _draft(name="guided")
    fake = _FakeFormatter([_good_candidate("guided-skill")])

    asyncio.run(format_skill(draft, format_fn=fake, guidance="House style: numbered imperative steps."))

    assert len(fake.calls) == 1
    assert "House style: numbered imperative steps." in fake.calls[0]
    assert _render_draft(draft) in fake.calls[0]


def test_format_skill_without_guidance_leaves_the_formatter_input_untouched():
    draft = _draft(name="unguided")
    fake = _FakeFormatter([_good_candidate("unguided-skill")])

    asyncio.run(format_skill(draft, format_fn=fake))

    assert fake.calls == [_render_draft(draft)]
    assert "Additional formatting guidance" not in fake.calls[0]


def test_render_draft_includes_name_description_and_body():
    draft = _draft(name="foo", body="the body text")
    rendered = _render_draft(draft)
    assert "foo" in rendered
    assert "foo description" in rendered
    assert "the body text" in rendered


# ---------------------------------------------------------------------------
# discovery.py -- format_skill: success path
# ---------------------------------------------------------------------------


def test_format_skill_produces_skill_with_provenance_metadata_and_passing_alignment():
    draft = _draft(name="my skill", strategy="preserve")
    draft.provenance.group_label = "theme-a"
    fake = _FakeFormatter([_good_candidate("My Skill")])

    skill = asyncio.run(format_skill(draft, format_fn=fake))

    assert skill.name == "my-skill"
    assert skill.frontmatter["version"] == 1
    metadata = skill.frontmatter["metadata"]
    assert metadata["trajectory_ids"] == ["t-my skill"]
    assert metadata["constituent_ids"] == ["t-my skill:0-1"]
    assert metadata["strategy"] == "preserve"
    assert metadata["group_label"] == "theme-a"
    assert "## Overview" in skill.body
    assert "## When to use this skill" in skill.body
    assert "## Instructions" in skill.body
    assert len(fake.calls) == 1


def test_format_skill_default_format_fn_is_module_level_ai_function_never_constructed_here():
    # Constructing format_skill's default binding must not require network/model
    # config; this test only exercises the injected-fake path, matching the
    # sibling modules' "default binding never called" precedent.
    draft = _draft()
    fake = _FakeFormatter([_good_candidate()])
    skill = asyncio.run(format_skill(draft, format_fn=fake))
    assert skill.name


# ---------------------------------------------------------------------------
# discovery.py -- format_skill: defensive validation of a fake's output
# ---------------------------------------------------------------------------


def test_format_skill_defensive_validation_of_fake_format_fn_output_raises():
    bad = FormattedSkillCandidate(name="n", description="d", overview="o", when_to_use="w", instructions="")
    fake = _FakeFormatter([bad])

    with pytest.raises(ValueError, match="instructions"):
        asyncio.run(format_skill(_draft(), format_fn=fake))


# ---------------------------------------------------------------------------
# discovery.py -- format_skill: outbound gate mechanism (format_alignment)
# ---------------------------------------------------------------------------


def test_format_skill_raises_format_alignment_error_for_violating_candidate():
    # Passes the blank-field post-condition (every required field non-empty)
    # but is far too short to satisfy FormatAlignmentMetric's body_min_words
    # floor (40 words) -- exercises the outbound gate, not the defensive
    # blank-field check above.
    thin = FormattedSkillCandidate(
        name="thin-skill",
        description="short",
        overview="x",
        when_to_use="y",
        instructions="z",
    )
    fake = _FakeFormatter([thin])
    draft = _draft(name="thin-draft")

    with pytest.raises(FormatAlignmentError) as excinfo:
        asyncio.run(format_skill(draft, format_fn=fake))

    assert excinfo.value.result is not None
    assert excinfo.value.result.passed is False
    # Folded minor (review fix): the error carries -- and its message names
    # -- the failing draft's provenance, matching mining.py's/mining.py's
    # own ValueError messages (episode id, group label).
    assert excinfo.value.draft is draft
    assert excinfo.value.skill.name == "thin-skill"
    assert str(draft.provenance.trajectory_ids) in str(excinfo.value)
    assert str(draft.provenance.constituent_ids) in str(excinfo.value)


# ---------------------------------------------------------------------------
# discovery.py -- DEFAULT_PRESET shape
# ---------------------------------------------------------------------------


def test_default_preset_names_the_documented_composition():
    assert isinstance(DEFAULT_PRESET.episode_builder, Subtask)
    # "repair" is included so `failed_trajectory_ids` is not a silent no-op;
    # it is gated per episode on the failure label, so it costs nothing
    # on a run that passes none (see the two repair tests below).
    assert DEFAULT_PRESET.modes == ("preserve", "prune", "repair")
    assert isinstance(DEFAULT_PRESET.grouper, LLMThematicGrouper)
    assert isinstance(DEFAULT_PRESET.gate, MetricGate)
    assert DEFAULT_PRESET.extract_fn is None
    assert DEFAULT_PRESET.merge_fn is None
    assert DEFAULT_PRESET.format_fn is None


# ---------------------------------------------------------------------------
# discovery.py -- end-to-end, all fakes, two synthetic trajectories
# ---------------------------------------------------------------------------


async def _segment_fn_whole_trajectory(summary: str, message_count: int) -> CutPoints:
    return CutPoints(cuts=[])


async def _fake_extract_fn(episode_text: str, mode: str) -> ExtractedSkill:
    # One skill per (episode, mode) call, name derived from the episode text
    # and mode so drafts are distinguishable.
    tag = episode_text.splitlines()[0][:12].strip().replace(" ", "-") or "ep"
    return ExtractedSkill(
        worth_extracting=True,
        name=f"{tag}-{mode}",
        description=f"Description for {tag} {mode}.",
        body=f"Body content for {tag} {mode} with enough words to be real.",
    )


async def _fake_group_fn(summary: str, draft_count: int) -> ThemeGroups:
    # Group every draft into one big theme -- exercises the merge phase.
    return ThemeGroups(groups={"one-theme": list(range(draft_count))})


class _FakeMergerAllAccept:
    async def __call__(self, constituents_text: str) -> MergedSkillCandidate:
        return MergedSkillCandidate(
            name="merged-skill",
            description="A merged skill from multiple constituents.",
            body="Merged body content describing the combined procedure end to end.",
        )


class _AcceptAllGate:
    async def accept(self, merged, constituents) -> GateResult:  # noqa: ANN001
        return GateResult(accepted=True, rationale="stub accept", scores={"merged": 1.0})


def _end_to_end_preset(*, format_fn) -> DiscoveryPreset:
    return DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_segment_fn_whole_trajectory),
        modes=("preserve",),
        extract_fn=_fake_extract_fn,
        grouper=None,  # exercised separately below; end-to-end test uses explicit grouping
        gate=_AcceptAllGate(),
        merge_fn=_FakeMergerAllAccept(),
        format_fn=format_fn,
    )


def test_discover_skills_end_to_end_writes_skills_with_provenance_and_passing_alignment(tmp_path: Path):
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    # grouper=None (_end_to_end_preset) keeps both drafts as unmerged
    # singletons, so two distinct format candidates avoid a slug collision.
    fake_format = _FakeFormatter([_good_candidate("skill-a"), _good_candidate("skill-b")])
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert len(skills) >= 1
    for skill in skills:
        assert (tmp_path / skill.name / "SKILL.md").exists()
        assert "metadata" in skill.frontmatter
        assert skill.frontmatter["metadata"]["trajectory_ids"]
        assert "## Overview" in skill.body
        assert "## Instructions" in skill.body
        assert skill.frontmatter["version"] == 1

    library = load_skill_library(tmp_path)
    assert len(library) == len(skills)


def test_discover_skills_no_out_root_returns_in_memory_skills_without_disk_io(tmp_path: Path):
    trajectories = [_traj("t1", 4)]
    fake_format = _FakeFormatter([_good_candidate("solo-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert len(skills) >= 1
    assert not (tmp_path / skills[0].name).exists()


def test_discover_skills_with_grouping_and_merging_end_to_end(tmp_path: Path):
    # Distinct from _end_to_end_preset's tests: an active grouper clusters
    # both drafts into one theme, and an accept-all gate + fake merger
    # actually merges them into a single skill -- exercises the merge phase
    # for real, not via a singleton short-circuit.
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    preset = DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_segment_fn_whole_trajectory),
        modes=("preserve",),
        extract_fn=_fake_extract_fn,
        grouper=LLMThematicGrouper(group_fn=_fake_group_fn),
        gate=_AcceptAllGate(),
        merge_fn=_FakeMergerAllAccept(),
        format_fn=_FakeFormatter([_good_candidate("merged-final")]),
    )

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert len(skills) == 1  # both drafts folded into one merged skill
    assert sorted(skills[0].frontmatter["metadata"]["trajectory_ids"]) == ["t1", "t2"]
    assert skills[0].frontmatter["metadata"]["strategy"] == "merge"
    assert skills[0].frontmatter["metadata"]["group_label"] == "one-theme"


def test_discover_skills_forwards_preset_format_guidance_to_the_formatter():
    trajectories = [_traj("t1", 4)]
    fake_format = _FakeFormatter([_good_candidate("guided-skill")])
    preset = replace(_end_to_end_preset(format_fn=fake_format), format_guidance="House style: numbered steps.")

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert len(skills) == 1
    assert all("House style: numbered steps." in call for call in fake_format.calls)


def test_discover_skills_without_preset_format_guidance_sends_no_guidance():
    trajectories = [_traj("t1", 4)]
    fake_format = _FakeFormatter([_good_candidate("unguided-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert preset.format_guidance is None
    assert all("Additional formatting guidance" not in call for call in fake_format.calls)


def test_discover_skills_no_trajectories_returns_empty_list():
    fake_format = _FakeFormatter([_good_candidate()])
    preset = _end_to_end_preset(format_fn=fake_format)
    skills = asyncio.run(discover_skills([], preset=preset, out_root=None))
    assert skills == []


# ---------------------------------------------------------------------------
# discovery.py -- skip-phase behavior
# ---------------------------------------------------------------------------


def test_discover_skills_episode_builder_none_uses_whole_trajectory_episode_no_llm(tmp_path: Path):
    calls: list[str] = []

    async def _spy_extract_fn(episode_text: str, mode: str) -> ExtractedSkill:
        calls.append(episode_text)
        return ExtractedSkill(
            worth_extracting=True,
            name="whole-traj-skill",
            description="Covers the whole trajectory.",
            body="Body content covering the whole trajectory end to end with detail.",
        )

    preset = DiscoveryPreset(
        episode_builder=None,
        modes=("preserve",),
        extract_fn=_spy_extract_fn,
        grouper=None,
        gate=None,
        format_fn=_FakeFormatter([_good_candidate("whole-traj-skill")]),
    )
    trajectories = [_traj("t1", 6)]

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert len(skills) == 1
    # One episode (the whole trajectory) x one mode => exactly one extract_fn call.
    assert len(calls) == 1


def test_discover_skills_grouper_none_yields_unmerged_singleton_drafts(tmp_path: Path):
    # Two episodes (via a segment_fn that cuts once) each extracted with the
    # same mode => 2 drafts; grouper=None must leave both unmerged (merge_fn
    # never called), even though gate is set to accept-everything.
    async def _one_cut(summary: str, message_count: int) -> CutPoints:
        return CutPoints(cuts=[message_count // 2])

    merge_calls: list[str] = []

    class _SpyMerger:
        async def __call__(self, constituents_text: str) -> MergedSkillCandidate:
            merge_calls.append(constituents_text)
            return MergedSkillCandidate(name="should-not-happen", description="d", body="b")

    preset = DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_one_cut),
        modes=("preserve",),
        extract_fn=_fake_extract_fn,
        grouper=None,
        gate=_AcceptAllGate(),
        merge_fn=_SpyMerger(),
        format_fn=_FakeFormatter([_good_candidate("a"), _good_candidate("b")]),
    )
    trajectories = [_traj("t1", 4)]

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert merge_calls == []  # merge_group's singleton short-circuit: merge_fn never invoked
    assert len(skills) == 2


def test_discover_skills_gate_none_skips_merging_even_with_grouping(tmp_path: Path):
    # grouper clusters both drafts into one group; gate=None must still leave
    # them unmerged (merge_group is never called at all).
    async def _one_cut(summary: str, message_count: int) -> CutPoints:
        return CutPoints(cuts=[message_count // 2])

    preset = DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_one_cut),
        modes=("preserve",),
        extract_fn=_fake_extract_fn,
        grouper=LLMThematicGrouper(group_fn=_fake_group_fn),
        gate=None,
        merge_fn=None,
        format_fn=_FakeFormatter([_good_candidate("a"), _good_candidate("b")]),
    )
    trajectories = [_traj("t1", 4)]

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert len(skills) == 2  # both drafts survive unmerged despite being grouped together


def test_discover_skills_extraction_yields_no_drafts_returns_empty_list():
    async def _nothing_worth_extracting(episode_text: str, mode: str) -> ExtractedSkill:
        return ExtractedSkill(worth_extracting=False)

    preset = DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_segment_fn_whole_trajectory),
        modes=("preserve",),
        extract_fn=_nothing_worth_extracting,
        grouper=None,
        gate=None,
    )
    trajectories = [_traj("t1", 4)]

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))
    assert skills == []


# ---------------------------------------------------------------------------
# discovery.py -- re-running discover_skills against the same out_root bumps version
# ---------------------------------------------------------------------------


def test_discover_skills_re_run_against_same_out_root_bumps_version(tmp_path: Path):
    trajectories = [_traj("t1", 4)]
    preset = _end_to_end_preset(format_fn=_FakeFormatter([_good_candidate("stable-skill")]))

    first = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))
    assert len(first) == 1
    assert first[0].frontmatter["version"] == 1

    preset2 = _end_to_end_preset(format_fn=_FakeFormatter([_good_candidate("stable-skill")]))
    second = asyncio.run(discover_skills(trajectories, preset=preset2, out_root=tmp_path))
    assert len(second) == 1
    assert second[0].frontmatter["version"] == 2


# ---------------------------------------------------------------------------
# discovery.py -- Finding 1 (review): same-call slug collisions raise loudly
# ---------------------------------------------------------------------------


def test_discover_skills_strict_same_call_slug_collision_raises_and_writes_nothing(tmp_path: Path):
    # grouper=None (_end_to_end_preset) keeps both drafts as unmerged
    # singletons; both format to the same slug here on purpose.
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_good_candidate("dup-skill"), _good_candidate("dup-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with pytest.raises(ValueError, match="dup-skill") as excinfo:
        asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path, strict_format=True))

    # Both colliding drafts' provenance (trajectory ids) named in the message.
    assert "t1" in str(excinfo.value)
    assert "t2" in str(excinfo.value)
    # Detected before any of the colliding writes happens: neither directory exists.
    assert list(tmp_path.iterdir()) == []


def test_discover_skills_strict_same_call_slug_collision_raises_even_without_out_root():
    # The check is not scoped to disk writes -- an in-memory caller should
    # not receive two same-named Skills either.
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_good_candidate("dup-skill"), _good_candidate("dup-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with pytest.raises(ValueError, match="dup-skill"):
        asyncio.run(discover_skills(trajectories, preset=preset, out_root=None, strict_format=True))


# A live unattended induction run lost ~55 minutes of discovery because two
# drafts from different trajectories formatted to the same slug and the whole
# call aborted. Non-strict mode already drops any other bad draft with a
# warning and keeps going; a name collision now gets the same treatment. The
# first draft wins, so nothing is overwritten and the return value never
# describes content that is not on disk.


def test_discover_skills_non_strict_collision_keeps_the_first_and_drops_the_duplicate(tmp_path: Path, caplog):
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_good_candidate("dup-skill"), _good_candidate("dup-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with caplog.at_level("WARNING"):
        skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert [s.name for s in skills] == ["dup-skill"]
    assert skills[0].frontmatter["metadata"]["trajectory_ids"] == ["t1"]
    assert [d.name for d in tmp_path.iterdir()] == ["dup-skill"]
    assert "dup-skill" in caplog.text and "t2" in caplog.text


def test_discover_skills_non_strict_collision_keeps_distinct_skills_alongside():
    trajectories = [_traj("t1", 4), _traj("t2", 4), _traj("t3", 4)]
    fake_format = _FakeFormatter(
        [_good_candidate("dup-skill"), _good_candidate("other-skill"), _good_candidate("dup-skill")]
    )
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=None))

    assert [s.name for s in skills] == ["dup-skill", "other-skill"]


def test_discover_skills_rejects_a_slug_colliding_with_the_reserved_index_parameter(tmp_path: Path):
    # Review fix: `SkillLibraryBackend` maps "-" to "_" when turning a slug
    # into a parameter name, so slug "skill-index" would collide with its
    # reserved `skill_index` index parameter and make the library unloadable.
    trajectories = [_traj("t1", 4)]
    fake_format = _FakeFormatter([_good_candidate("Skill Index")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with pytest.raises(ValueError, match="skill_index") as excinfo:
        asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert "skill-index" in str(excinfo.value)
    assert "t1" in str(excinfo.value)  # the offending draft's provenance
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_discover_skills_distinct_slugs_within_one_call_do_not_collide(tmp_path: Path):
    # Regression guard: the collision check must not false-positive on the
    # ordinary case (this mirrors the pre-existing end-to-end test).
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_good_candidate("skill-a"), _good_candidate("skill-b")])
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert {s.name for s in skills} == {"skill-a", "skill-b"}


# ---------------------------------------------------------------------------
# discovery.py -- Finding 2 (review): partial-success formatting / strict_format
# ---------------------------------------------------------------------------


def test_discover_skills_default_mode_drops_violating_draft_saves_the_rest_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_thin_candidate("thin-skill"), _good_candidate("good-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with caplog.at_level(logging.WARNING, logger="full_pipeline.discovery"):
        skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    # strict_format defaults to False: the thin draft is dropped, the good
    # one survives and is saved -- the whole batch is not discarded.
    assert len(skills) == 1
    assert skills[0].name == "good-skill"
    assert (tmp_path / "good-skill" / "SKILL.md").exists()
    assert not (tmp_path / "thin-skill").exists()

    # The dropped draft is visible only via the log in default mode -- the
    # warning names the slug and the source draft's provenance.
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("thin-skill" in msg for msg in warnings)
    assert any("trajectory_ids" in msg for msg in warnings)


def test_discover_skills_default_mode_all_drafts_violating_returns_empty_list(tmp_path: Path):
    trajectories = [_traj("t1", 4)]
    fake_format = _FakeFormatter([_thin_candidate("thin-only")])
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path))

    assert skills == []
    assert list(tmp_path.iterdir()) == []


def test_discover_skills_strict_format_true_raises_and_writes_nothing(tmp_path: Path):
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_thin_candidate("thin-skill"), _good_candidate("good-skill")])
    preset = _end_to_end_preset(format_fn=fake_format)

    with pytest.raises(FormatAlignmentError):
        asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path, strict_format=True))

    # Abort-all: formatting happens before any write, so nothing is saved --
    # not even the good draft that would have survived in default mode.
    assert list(tmp_path.iterdir()) == []


def test_discover_skills_strict_format_true_all_good_drafts_unaffected(tmp_path: Path):
    # strict_format=True must not change behavior when nothing violates.
    trajectories = [_traj("t1", 4), _traj("t2", 4)]
    fake_format = _FakeFormatter([_good_candidate("skill-a"), _good_candidate("skill-b")])
    preset = _end_to_end_preset(format_fn=fake_format)

    skills = asyncio.run(discover_skills(trajectories, preset=preset, out_root=tmp_path, strict_format=True))

    assert {s.name for s in skills} == {"skill-a", "skill-b"}


def test_default_preset_runs_repair_extraction_for_a_failed_trajectory(tmp_path: Path):
    """``failed_trajectory_ids`` must actually enable repair under the default preset.

    ``discover_skills``' own docstring says the argument gates ``"repair"``
    mode, and the design it implements says repair "activates when the
    trajectory carries a failure label". A caller who passes the argument
    has done the only thing that could express that intent.

    Fails if ``DEFAULT_PRESET.modes`` omits ``"repair"``: the extraction spy
    records only ``preserve``/``prune`` and the argument is a silent no-op.
    Only the LLM-backed phases are swapped for fakes -- ``modes`` comes from
    ``DEFAULT_PRESET`` itself, which is the thing under test.
    """
    modes_seen: list[str] = []

    async def spy_extract(episode_text: str, mode: str) -> ExtractedSkill:
        modes_seen.append(mode)
        return ExtractedSkill(
            worth_extracting=True,
            name=f"{mode}-skill",
            description=f"{mode} description with enough words to pass.",
            body="do the thing carefully and then verify it",
        )

    preset = replace(
        DEFAULT_PRESET,
        episode_builder=None,  # whole-trajectory episodes, no segmentation LLM
        grouper=None,  # singleton groups, no grouping LLM
        gate=None,  # no merging, no gate LLM
        extract_fn=spy_extract,
        format_fn=_FakeFormatter([_good_candidate(f"s{i}") for i in range(12)]),
    )

    asyncio.run(
        discover_skills(
            [_traj("t1", 4)],
            preset=preset,
            out_root=tmp_path / "out",
            failed_trajectory_ids=frozenset({"t1"}),
        )
    )

    assert "repair" in modes_seen


def test_default_preset_does_not_run_repair_without_a_failure_label(tmp_path: Path):
    """Repair stays inert when no trajectory is labelled failed.

    Adding ``"repair"`` to the default modes must not make it fire on every
    run: it is gated on the failure label too, so a default call with no
    ``failed_trajectory_ids`` costs exactly what it did before.

    Fails if the per-episode failure gate in ``extract_skills`` were dropped.
    """
    modes_seen: list[str] = []

    async def spy_extract(episode_text: str, mode: str) -> ExtractedSkill:
        modes_seen.append(mode)
        return ExtractedSkill(
            worth_extracting=True,
            name=f"{mode}-skill",
            description=f"{mode} description with enough words to pass.",
            body="do the thing carefully and then verify it",
        )

    preset = replace(
        DEFAULT_PRESET,
        episode_builder=None,
        grouper=None,
        gate=None,
        extract_fn=spy_extract,
        format_fn=_FakeFormatter([_good_candidate(f"s{i}") for i in range(12)]),
    )

    asyncio.run(discover_skills([_traj("t1", 4)], preset=preset, out_root=tmp_path / "out"))

    assert "repair" not in modes_seen
    assert set(modes_seen) == {"preserve", "prune"}


# ---------------------------------------------------------------------------
# discovery.py -- refusal filter
# ---------------------------------------------------------------------------
#
# Regression: a model refusal could be saved as a skill -- e.g. a draft saying combusting
# a compound was an acceptable way to complete the task, the formatter
# declined, and its refusal text filled every field of a well-formed
# candidate. It passed format_alignment (all headings present), passed the
# Tier-1 prefilter, and was the top BM25 hit for the task it refused.


def _refusal_candidate() -> FormattedSkillCandidate:
    # Verbatim shape of the live refusal.
    return FormattedSkillCandidate(
        name="handle-compounds-without-boiling-point",
        description=(
            "This skill is refused because it describes deliberately igniting substances "
            "on fire, which is dangerous and harmful."
        ),
        overview="This candidate skill cannot be formatted or added to a shared skill library.",
        when_to_use="This skill should never be used. It does not describe a safe, valid, or reusable procedure.",
        instructions=_GOOD_INSTRUCTIONS,
    )


def test_format_skill_rejects_a_refusal_rendered_as_a_skill():
    from full_pipeline.discovery import SkillRefusalError

    fake = _FakeFormatter([_refusal_candidate()])

    with pytest.raises(SkillRefusalError) as excinfo:
        asyncio.run(format_skill(_draft(name="combust"), format_fn=fake))

    assert excinfo.value.skill.name == "handle-compounds-without-boiling-point"
    assert "refus" in excinfo.value.result.rationale.lower()


def test_a_refusal_is_a_format_alignment_error_so_non_strict_discovery_drops_it():
    # Subclassing is what makes discover_skills(strict_format=False) drop a
    # refusal with a warning, with no change to the pipeline itself.
    from full_pipeline.discovery import SkillRefusalError

    assert issubclass(SkillRefusalError, FormatAlignmentError)


def test_format_skill_does_not_flag_a_legitimate_skill_that_mentions_danger_and_never():
    # Safety language inside a real procedure is not a refusal. The filter
    # has to key on the candidate talking about *itself*, not on words like
    # "dangerous" or "never" that ordinary instructions use.
    legit = FormattedSkillCandidate(
        name="safe-stove-heating",
        description="Heat a substance on the stove without damaging the container or the room.",
        overview="Heating on a stove is dangerous if the container is not heat-safe.",
        when_to_use="Use when a task requires raising a substance's temperature.",
        instructions="Never place a plastic container on an active stove. " + _GOOD_INSTRUCTIONS,
    )

    skill = asyncio.run(format_skill(_draft(name="stove"), format_fn=_FakeFormatter([legit])))

    assert skill.name == "safe-stove-heating"


# ---------------------------------------------------------------------------
# FullDiscoveryMiner -- the round's scored calls through discover_skills
# ---------------------------------------------------------------------------

_TID = ThreadId("t-1")


def _call(call_id: str, answer: str, *, passed: bool) -> RecordedCall:
    events = [
        MessageUserEvent(thread_id=_TID, text=f"task for {call_id}"),
        MessageAssistantCompleteEvent(thread_id=_TID, content=[{"text": answer}]),
    ]
    return RecordedCall(call_id=call_id, outcome=1.0 if passed else 0.0, passed=passed, events=events)


def test_full_discovery_miner_writes_skills_and_labels_only_failed_calls_for_repair(tmp_path: Path):
    seen_modes: list[tuple[str, str]] = []

    async def extract_fn(episode_text: str, mode: str) -> ExtractedSkill:
        tag = "ok" if "worked" in episode_text else "bad"
        seen_modes.append((tag, mode))
        return ExtractedSkill(worth_extracting=True, name=f"{tag}-{mode}", description="d.", body="b")

    names = iter(["skill-one", "skill-two", "skill-three"])

    async def format_fn(draft_text: str) -> FormattedSkillCandidate:
        return _good_candidate(next(names))

    preset = replace(_end_to_end_preset(format_fn=format_fn), modes=("preserve", "repair"), extract_fn=extract_fn)
    calls = [_call("c-ok", "it worked", passed=True), _call("c-bad", "it broke", passed=False)]

    skills = asyncio.run(FullDiscoveryMiner(preset=preset)(calls, tmp_path / "candidates"))

    assert sorted(seen_modes) == [("bad", "preserve"), ("bad", "repair"), ("ok", "preserve")]
    assert [s.name for s in skills] == ["skill-one", "skill-two", "skill-three"]
    assert all((tmp_path / "candidates" / s.name / "SKILL.md").exists() for s in skills)


def test_full_discovery_miner_with_no_calls_mines_nothing(tmp_path: Path):
    preset = _end_to_end_preset(format_fn=_FakeFormatter([_good_candidate("never")]))
    assert asyncio.run(FullDiscoveryMiner(preset=preset)([], tmp_path / "candidates")) == []


# ---------------------------------------------------------------------------
# PrefilteredGate -- Tier-1 prefilter, then the inner gate, highest Tier-1 first
# ---------------------------------------------------------------------------


class _PerSkillJudge:
    """Stub judge: the same score on every Tier-1 verdict about the skill the prompt shows."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    async def score(self, prompt, schema, *, system_prompt=None):  # noqa: ANN001, ANN201
        [name] = [n for n in self._scores if f"name: {n}\n" in prompt]
        s = self._scores[name]
        return StubJudge(
            {
                _CompletenessVerdict: _CompletenessVerdict(
                    scope_clarity=s, when_to_use=s, step_specificity=s, examples=s, aggregate=s, rationale="stub"
                ),
                _DistinctVerdict: _DistinctVerdict(score=s, most_similar_skill="", reasoning="stub"),
            }
        )._verdicts[schema]


class _RecordingGate:
    """Inner gate: admits everything, records the order it was handed candidates in."""

    def __init__(self, scores: dict[str, object] | None = None) -> None:
        self.seen: list[str] = []
        self._scores = scores or {}

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        self.seen = [c.skill.name for c in candidates]
        return [Verdict(c, True, 0.5, "ok", dict(self._scores)) for c in candidates]


def _skills(tmp_path: Path, *names: str) -> list[Candidate]:
    out = []
    for name in names:
        d = tmp_path / "candidates" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: A well formed description with enough words.\n---\n"
            f"## Overview\nx\n\n## When to use this skill\ny\n\n## Instructions\n{_GOOD_INSTRUCTIONS}\n"
        )
        out.append(Candidate(load_skill(d)))
    return out


async def _never_replayed(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
    raise AssertionError("PrefilteredGate itself never replays")


def test_prefiltered_gate_rejects_below_min_tier1_and_never_hands_it_to_the_inner_gate(tmp_path: Path):
    cands = _skills(tmp_path, "weak", "strong")
    inner = _RecordingGate()
    # Tier-1 is the mean of completeness, distinctiveness and format alignment; with no
    # installed library distinctiveness is 1.0, so a 0.0 judge score lands at 2/3.
    gate = PrefilteredGate(inner, min_tier1=0.7, judge=_PerSkillJudge({"weak": 0.0, "strong": 0.9}))

    weak, strong = asyncio.run(gate(cands, [], [], _never_replayed))

    assert inner.seen == ["strong"]
    assert not weak.admitted and weak.scores["stage"] == "prefilter" and weak.scores["tier1"] < 0.7
    assert strong.admitted and strong.scores["tier1"] >= 0.7


def test_prefiltered_gate_hands_survivors_highest_tier1_first_and_answers_in_input_order(tmp_path: Path):
    cands = _skills(tmp_path, "low", "high", "mid")
    inner = _RecordingGate()
    gate = PrefilteredGate(inner, min_tier1=0.0, judge=_PerSkillJudge({"low": 0.2, "high": 0.9, "mid": 0.5}))

    verdicts = asyncio.run(gate(cands, [], [], _never_replayed))

    assert inner.seen == ["high", "mid", "low"]
    assert [v.candidate for v in verdicts] == cands
    assert [v.scores["order"] for v in verdicts] == [2.0, 0.0, 1.0]  # the inner gate's order, offset by none


def test_prefiltered_gate_decides_prefilter_rejections_first_and_names_tier1_in_budget_cuts(tmp_path: Path):
    cands = _skills(tmp_path, "keep", "drop", "cut")
    inner = _RecordingGate({"budget_cuts": ["candidate cut", "trial input c-9"]})
    gate = PrefilteredGate(inner, min_tier1=0.7, judge=_PerSkillJudge({"keep": 0.9, "drop": 0.0, "cut": 0.6}))

    keep, drop, cut = asyncio.run(gate(cands, [], [], _never_replayed))

    assert drop.scores["order"] == 0.0 and keep.scores["order"] == 1.0 and cut.scores["order"] == 2.0
    assert keep.scores["budget_cuts"][0] == f"candidate cut (tier1 {cut.scores['tier1']:.2f})"
    assert keep.scores["budget_cuts"][1] == "trial input c-9"


def test_prefiltered_gate_measures_distinctiveness_against_the_installed_library(tmp_path: Path):
    [cand] = _skills(tmp_path, "new")
    installed = tmp_path / "lib" / "old"
    installed.mkdir(parents=True)
    (installed / "SKILL.md").write_text("---\nname: old\ndescription: already here\n---\nStep 1.\n")
    judge = StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=1, when_to_use=1, step_specificity=1, examples=1, aggregate=1.0, rationale="s"
            ),
            _DistinctVerdict: _DistinctVerdict(score=1.0, most_similar_skill="", reasoning="s"),
        }
    )

    asyncio.run(PrefilteredGate(_RecordingGate(), min_tier1=0.0, judge=judge)([cand], [installed], [], _never_replayed))

    distinct_prompts = [p for p, schema, _ in judge.calls if schema is _DistinctVerdict]
    assert len(distinct_prompts) == 1 and "- old: already here" in distinct_prompts[0]


# ---------------------------------------------------------------------------
# PrefilteredGate -- review fixes (T3-c): inner-gate verdict validation, and a
# revision's distinctiveness measured with the skill it replaces removed.
# ---------------------------------------------------------------------------


class _WrongCountInnerGate:
    """An inner ``Gate`` that returns no verdicts at all."""

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        return []


def test_prefiltered_gate_raises_if_inner_gate_returns_the_wrong_verdict_count(tmp_path: Path):
    cands = _skills(tmp_path, "a", "b")
    gate = PrefilteredGate(_WrongCountInnerGate(), min_tier1=0.0, judge=_PerSkillJudge({"a": 0.9, "b": 0.9}))

    with pytest.raises(RuntimeError, match="verdicts"):
        asyncio.run(gate(cands, [], [], _never_replayed))


class _ForeignVerdictInnerGate:
    """An inner ``Gate`` that returns the right count, but about candidates it was not given."""

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        return [Verdict(Candidate(c.skill), True, 1.0, "copied", {}) for c in candidates]


def test_prefiltered_gate_raises_if_inner_gate_returns_verdicts_about_other_candidates(tmp_path: Path):
    cands = _skills(tmp_path, "a", "b")
    gate = PrefilteredGate(_ForeignVerdictInnerGate(), min_tier1=0.0, judge=_PerSkillJudge({"a": 0.9, "b": 0.9}))

    with pytest.raises(RuntimeError, match="one per candidate"):
        asyncio.run(gate(cands, [], [], _never_replayed))


def test_prefiltered_gate_measures_a_revisions_distinctiveness_with_the_replaced_skill_removed(tmp_path: Path):
    # "new" revises the installed "old" skill: distinctiveness must compare it against
    # the rest of the library only, or a revision always reads as a near-duplicate of
    # the very skill it is meant to replace.
    cand_dir = tmp_path / "candidates" / "old"
    cand_dir.mkdir(parents=True)
    (cand_dir / "SKILL.md").write_text(
        f"---\nname: old\ndescription: A well formed description with enough words.\n---\n"
        f"## Overview\nx\n\n## When to use this skill\ny\n\n## Instructions\n{_GOOD_INSTRUCTIONS}\n"
    )
    candidate = Candidate(load_skill(cand_dir), replaces="old")
    installed_old = tmp_path / "lib" / "old"
    installed_old.mkdir(parents=True)
    (installed_old / "SKILL.md").write_text("---\nname: old\ndescription: already here\n---\nStep 1.\n")
    installed_other = tmp_path / "lib" / "other"
    installed_other.mkdir(parents=True)
    (installed_other / "SKILL.md").write_text("---\nname: other\ndescription: something else\n---\nStep 1.\n")
    judge = StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=1, when_to_use=1, step_specificity=1, examples=1, aggregate=1.0, rationale="s"
            ),
            _DistinctVerdict: _DistinctVerdict(score=1.0, most_similar_skill="", reasoning="s"),
        }
    )

    asyncio.run(
        PrefilteredGate(_RecordingGate(), min_tier1=0.0, judge=judge)(
            [candidate], [installed_old, installed_other], [], _never_replayed
        )
    )

    distinct_prompts = [p for p, schema, _ in judge.calls if schema is _DistinctVerdict]
    assert len(distinct_prompts) == 1
    assert "- old: already here" not in distinct_prompts[0]
    assert "- other: something else" in distinct_prompts[0]
