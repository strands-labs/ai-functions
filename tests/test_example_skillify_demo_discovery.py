"""Tests for the demo's discovery stages: episodes, extraction, grouping, trajectory input.

Moved with ``examples/skillify_demo/full_pipeline`` from the library's
own discovery tests; only the builders, groupers and input path the demo runs
are kept. Every LLM-backed stage gets a fake (``segment_fn``, ``extract_fn``,
``group_fn``) matching the real template's call shape, so the module-level
``@ai_function`` templates are constructed but never called.

The post-conditions (``_check_cuts``, ``_check_extracted_skill``,
``_check_groups``) are only wired into a live model turn inside ``AIThread``'s
own retry loop, so they are tested the way the ``PostCondition`` contract
defines them: as plain callables, called directly with the keyword the real
``AIThread._validate_result`` would extract from the prompt function's bound
arguments.

Async code is driven via ``asyncio.run()`` inside plain ``def test_...``
functions. No LLM/network.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

from ai_functions.ai_thread import PostConditionResult
from ai_functions.types.events import (
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    ThreadId,
    ToolResultEvent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"))

from full_pipeline.mining import (  # noqa: E402
    MODES,
    UNGROUPED_LABEL,
    CutPoints,
    Episode,
    ExtractedSkill,
    Grouper,
    LLMThematicGrouper,
    Message,
    Provenance,
    SkillDraft,
    SkillGroup,
    Subtask,
    ThemeGroups,
    ToolCall,
    Trajectory,
    _check_cuts,
    _check_extracted_skill,
    _check_groups,
    _compact_summary,
    _episode_id,
    _render_episode,
    _target_for,
    extract_skills,
    trajectories_from_events,
)

# ===========================================================================
# mining.py
# ===========================================================================


def _messages(n: int, *, tool_at: set[int] = frozenset()) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        if i in tool_at:
            out.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id=f"c{i}", name=f"tool_{i}", arguments={"path": f"/a/b/file_{i}.py"})],
                )
            )
        else:
            out.append(Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"))
    return out


def _tool_traj(n: int, *, tool_at: set[int] = frozenset(), traj_id: str = "t1") -> Trajectory:
    return Trajectory(id=traj_id, messages=_messages(n, tool_at=tool_at))


# ---------------------------------------------------------------------------
# _target_for / _compact_summary -- the Subtask prompt input
# ---------------------------------------------------------------------------


def test_target_for_prefers_path_key_and_takes_basename():
    call = ToolCall(id="c1", name="read_file", arguments={"path": "/a/b/c/notes.txt", "other": "ignored"})
    assert _target_for(call) == "notes.txt"


def test_target_for_falls_back_to_first_argument_when_no_preferred_key():
    call = ToolCall(id="c1", name="custom_tool", arguments={"weird_key": "value123"})
    assert _target_for(call) == "value123"


def test_target_for_empty_when_no_arguments():
    call = ToolCall(id="c1", name="noop", arguments={})
    assert _target_for(call) == ""


def test_target_for_truncates_to_40_chars():
    long_value = "x" * 100
    call = ToolCall(id="c1", name="t", arguments={"text": long_value})
    assert _target_for(call) == "x" * 40


def test_compact_summary_one_line_per_tool_call_message_using_first_call():
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": "/x/y.py"}),
                ToolCall(id="c2", name="ignored_second_call", arguments={}),
            ],
        ),
        Message(role="user", content="tool result placeholder"),
    ]
    assert _compact_summary(messages) == "1: read_file y.py"


def test_compact_summary_empty_when_no_tool_calls():
    messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    assert _compact_summary(messages) == ""


# ---------------------------------------------------------------------------
# _check_cuts -- the real post-condition contract, called directly
# ---------------------------------------------------------------------------


def test_check_cuts_passes_for_strictly_increasing_in_range_cuts():
    result = _check_cuts(CutPoints(cuts=[2, 5]), message_count=8)
    assert result is None or result.passed is True


def test_check_cuts_passes_for_empty_cuts():
    result = _check_cuts(CutPoints(cuts=[]), message_count=8)
    assert result is None or result.passed is True


def test_check_cuts_rejects_non_monotonic():
    result = _check_cuts(CutPoints(cuts=[5, 3]), message_count=8)
    assert isinstance(result, PostConditionResult)
    assert result.passed is False
    assert result.message is not None


def test_check_cuts_rejects_duplicate_consecutive_cuts():
    result = _check_cuts(CutPoints(cuts=[3, 3]), message_count=8)
    assert isinstance(result, PostConditionResult)
    assert result.passed is False


def test_check_cuts_rejects_cut_at_or_above_message_count():
    result = _check_cuts(CutPoints(cuts=[8]), message_count=8)
    assert isinstance(result, PostConditionResult)
    assert result.passed is False


def test_check_cuts_rejects_cut_at_or_below_zero():
    result = _check_cuts(CutPoints(cuts=[0]), message_count=8)
    assert isinstance(result, PostConditionResult)
    assert result.passed is False


def test_check_cuts_failure_message_is_non_none_matching_the_real_invariant():
    # PostConditionResult's own invariant: passed=False implies message is not None
    # (ai_thread/postcondition.py's model_post_init). Confirms _check_cuts honors it.
    result = _check_cuts(CutPoints(cuts=[-1]), message_count=8)
    assert result is not None
    assert result.passed is False
    assert result.message is not None


# ---------------------------------------------------------------------------
# Subtask -- fake segment_fn injection
# ---------------------------------------------------------------------------


class _FakeSegmenter:
    """Records calls and returns a canned ``CutPoints``; no LLM involved."""

    def __init__(self, cuts: list[int]) -> None:
        self._cuts = cuts
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, summary: str, message_count: int) -> CutPoints:
        self.calls.append((summary, message_count))
        return CutPoints(cuts=self._cuts)


def test_subtask_valid_cuts_slice_into_correct_episodes():
    fake = _FakeSegmenter(cuts=[2, 4])
    builder = Subtask(segment_fn=fake)
    traj = _tool_traj(6, tool_at={0, 2, 4})

    result = asyncio.run(builder.build(traj))

    assert result == [
        Episode(trajectory_id="t1", start=0, end=2),
        Episode(trajectory_id="t1", start=2, end=4),
        Episode(trajectory_id="t1", start=4, end=6),
    ]
    assert len(fake.calls) == 1
    summary, message_count = fake.calls[0]
    assert message_count == 6
    assert summary == _compact_summary(traj.messages)


def test_subtask_empty_cuts_yields_one_whole_episode():
    fake = _FakeSegmenter(cuts=[])
    builder = Subtask(segment_fn=fake)
    traj = _tool_traj(5, tool_at={0})

    result = asyncio.run(builder.build(traj))

    assert result == [Episode(trajectory_id="t1", start=0, end=5)]


def test_subtask_empty_trajectory_yields_no_episodes_and_no_segment_fn_call():
    fake = _FakeSegmenter(cuts=[1])
    builder = Subtask(segment_fn=fake)

    result = asyncio.run(builder.build(_tool_traj(0)))

    assert result == []
    assert fake.calls == []


def test_subtask_trajectory_with_no_tool_calls_skips_segment_fn_call():
    # No tool-call messages -> empty compact summary -> whole-trajectory
    # episode without spending an LLM call (documented short-circuit).
    fake = _FakeSegmenter(cuts=[1])
    builder = Subtask(segment_fn=fake)
    traj = _tool_traj(4)

    result = asyncio.run(builder.build(traj))

    assert result == [Episode(trajectory_id="t1", start=0, end=4)]
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Defense-in-depth: build() re-validates cuts from ANY segment_fn (review fix)
# ---------------------------------------------------------------------------


def test_subtask_out_of_bounds_cuts_from_fake_raise_value_error():
    # cuts=[-1] would otherwise silently become Episode(start=0, end=-1) +
    # Episode(start=-1, end=8) -- a wrong-LOCATION slice via Python's
    # negative-index wraparound, not a harmless empty episode.
    fake = _FakeSegmenter(cuts=[-1])
    builder = Subtask(segment_fn=fake)
    traj = _tool_traj(8, tool_at={0})

    with pytest.raises(ValueError, match=r"-1") as exc_info:
        asyncio.run(builder.build(traj))

    assert "-1" in str(exc_info.value)
    assert "8" in str(exc_info.value)


def test_subtask_non_monotonic_cuts_from_fake_raise_value_error():
    fake = _FakeSegmenter(cuts=[5, 3])
    builder = Subtask(segment_fn=fake)
    traj = _tool_traj(8, tool_at={0})

    with pytest.raises(ValueError, match=r"\[5, 3\]"):
        asyncio.run(builder.build(traj))


def test_subtask_default_segment_fn_is_the_module_level_ai_function_never_called():
    # Constructing Subtask() with no segment_fn must not require
    # network/model config -- it only binds the already-built, I/O-free
    # module-level AIFunction object. Never invoked here.
    Subtask()


# ===========================================================================
# mining.py / mining.py
# ===========================================================================


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _traj(traj_id: str, n: int) -> Trajectory:
    messages = [Message(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}") for i in range(n)]
    return Trajectory(id=traj_id, messages=messages)


def _draft(name: str = "d", strategy: str = "preserve") -> SkillDraft:
    return SkillDraft(
        name=name,
        description=f"{name} description",
        body=f"{name} body",
        strategy=strategy,
        provenance=Provenance(trajectory_ids=["t1"], constituent_ids=["t1:0-1"], strategy=strategy),
    )


# ---------------------------------------------------------------------------
# mining.py -- _render_episode / _episode_id
# ---------------------------------------------------------------------------


def test_render_episode_includes_role_and_content_for_each_message_in_slice():
    traj = _traj("t1", 4)
    episode = Episode(trajectory_id="t1", start=1, end=3)
    text = _render_episode(traj, episode)
    assert "msg 1" in text
    assert "msg 2" in text
    assert "msg 0" not in text
    assert "msg 3" not in text


def test_episode_id_combines_trajectory_id_and_bounds():
    episode = Episode(trajectory_id="t7", start=2, end=5)
    assert _episode_id(episode) == "t7:2-5"


# ---------------------------------------------------------------------------
# mining.py -- _check_extracted_skill
# ---------------------------------------------------------------------------


def test_check_extracted_skill_passes_when_not_worth_extracting_regardless_of_fields():
    result = _check_extracted_skill(ExtractedSkill(worth_extracting=False))
    assert result is None or result.passed is True


def test_check_extracted_skill_passes_when_worth_extracting_with_full_fields():
    result = _check_extracted_skill(ExtractedSkill(worth_extracting=True, name="n", description="d", body="b"))
    assert result is None or result.passed is True


def test_check_extracted_skill_rejects_worth_extracting_with_blank_name():
    result = _check_extracted_skill(ExtractedSkill(worth_extracting=True, name="", description="d", body="b"))
    assert result is not None
    assert result.passed is False
    assert "name" in result.message


def test_check_extracted_skill_rejects_worth_extracting_with_all_blank_fields():
    result = _check_extracted_skill(ExtractedSkill(worth_extracting=True))
    assert result is not None
    assert result.passed is False
    assert "name" in result.message
    assert "description" in result.message
    assert "body" in result.message


# ---------------------------------------------------------------------------
# mining.py -- extract_skills, fake extract_fn per mode
# ---------------------------------------------------------------------------


class _FakeExtractor:
    """Records calls; returns a canned per-mode ``ExtractedSkill`` (or a scripted sequence)."""

    def __init__(self, by_mode: dict[str, ExtractedSkill]) -> None:
        self._by_mode = by_mode
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, episode_text: str, mode: str) -> ExtractedSkill:
        self.calls.append((episode_text, mode))
        return self._by_mode[mode]


def _found(name: str) -> ExtractedSkill:
    return ExtractedSkill(worth_extracting=True, name=name, description=f"{name} desc", body=f"{name} body")


def _not_found() -> ExtractedSkill:
    return ExtractedSkill(worth_extracting=False)


def test_extract_skills_preserve_mode_draft_when_worth_extracting():
    traj = _traj("t1", 3)
    episode = Episode(trajectory_id="t1", start=0, end=3)
    fake = _FakeExtractor({"preserve": _found("preserve-skill")})

    drafts = asyncio.run(extract_skills([traj], [episode], modes=("preserve",), extract_fn=fake))

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.name == "preserve-skill"
    assert draft.strategy == "preserve"
    assert draft.provenance.trajectory_ids == ["t1"]
    assert draft.provenance.constituent_ids == ["t1:0-3"]
    assert draft.provenance.strategy == "preserve"
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == "preserve"


def test_extract_skills_mode_returning_not_worth_extracting_yields_no_draft():
    traj = _traj("t1", 3)
    episode = Episode(trajectory_id="t1", start=0, end=3)
    fake = _FakeExtractor({"prune": _not_found()})

    drafts = asyncio.run(extract_skills([traj], [episode], modes=("prune",), extract_fn=fake))

    assert drafts == []
    assert len(fake.calls) == 1


def test_extract_skills_runs_every_requested_mode_per_episode():
    traj = _traj("t1", 3)
    episode = Episode(trajectory_id="t1", start=0, end=3)
    fake = _FakeExtractor({"preserve": _found("p"), "prune": _not_found()})

    drafts = asyncio.run(extract_skills([traj], [episode], modes=("preserve", "prune"), extract_fn=fake))

    assert [d.strategy for d in drafts] == ["preserve"]
    assert {mode for _, mode in fake.calls} == {"preserve", "prune"}


def test_extract_skills_multiple_episodes_each_get_all_modes():
    traj = _traj("t1", 6)
    episodes = [Episode(trajectory_id="t1", start=0, end=3), Episode(trajectory_id="t1", start=3, end=6)]
    fake = _FakeExtractor({"preserve": _found("p")})

    drafts = asyncio.run(extract_skills([traj], episodes, modes=("preserve",), extract_fn=fake))

    assert len(drafts) == 2
    assert len(fake.calls) == 2
    assert {c[1] for c in fake.calls} == {"preserve"}


# ---------------------------------------------------------------------------
# mining.py -- repair gating, both ways
# ---------------------------------------------------------------------------


def test_extract_skills_repair_runs_for_failed_trajectory():
    traj = _traj("t-fail", 3)
    episode = Episode(trajectory_id="t-fail", start=0, end=3)
    fake = _FakeExtractor({"repair": _found("fix")})

    drafts = asyncio.run(
        extract_skills(
            [traj],
            [episode],
            modes=("repair",),
            extract_fn=fake,
            failed_trajectory_ids=frozenset({"t-fail"}),
        )
    )

    assert len(drafts) == 1
    assert drafts[0].strategy == "repair"
    assert len(fake.calls) == 1


def test_extract_skills_repair_skipped_for_non_failed_trajectory():
    traj = _traj("t-ok", 3)
    episode = Episode(trajectory_id="t-ok", start=0, end=3)
    fake = _FakeExtractor({"repair": _found("fix")})

    drafts = asyncio.run(
        extract_skills(
            [traj],
            [episode],
            modes=("repair",),
            extract_fn=fake,
            failed_trajectory_ids=frozenset(),
        )
    )

    assert drafts == []
    assert fake.calls == []


def test_extract_skills_repair_gate_is_per_trajectory_not_global():
    traj_fail = _traj("t-fail", 3)
    traj_ok = _traj("t-ok", 3)
    episodes = [
        Episode(trajectory_id="t-fail", start=0, end=3),
        Episode(trajectory_id="t-ok", start=0, end=3),
    ]
    fake = _FakeExtractor({"repair": _found("fix")})

    drafts = asyncio.run(
        extract_skills(
            [traj_fail, traj_ok],
            episodes,
            modes=("repair",),
            extract_fn=fake,
            failed_trajectory_ids=frozenset({"t-fail"}),
        )
    )

    assert len(drafts) == 1
    assert drafts[0].provenance.trajectory_ids == ["t-fail"]
    assert len(fake.calls) == 1


def test_extract_skills_preserve_and_prune_ignore_failure_label():
    traj = _traj("t-fail", 3)
    episode = Episode(trajectory_id="t-fail", start=0, end=3)
    fake = _FakeExtractor({"preserve": _found("p"), "prune": _found("pr")})

    drafts = asyncio.run(
        extract_skills(
            [traj],
            [episode],
            modes=("preserve", "prune"),
            extract_fn=fake,
            failed_trajectory_ids=frozenset(),
        )
    )

    assert {d.strategy for d in drafts} == {"preserve", "prune"}


def test_extract_skills_repair_with_totally_unmatched_failed_ids_raises_value_error():
    # failed_trajectory_ids names no trajectory in this call at all -- a
    # likely typo, structurally indistinguishable from "no failures" if
    # silently accepted (repair would just run for zero episodes either
    # way), so reject eagerly rather than silently extracting nothing.
    traj = _traj("t-ok", 3)
    episode = Episode(trajectory_id="t-ok", start=0, end=3)
    fake = _FakeExtractor({"repair": _found("fix")})

    with pytest.raises(ValueError, match=r"failed_trajectory_ids"):
        asyncio.run(
            extract_skills(
                [traj],
                [episode],
                modes=("repair",),
                extract_fn=fake,
                failed_trajectory_ids=frozenset({"t-totally-unrelated"}),
            )
        )
    assert fake.calls == []


def test_extract_skills_repair_with_failed_ids_and_no_trajectories_returns_empty_without_raising():
    # Review fix: the total-mismatch guard must not misfire on a trivially
    # empty call. With no trajectories at all there is no id set to mistype
    # against, so `failed_trajectory_ids` cannot be evidence of a typo here
    # -- the call simply extracts nothing.
    fake = _FakeExtractor({"repair": _found("fix")})

    drafts = asyncio.run(
        extract_skills(
            [],
            [],
            modes=("repair",),
            extract_fn=fake,
            failed_trajectory_ids=frozenset({"t-from-a-broader-run"}),
        )
    )

    assert drafts == []
    assert fake.calls == []


def test_extract_skills_repair_with_partially_unmatched_failed_ids_still_works():
    # A partial mismatch is the documented superset use case: the failure
    # set covers one matching trajectory plus one id unrelated to this
    # call. Repair fires only for the matching trajectory; no error.
    traj_fail = _traj("t-fail", 3)
    traj_ok = _traj("t-ok", 3)
    episodes = [
        Episode(trajectory_id="t-fail", start=0, end=3),
        Episode(trajectory_id="t-ok", start=0, end=3),
    ]
    fake = _FakeExtractor({"repair": _found("fix")})

    drafts = asyncio.run(
        extract_skills(
            [traj_fail, traj_ok],
            episodes,
            modes=("repair",),
            extract_fn=fake,
            failed_trajectory_ids=frozenset({"t-fail", "t-elsewhere-not-in-this-call"}),
        )
    )

    assert len(drafts) == 1
    assert drafts[0].provenance.trajectory_ids == ["t-fail"]
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# mining.py -- validation / error paths
# ---------------------------------------------------------------------------


def test_extract_skills_rejects_empty_modes():
    with pytest.raises(ValueError, match="modes"):
        asyncio.run(extract_skills([], [], modes=()))


def test_extract_skills_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown"):
        asyncio.run(extract_skills([], [], modes=("bogus",)))


def test_extract_skills_rejects_episode_with_no_matching_trajectory():
    episode = Episode(trajectory_id="t-nonexistent", start=0, end=1)
    with pytest.raises(ValueError, match=r"not present in `trajectories`"):
        asyncio.run(extract_skills([], [episode], modes=("preserve",)))


def test_extract_skills_empty_episodes_yields_empty_list_no_calls():
    fake = _FakeExtractor({})
    drafts = asyncio.run(extract_skills([], [], modes=("preserve",), extract_fn=fake))
    assert drafts == []
    assert fake.calls == []


def test_extract_skills_all_modes_constant_matches_documented_set():
    assert set(MODES) == {"preserve", "prune", "repair"}


# ---------------------------------------------------------------------------
# mining.py -- defense-in-depth: a fake violating the post-condition raises
# ---------------------------------------------------------------------------


def test_extract_skills_fake_returning_worth_extracting_with_blank_body_raises_value_error():
    traj = _traj("t1", 3)
    episode = Episode(trajectory_id="t1", start=0, end=3)
    bad = ExtractedSkill(worth_extracting=True, name="n", description="d", body="")
    fake = _FakeExtractor({"preserve": bad})

    with pytest.raises(ValueError, match="body"):
        asyncio.run(extract_skills([traj], [episode], modes=("preserve",), extract_fn=fake))


def test_extract_skills_default_extract_fn_is_module_level_ai_functions_never_called():
    # Constructing/using extract_skills with no extract_fn must not require
    # network/model config when no episodes are given -- only binds the
    # already-built, I/O-free module-level AIFunction objects.
    drafts = asyncio.run(extract_skills([], [], modes=("preserve",)))
    assert drafts == []


# ---------------------------------------------------------------------------
# mining.py -- _check_groups
# ---------------------------------------------------------------------------


def test_check_groups_passes_for_exact_partition():
    result = _check_groups(ThemeGroups(groups={"a": [0, 1], "b": [2]}), draft_count=3)
    assert result is None or result.passed is True


def test_check_groups_rejects_missing_index():
    result = _check_groups(ThemeGroups(groups={"a": [0]}), draft_count=3)
    assert result is not None
    assert result.passed is False
    assert "missing" in result.message


def test_check_groups_rejects_duplicate_index():
    result = _check_groups(ThemeGroups(groups={"a": [0, 1], "b": [1, 2]}), draft_count=3)
    assert result is not None
    assert result.passed is False
    assert "duplicated" in result.message


def test_check_groups_rejects_out_of_range_index():
    result = _check_groups(ThemeGroups(groups={"a": [0, 1, 5]}), draft_count=3)
    assert result is not None
    assert result.passed is False
    assert "out_of_range" in result.message


# ---------------------------------------------------------------------------
# mining.py -- LLMThematicGrouper, fake group_fn injection
# ---------------------------------------------------------------------------


class _FakeGrouper:
    def __init__(self, groups: dict[str, list[int]]) -> None:
        self._groups = groups
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, summary: str, draft_count: int) -> ThemeGroups:
        self.calls.append((summary, draft_count))
        return ThemeGroups(groups=self._groups)


def test_thematic_grouper_two_clusters_no_singletons():
    drafts = [_draft("a"), _draft("b"), _draft("c"), _draft("d")]
    fake = _FakeGrouper({"Theme1": [0, 1], "Theme2": [2, 3]})
    grouper = LLMThematicGrouper(group_fn=fake)

    groups = asyncio.run(grouper.group(drafts))

    labels = {g.label for g in groups}
    assert labels == {"Theme1", "Theme2"}
    all_members = [m for g in groups for m in g.members]
    assert {m.name for m in all_members} == {"a", "b", "c", "d"}
    assert len(fake.calls) == 1
    assert fake.calls[0][1] == 4


def test_grouper_singleton_group_from_llm_folded_into_ungrouped():
    drafts = [_draft("a"), _draft("b"), _draft("c")]
    # "b" is left as its own size-1 cluster by the (fake) LLM.
    fake = _FakeGrouper({"Theme1": [0, 2], "Solo": [1]})
    grouper = LLMThematicGrouper(group_fn=fake)

    groups = asyncio.run(grouper.group(drafts))

    ungrouped = [g for g in groups if g.label == UNGROUPED_LABEL]
    assert len(ungrouped) == 1
    assert [m.name for m in ungrouped[0].members] == ["b"]
    themed = [g for g in groups if g.label != UNGROUPED_LABEL]
    assert len(themed) == 1
    assert {m.name for m in themed[0].members} == {"a", "c"}


def test_grouper_all_singletons_folded_into_one_combined_ungrouped_group():
    drafts = [_draft("a"), _draft("b"), _draft("c")]
    fake = _FakeGrouper({"A": [0], "B": [1], "C": [2]})
    grouper = LLMThematicGrouper(group_fn=fake)

    groups = asyncio.run(grouper.group(drafts))

    assert len(groups) == 1
    assert groups[0].label == UNGROUPED_LABEL
    assert {m.name for m in groups[0].members} == {"a", "b", "c"}


def test_grouper_single_draft_input_is_ungrouped_without_calling_group_fn():
    drafts = [_draft("solo")]
    fake = _FakeGrouper({})
    grouper = LLMThematicGrouper(group_fn=fake)

    groups = asyncio.run(grouper.group(drafts))

    assert groups == [SkillGroup(label=UNGROUPED_LABEL, members=drafts)]
    assert fake.calls == []


def test_grouper_empty_drafts_yields_no_groups_without_calling_group_fn():
    fake = _FakeGrouper({})
    grouper = LLMThematicGrouper(group_fn=fake)

    groups = asyncio.run(grouper.group([]))

    assert groups == []
    assert fake.calls == []


# ---------------------------------------------------------------------------
# mining.py -- defense-in-depth: a fake dropping/duplicating a draft raises
# ---------------------------------------------------------------------------


def test_grouper_fake_dropping_a_draft_raises_value_error():
    drafts = [_draft("a"), _draft("b"), _draft("c")]
    fake = _FakeGrouper({"Theme1": [0, 1]})  # index 2 dropped
    grouper = LLMThematicGrouper(group_fn=fake)

    with pytest.raises(ValueError, match="missing"):
        asyncio.run(grouper.group(drafts))


def test_grouper_fake_duplicating_a_draft_raises_value_error():
    drafts = [_draft("a"), _draft("b"), _draft("c")]
    fake = _FakeGrouper({"Theme1": [0, 1], "Theme2": [1, 2]})  # index 1 duplicated
    grouper = LLMThematicGrouper(group_fn=fake)

    with pytest.raises(ValueError, match="duplicated"):
        asyncio.run(grouper.group(drafts))


def test_grouper_fake_returning_out_of_range_index_raises_value_error():
    drafts = [_draft("a"), _draft("b")]
    fake = _FakeGrouper({"Theme1": [0, 1, 9]})
    grouper = LLMThematicGrouper(group_fn=fake)

    with pytest.raises(ValueError, match="out_of_range"):
        asyncio.run(grouper.group(drafts))


def test_grouper_default_group_fn_is_module_level_ai_function_never_called():
    LLMThematicGrouper()


def test_grouper_protocol_is_structurally_satisfied_by_the_thematic_grouper():
    # Real structural check: an object satisfying the `Grouper` protocol
    # must expose an async `group(drafts) -> list[SkillGroup]` -- not just
    # any attribute named "group". Verified by actually running it through
    # a `Grouper`-typed call site with a fake `group_fn`, plus an explicit
    # coroutine-function check on the bound method itself.
    thematic: Grouper = LLMThematicGrouper(group_fn=_FakeGrouper({"A": [0, 1]}))
    assert inspect.iscoroutinefunction(thematic.group)

    groups = asyncio.run(thematic.group([_draft("a"), _draft("b")]))
    assert all(isinstance(g, SkillGroup) for g in groups)


# ===========================================================================
# mining.py: trajectories_from_events
# ===========================================================================


_TID = ThreadId("t-1")


# ---------------------------------------------------------------------------
# trajectories_from_events -- message grain
# ---------------------------------------------------------------------------


def test_user_event_yields_user_message():
    events = [MessageUserEvent(thread_id=_TID, text="hello there")]
    traj = trajectories_from_events(events, trajectory_id="t1")

    assert isinstance(traj, Trajectory)
    assert traj.id == "t1"
    assert len(traj.messages) == 1
    assert traj.messages[0].role == "user"
    assert traj.messages[0].content == "hello there"
    assert traj.messages[0].tool_calls == []
    assert traj.messages[0].tool_results == []


def test_assistant_complete_yields_assistant_message_with_text():
    events = [
        MessageUserEvent(thread_id=_TID, text="do a thing"),
        MessageAssistantCompleteEvent(thread_id=_TID, content=[{"text": "sure, on it"}]),
    ]
    traj = trajectories_from_events(events, trajectory_id="t1")

    assert [m.role for m in traj.messages] == ["user", "assistant"]
    assert traj.messages[1].content == "sure, on it"


def test_tool_call_attached_to_owning_assistant_message():
    events = [
        MessageUserEvent(thread_id=_TID, text="read a.py"),
        MessageAssistantCompleteEvent(
            thread_id=_TID,
            content=[
                {"text": "let me check"},
                {"toolUse": {"toolUseId": "tu-1", "name": "Read", "input": {"file_path": "a.py"}}},
            ],
        ),
        ToolResultEvent(thread_id=_TID, tool_use_id="tu-1", status="success", content=[{"text": "contents"}]),
    ]
    traj = trajectories_from_events(events, trajectory_id="t1")

    assert [m.role for m in traj.messages] == ["user", "assistant", "user"]
    assistant = traj.messages[1]
    assert assistant.content == "let me check"
    assert len(assistant.tool_calls) == 1
    call = assistant.tool_calls[0]
    assert call.id == "tu-1"
    assert call.name == "Read"
    assert call.arguments == {"file_path": "a.py"}
    # The tool call does not itself carry a result.
    assert assistant.tool_results == []

    tool_result_message = traj.messages[2]
    assert tool_result_message.tool_calls == []
    assert len(tool_result_message.tool_results) == 1
    result = tool_result_message.tool_results[0]
    assert result.tool_call_id == "tu-1"
    assert result.status == "success"
    assert result.content == "contents"


def test_tool_result_error_status_preserved():
    events = [
        MessageAssistantCompleteEvent(
            thread_id=_TID,
            content=[{"toolUse": {"toolUseId": "tu-1", "name": "Bash", "input": {"command": "rm a.py"}}}],
        ),
        ToolResultEvent(thread_id=_TID, tool_use_id="tu-1", status="error", content=[{"text": "denied"}]),
    ]
    traj = trajectories_from_events(events, trajectory_id="t1")
    result = traj.messages[1].tool_results[0]
    assert result.status == "error"
    assert result.content == "denied"


def test_multiple_tool_calls_in_one_turn_all_attached():
    events = [
        MessageAssistantCompleteEvent(
            thread_id=_TID,
            content=[
                {"toolUse": {"toolUseId": "tu-1", "name": "Read", "input": {"file_path": "a.py"}}},
                {"toolUse": {"toolUseId": "tu-2", "name": "Read", "input": {"file_path": "b.py"}}},
            ],
        ),
    ]
    traj = trajectories_from_events(events, trajectory_id="t1")
    assert [c.id for c in traj.messages[0].tool_calls] == ["tu-1", "tu-2"]


def test_empty_event_list_yields_empty_trajectory():
    traj = trajectories_from_events([], trajectory_id="t1")
    assert traj.id == "t1"
    assert traj.messages == []


def test_reasoning_content_becomes_message_text():
    events = [
        MessageAssistantCompleteEvent(
            thread_id=_TID,
            content=[{"reasoningContent": {"reasoningText": {"text": "thinking..."}}}],
        ),
    ]
    traj = trajectories_from_events(events, trajectory_id="t1")
    assert traj.messages[0].content == "thinking..."
