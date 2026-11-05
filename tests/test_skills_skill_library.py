"""``skill_library`` -- the Skill model, ``SkillLibraryBackend``, ``LibraryStore``, and the ``Skill`` tool.

Four parts, in dependency order:

  * ``LibraryStore`` -- the on-disk state behind ``@skillify``. Pins: an
    atomic version switch, rollback, recovery after a restart, the
    cross-process round lock, and records that round-trip their arguments
    and event logs.
  * The ``Skill`` tool -- how an AI Function sees a skill library. The agent
    gets an index (name + description of every skill) and a ``Skill`` tool
    that returns one skill's body, the contract Claude Code uses. These
    tests pin the index, the tool, and ``with_skills``' refusal to hand a
    with-skill run nothing.
  * The Skill model -- parsing, saving and loading SKILL.md files and a
    directory of them.
  * ``SkillLibraryBackend`` -- the ``MemoryBackend`` over a directory of
    skills: schema construction, recall, search, save, delete,
    and consolidation.
"""

from __future__ import annotations

import fcntl
import threading
import time
from pathlib import Path

import pytest

from ai_functions import ai_function
from ai_functions.experimental.skills.skill_library import (
    INDEX,
    LibraryStore,
    SkillDeliveryError,
    SkillLibraryBackend,
    SkillToolbox,
    backend_from_dirs,
    load_skill,
    load_skill_library,
    save_skill,
    with_skills,
)
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types.graph import GradFeedback

# ── LibraryStore ────────────────────────────────────────────────────────────


def _write_skill(root: Path, name: str, description: str = "does x", body: str = "Step 1. Do x.") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return directory


def _record(store: LibraryStore, outcome: float | None = 1.0, args=("t",)):  # noqa: ANN001, ANN202
    return store.append_record(
        outcome=outcome, version=0, skills_available=[], skills_loaded=[], events=[], args=args, kwargs={}
    )


def test_a_fresh_store_is_version_zero_with_no_skills(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")

    assert store.current_version() == 0
    assert store.skill_dirs() == []


def test_publish_writes_a_new_version_and_switches_current(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    alpha = _write_skill(tmp_path / "src", "alpha")

    version = store.publish([alpha])

    assert version == 1 and store.current_version() == 1
    assert [d.name for d in store.skill_dirs()] == ["alpha"]
    assert (store.version_dir(1) / "alpha" / "SKILL.md").is_file()
    # The source is copied, not moved.
    assert alpha.is_dir()


def test_rollback_defaults_to_the_previous_version_and_is_logged(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    store.publish([_write_skill(tmp_path / "a", "alpha")])
    store.publish([_write_skill(tmp_path / "b", "beta")])

    assert store.rollback() == 1
    assert [d.name for d in store.skill_dirs()] == ["alpha"]
    assert store.rollback() == 0
    assert store.skill_dirs() == []
    assert [e["type"] for e in store.history()] == ["rollback", "rollback"]


def test_pruning_keeps_the_newest_versions_and_never_current(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib", keep_versions=2)
    for i in range(4):
        store.publish([_write_skill(tmp_path / f"s{i}", f"skill-{i}")])
    assert store.versions() == [3, 4]

    store.rollback(to=3)
    store.publish([_write_skill(tmp_path / "s5", "skill-5")])

    assert store.current_version() == 5
    assert 5 in store.versions() and len(store.versions()) == 2


def test_a_second_store_on_the_same_root_sees_everything(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    store.publish([_write_skill(tmp_path / "a", "alpha")])
    _record(store, 0.0)
    store.append_history({"type": "round", "round": 1, "window_end": 0})
    _record(store, 1.0)

    reopened = LibraryStore(tmp_path / "lib")

    assert reopened.current_version() == 1
    assert [r.seq for r in reopened.records()] == [0, 1]
    assert reopened.last_window_end() == 0
    assert [r.seq for r in reopened.new_records()] == [1]
    assert reopened.next_round() == 2
    # A new record continues the sequence rather than restarting it.
    assert _record(reopened).seq == 2


def test_two_writers_on_one_library_never_reuse_a_seq(tmp_path: Path):
    """Two stores on one root (two processes, or two decorations of one dir) interleave appends.

    ``seq`` must stay dense and unique, or a round's ``window_end`` would
    silently hide the other writer's later calls from every future round.
    """
    a = LibraryStore(tmp_path / "lib")
    b = LibraryStore(tmp_path / "lib")
    for _ in range(5):
        _record(a, 0.0)
    _record(b, 0.0)
    a.append_history({"type": "round", "round": 1, "window_end": 5})  # a round consumed seq 0..5
    for _ in range(3):
        _record(b, 0.0)
    _record(a, 0.0)

    assert [r.seq for r in a.records()] == list(range(10))
    assert [r.seq for r in a.new_records()] == [6, 7, 8, 9]


def test_many_appends_from_two_stores_stay_dense_and_a_fresh_store_continues(tmp_path: Path):
    """``_count_records`` caches ``(file_size, line_count)`` instead of re-scanning the whole file each
    call; alternating writers repeatedly force it to catch up on bytes the other store appended, which
    must still leave ``seq`` dense and unique, and a freshly opened store must pick up the right next one.
    """
    a = LibraryStore(tmp_path / "lib")
    b = LibraryStore(tmp_path / "lib")
    n = 40
    for i in range(n):
        _record(a if i % 2 == 0 else b, float(i % 2))

    seqs = [r.seq for r in a.records()]
    assert seqs == list(range(n))
    assert len(set(seqs)) == n

    fresh = LibraryStore(tmp_path / "lib")
    assert _record(fresh).seq == n


def test_arguments_round_trip_and_unpicklable_ones_are_not_replayable(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")

    ok = store.append_record(
        outcome=1.0, version=0, skills_available=[], skills_loaded=[], events=[], args=("t",), kwargs={"n": 2}
    )
    bad = store.append_record(
        outcome=1.0, version=0, skills_available=[], skills_loaded=[], events=[], args=(lambda: 1,), kwargs={}
    )

    assert ok.replayable is True
    assert store.load_args(ok.call_id) == (("t",), {"n": 2})
    assert bad.replayable is False


def test_records_reads_are_well_formed_while_another_store_appends(tmp_path: Path):
    """``records()`` takes the records lock (shared) while reading, so a reader racing an in-progress
    ``append_record`` (exclusive) never observes a half-written last line.
    """
    writer_store = LibraryStore(tmp_path / "lib")
    reader_store = LibraryStore(tmp_path / "lib")
    n = 150
    stop = threading.Event()
    errors: list[Exception] = []
    reads = 0

    def write() -> None:
        for i in range(n):
            _record(writer_store, float(i % 2))

    def read() -> None:
        nonlocal reads
        while not stop.is_set():
            try:
                records = reader_store.records()
            except Exception as exc:  # noqa: BLE001 - captured for the assertion below
                errors.append(exc)
                return
            reads += 1
            for r in records:
                if not isinstance(r.seq, int) or not isinstance(r.call_id, str) or not r.call_id:
                    errors.append(AssertionError(f"malformed record: {r!r}"))
                    return

    writer = threading.Thread(target=write)
    reader = threading.Thread(target=read)
    reader.start()
    writer.start()
    writer.join(timeout=10)
    stop.set()
    reader.join(timeout=10)

    assert errors == []
    assert reads > 0
    assert [r.seq for r in reader_store.records()] == list(range(n))


def test_the_round_lock_excludes_a_second_holder(tmp_path: Path):
    first = LibraryStore(tmp_path / "lib")
    second = LibraryStore(tmp_path / "lib")

    with first.round_lock() as got_first:
        with second.round_lock() as got_second:
            assert got_first is True
            assert got_second is False
    with second.round_lock() as got_again:
        assert got_again is True


def test_publish_if_current_publishes_when_the_version_matches(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    alpha = _write_skill(tmp_path / "a", "alpha")

    version = store.publish_if_current(0, [alpha])

    assert version == 1 and store.current_version() == 1


def test_publish_if_current_declines_and_writes_nothing_when_stale(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    store.publish([_write_skill(tmp_path / "a", "alpha")])
    versions_before = store.versions()

    result = store.publish_if_current(0, [_write_skill(tmp_path / "b", "beta")])

    assert result is None
    assert store.current_version() == 1
    # No new version directory was written, and CURRENT is untouched.
    assert store.versions() == versions_before


def test_publish_and_rollback_block_on_the_shared_version_lock(tmp_path: Path):
    """``_version_lock`` -- the lock ``publish``, ``publish_if_current`` and ``rollback`` share -- excludes
    a second holder and blocks (rather than fails) until it is released.
    """
    store = LibraryStore(tmp_path / "lib")
    other = LibraryStore(tmp_path / "lib")

    with store._version_lock():
        handle = other._versions_lock_path.open("a")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            handle.close()


def test_publish_blocks_until_a_concurrent_holder_releases_the_version_lock(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    other = LibraryStore(tmp_path / "lib")
    alpha = _write_skill(tmp_path / "a", "alpha")
    holding = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with other._version_lock():
            holding.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holding.wait(timeout=5)

    result: dict[str, int] = {}

    def do_publish() -> None:
        result["version"] = store.publish([alpha])

    publisher = threading.Thread(target=do_publish)
    publisher.start()
    time.sleep(0.2)
    assert publisher.is_alive()  # still blocked on the lock ``other`` holds

    release.set()
    publisher.join(timeout=5)
    holder.join(timeout=5)

    assert not publisher.is_alive()
    assert result["version"] == 1


def test_staged_write_never_touches_the_published_version(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    v = store.publish([_write_skill(tmp_path / "src", "alpha", body="old")])
    backend = SkillLibraryBackend(store.version_dir(v), actor_id="t", stage_root=store.pending_dir)
    backend._save(backend.param_for("alpha"), "new")
    assert (store.version_dir(v) / "alpha" / "SKILL.md").read_text().endswith("old\n")
    assert "new" in (store.pending_dir / "alpha" / "SKILL.md").read_text()


def test_second_revision_of_the_same_skill_replaces_the_first(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    v = store.publish([_write_skill(tmp_path / "src", "alpha", body="old")])
    backend = SkillLibraryBackend(store.version_dir(v), actor_id="t", stage_root=store.pending_dir)
    backend._save(backend.param_for("alpha"), "first")
    backend._save(backend.param_for("alpha"), "second")
    [(slug, path)] = store.take_pending(tmp_path / "taken")
    assert slug == "alpha" and "second" in (path / "SKILL.md").read_text()
    assert store.take_pending(tmp_path / "taken") == []


def test_a_staged_index_edit_stages_the_skill_and_keeps_its_staged_body(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    v = store.publish([_write_skill(tmp_path / "src", "alpha", body="old")])
    backend = SkillLibraryBackend(store.version_dir(v), actor_id="t", stage_root=store.pending_dir)
    backend._save(backend.param_for("alpha"), "new")
    backend._save(INDEX, ["alpha: sharper description"])
    staged = load_skill(store.pending_dir / "alpha")
    assert staged.description == "sharper description" and staged.body.strip() == "new"
    assert load_skill(store.version_dir(v) / "alpha").description == "does x"


def test_a_staged_index_edit_stages_only_the_skills_whose_description_changed(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    v = store.publish([_write_skill(tmp_path / "src", "alpha"), _write_skill(tmp_path / "src", "beta")])
    backend = SkillLibraryBackend(store.version_dir(v), actor_id="t", stage_root=store.pending_dir)
    lines, _ = backend._recall(INDEX)
    backend._save(INDEX, [line.replace("alpha: does x", "alpha: sharper description") for line in lines])
    [(slug, path)] = store.take_pending(tmp_path / "taken")
    assert slug == "alpha" and load_skill(path).description == "sharper description"


def _published_bytes(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_a_staged_backend_refuses_to_delete_from_the_published_version(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    v = store.publish([_write_skill(tmp_path / "src", "alpha", body="old")])
    backend = SkillLibraryBackend(store.version_dir(v), actor_id="t", stage_root=store.pending_dir)
    before = _published_bytes(store.version_dir(v))
    with pytest.raises(TypeError, match="read-only except for staged rewrites: delete"):
        backend.delete(backend.param_for("alpha"))
    assert _published_bytes(store.version_dir(v)) == before
    assert sorted(p.name for p in store.version_dir(v).iterdir()) == ["alpha"]


def test_take_pending_moves_into_the_given_directory_and_nowhere_else(tmp_path: Path):
    store = LibraryStore(tmp_path / "lib")
    store.stage_revision("alpha", _write_skill(tmp_path / "src", "alpha", body="new"))
    [(slug, path)] = store.take_pending(tmp_path / "dest")
    assert (slug, path) == ("alpha", tmp_path / "dest" / "alpha")
    assert list((store.root / "rounds").iterdir()) == []
    assert list(store.pending_dir.iterdir()) == []


@ai_function[str](structured_output=False)
def _echo(task: str) -> str:
    """Echo: {task}"""


async def test_event_logs_round_trip(tmp_path: Path):
    result = await _echo.replace(model=ScriptedModel([Turn(text="hello there")])).trace(task="t")
    events = list(await result.coordinator.get_events(result.thread_id))
    store = LibraryStore(tmp_path / "lib")

    record = store.append_record(
        outcome=1.0, version=0, skills_available=[], skills_loaded=[], events=events, args=("t",), kwargs={}
    )
    loaded = store.load_events(record.call_id)

    assert loaded is not None
    assert [e.kind for e in loaded] == [e.kind for e in events]
    assert store.load_events("missing") is None


# ── The Skill tool ───────────────────────────────────────────────────────────


@ai_function[str](structured_output=False, system_prompt="Be brief.")
def _solve(task: str) -> str:
    """Solve: {task}"""


def test_index_lists_every_skill_with_its_description(tmp_path: Path):
    _write_skill(tmp_path, "alpha", "alpha desc")
    _write_skill(tmp_path, "beta", "beta desc")

    toolbox = SkillToolbox(SkillLibraryBackend(tmp_path, actor_id="t"))

    assert toolbox.available == ["alpha", "beta"]
    assert "- alpha: alpha desc" in toolbox.index_text()
    assert "- beta: beta desc" in toolbox.index_text()


async def test_skill_tool_returns_the_body_and_records_the_load(tmp_path: Path):
    _write_skill(tmp_path, "alpha", body="Heat it slowly.")
    toolbox = SkillToolbox(SkillLibraryBackend(tmp_path, actor_id="t"))
    tool = toolbox.make_tool()

    out = await tool(skill="alpha")

    assert tool.tool_name == "Skill"
    assert "Heat it slowly." in out
    assert toolbox.loaded == ["alpha"]


async def test_unknown_skill_name_lists_the_available_ones_and_records_nothing(tmp_path: Path):
    _write_skill(tmp_path, "alpha")
    toolbox = SkillToolbox(SkillLibraryBackend(tmp_path, actor_id="t"))

    out = await toolbox.make_tool()(skill="nope")

    assert "No skill named 'nope'" in out and "alpha" in out
    assert toolbox.loaded == []


def test_backend_from_dirs_copies_into_root_and_returns_none_for_no_dirs(tmp_path: Path):
    alpha = _write_skill(tmp_path / "src", "alpha")

    assert backend_from_dirs([], tmp_path / "empty", actor_id="t") is None
    backend = backend_from_dirs([alpha], tmp_path / "lib", actor_id="t")

    assert backend is not None
    assert (tmp_path / "lib" / "alpha" / "SKILL.md").is_file()


def test_backend_from_dirs_onto_a_reused_root_holds_exactly_the_new_dirs(tmp_path: Path):
    """A root reused across runs must not leak the previous run's skills into the next."""
    alpha = _write_skill(tmp_path / "src", "alpha")
    beta = _write_skill(tmp_path / "src", "beta")
    gamma = _write_skill(tmp_path / "src", "gamma")
    root = tmp_path / "lib"

    backend_from_dirs([alpha, beta], root, actor_id="t")
    backend = backend_from_dirs([alpha, gamma], root, actor_id="t")

    assert backend is not None
    assert SkillToolbox(backend).available == ["alpha", "gamma"]
    assert sorted(p.name for p in root.iterdir()) == ["alpha", "gamma"]


def test_backend_from_dirs_rejects_a_directory_without_skill_md(tmp_path: Path):
    (tmp_path / "bare").mkdir()

    with pytest.raises(FileNotFoundError):
        backend_from_dirs([tmp_path / "bare"], tmp_path / "lib", actor_id="t")


def test_with_skills_none_returns_the_function_unchanged():
    variant, toolbox = with_skills(_solve, None)

    assert variant is _solve
    assert toolbox is None


def test_with_skills_appends_the_index_to_the_system_prompt_and_adds_the_tool(tmp_path: Path):
    _write_skill(tmp_path, "alpha", "alpha desc")

    variant, toolbox = with_skills(_solve, SkillLibraryBackend(tmp_path, actor_id="t"))

    assert toolbox is not None
    prompt = variant.config.system_prompt or ""
    assert prompt.startswith("Be brief.")
    assert "- alpha: alpha desc" in prompt
    assert len(variant.config.tools) == len(_solve.config.tools) + 1
    # The original template is untouched.
    assert _solve.config.system_prompt == "Be brief."


def test_with_skills_refuses_an_empty_library(tmp_path: Path):
    with pytest.raises(SkillDeliveryError):
        with_skills(_solve, SkillLibraryBackend(tmp_path, actor_id="t"))


# ── Skill model ───────────────────────────────────────────────────────────

SKILL_MD = """---
name: csv-report
description: Build CSV summary reports.
---

## Steps

1. Load the CSV.
"""


def _write_skill_md(root: Path, slug: str, text: str = SKILL_MD) -> Path:
    d = root / slug
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text)
    return d


def test_load_skill_parses_frontmatter_and_body(tmp_path):
    path = _write_skill_md(tmp_path, "csv-report")
    skill = load_skill(path)
    assert skill.name == "csv-report"
    assert skill.description == "Build CSV summary reports."
    assert "## Steps" in skill.body


def test_save_skill_round_trips(tmp_path):
    skill = load_skill(_write_skill_md(tmp_path, "csv-report"))
    out = tmp_path / "out"
    save_skill(skill, out)
    again = load_skill(out / "csv-report")
    assert again.frontmatter == skill.frontmatter
    assert again.body == skill.body


def test_library_by_name(tmp_path):
    _write_skill_md(tmp_path, "csv-report")
    _write_skill_md(tmp_path, "pdf-extract", SKILL_MD.replace("csv-report", "pdf-extract"))
    lib = load_skill_library(tmp_path)
    assert len(lib) == 2
    assert lib.by_name("pdf-extract") is not None


# ── SkillLibraryBackend ──────────────────────────────────────────────────────


@pytest.fixture
def backend(tmp_path):
    _write_skill_md(tmp_path, "csv-report")
    _write_skill_md(
        tmp_path,
        "pdf-extract",
        SKILL_MD.replace("csv-report", "pdf-extract").replace("Build CSV summary reports.", "Pull tables out of PDFs."),
    )
    return SkillLibraryBackend(tmp_path, actor_id="test")


def test_schema_has_one_param_per_skill_plus_index(backend):
    names = set(backend.schema.model_fields)
    assert names == {INDEX, "csv_report", "pdf_extract"}


def test_recall_index_and_body(backend):
    value, _ = backend._recall(INDEX)
    assert "csv-report: Build CSV summary reports." in value
    body, _ = backend._recall("csv_report")
    assert "## Steps" in body


def test_search_index_returns_slug_keyed_results(backend):
    matches, meta = backend._search(INDEX, "pdf tables", k=1)
    assert list(meta["results"]) == ["pdf-extract"]
    with pytest.raises(TypeError):
        backend._search("csv_report", "anything")


def test_save_body_preserves_frontmatter(backend, tmp_path):
    backend._save("csv_report", "## New steps\n\n1. Different.")
    body, _ = backend._recall("csv_report")
    assert body.startswith("## New steps")
    value, _ = backend._recall(INDEX)
    assert "csv-report: Build CSV summary reports." in value


def test_delete_removes_param_and_dir(backend, tmp_path):
    backend._delete("pdf_extract")
    assert not (tmp_path / "pdf-extract").exists()
    assert "pdf_extract" not in backend.schema.model_fields


def test_save_body_writes_back_to_slug_dir_even_when_frontmatter_name_differs(tmp_path):
    # Directory slug is "csv-report", but frontmatter `name` disagrees with
    # it. A body save must land back in the directory it was read from
    # (the binding identity), not a second directory named after `name`.
    mismatched = SKILL_MD.replace("name: csv-report", "name: totally-different-name")
    _write_skill_md(tmp_path, "csv-report", mismatched)
    backend = SkillLibraryBackend(tmp_path, actor_id="test")

    backend._save("csv_report", "## New steps\n\n1. Different.")

    body, _ = backend._recall("csv_report")
    assert body.startswith("## New steps")
    assert not (tmp_path / "totally-different-name").exists()
    assert (tmp_path / "csv-report" / "SKILL.md").exists()


def test_save_index_refreshes_schema_descriptions(backend):
    assert backend.schema.model_fields["csv_report"].description == "Build CSV summary reports."

    backend._save(INDEX, ["csv-report: New description after edit."])

    assert backend.schema.model_fields["csv_report"].description == "New description after edit."


def test_consolidate_body_rewrites_and_persists(tmp_path):
    _write_skill_md(tmp_path, "csv-report")
    calls = {}

    def fake(value, feedback, description=""):
        calls["feedback"] = feedback
        return "## Rewritten"

    b = SkillLibraryBackend(tmp_path, actor_id="t", consolidate_fn=fake)
    b._consolidate("csv_report", [GradFeedback(text="add a validation step", score=0.0)])
    # save_skill always appends exactly one trailing newline (a
    # skill_library.py contract) — assert against the true persisted
    # value rather than the fake's raw return.
    assert b._recall("csv_report")[0] == "## Rewritten\n"
    assert calls["feedback"] == ["add a validation step"]


def test_scan_slugs_raises_on_index_collision(tmp_path):
    # A skill dir named "skill-index" sanitizes to "skill_index" -- the same
    # param name as the reserved INDEX field -- which would otherwise
    # silently overwrite the index field definition in the built schema.
    _write_skill_md(tmp_path, "skill-index", SKILL_MD.replace("csv-report", "skill-index"))

    with pytest.raises(ValueError, match="skill_index|skill-index"):
        SkillLibraryBackend(tmp_path, actor_id="test")


def test_scan_slugs_raises_on_sanitization_collision(tmp_path):
    # Two distinct slugs, "csv-report" and "csv_report", both sanitize to
    # the same param name "csv_report".
    _write_skill_md(tmp_path, "csv-report")
    _write_skill_md(tmp_path, "csv_report", SKILL_MD.replace("csv-report", "csv_report"))

    with pytest.raises(ValueError, match="csv_report"):
        SkillLibraryBackend(tmp_path, actor_id="test")


def test_consolidate_index_touches_only_retrieved(tmp_path):
    _write_skill_md(tmp_path, "csv-report")
    _write_skill_md(
        tmp_path,
        "pdf-extract",
        SKILL_MD.replace("csv-report", "pdf-extract").replace("Build CSV summary reports.", "Pull tables out of PDFs."),
    )

    def fake(value, feedback, description=""):
        return "csv-report: Summarize CSVs with totals."

    b = SkillLibraryBackend(tmp_path, actor_id="t", consolidate_fn=fake)
    b._consolidate(
        INDEX,
        [GradFeedback(text="mention totals", score=0.0)],
        retrieved={"csv-report": "csv-report: Build CSV summary reports."},
    )
    value, _ = b._recall(INDEX)
    assert "csv-report: Summarize CSVs with totals." in value
    assert "pdf-extract: Pull tables out of PDFs." in value
