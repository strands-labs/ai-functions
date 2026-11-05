"""Portable skill inspection archives, source maps, and deterministic replay.

Only the capture callback touches live skill files. Every source revision is
frozen into the journal before temporary candidates or old versions disappear.
The viewer consumes this archive without importing an agent.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

_SKIP = {"message_assistant_token", "message_assistant_thinking"}


def source_map(raw: str) -> dict:
    """Map headings and instruction blocks to exact, one-based SKILL.md lines."""
    lines = raw.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        start = next((i + 1 for i in range(1, len(lines)) if lines[i].strip() == "---"), 0)
    headings, numbered, fence = [], [], None
    for i in range(start, len(lines)):
        line = lines[i]
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            continue
        if fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            headings.append((i, heading[2].strip(), len(heading[1])))
        step = re.match(r"^(\s*)(\d+)[.)]\s+(.+)", line)
        if step:
            numbered.append((i, step[3].strip(), len(step[1].expandtabs(4)), step[2]))
    sections = [
        {
            "title": title,
            "line_start": i + 1,
            "line_end": next((j for j, _, depth in headings[n + 1 :] if depth <= level), len(lines)),
        }
        for n, (i, title, level) in enumerate(headings)
    ]
    instruction = next(
        (s for s in sections if re.search(r"instructions|procedure|steps|workflow", s["title"], re.I)), None
    )
    lo, hi = (instruction["line_start"], instruction["line_end"]) if instruction else (start, len(lines))
    candidates = [(i, title, indent, number) for i, title, indent, number in numbered if lo <= i < hi]
    outer_indent = min((indent for _, _, indent, _ in candidates), default=0)
    points = [(i, title) for i, title, indent, _ in candidates if indent == outer_indent]
    numbers = {i: number for i, _, indent, number in candidates if indent == outer_indent}
    if not points:
        points = [(i, title) for i, title, _ in headings if lo <= i < hi]
    if not points and any(line.strip() for line in lines[lo:hi]):
        points = [(lo, "Instruction text")]
    steps = []
    for n, (i, title) in enumerate(points):
        end = points[n + 1][0] if n + 1 < len(points) else hi
        if i in numbers:
            for j in range(i + 1, end):
                indent = len(lines[j].expandtabs(4)) - len(lines[j].expandtabs(4).lstrip())
                if lines[j].strip() and not lines[j - 1].strip() and indent <= outer_indent:
                    end = j
                    break
        while end > i + 1 and not lines[end - 1].strip():
            end -= 1
        steps.append(
            {
                "id": f"L{i + 1}",
                "title": title,
                "number": numbers.get(i),
                "line_start": i + 1,
                "line_end": end,
                "text": "\n".join(lines[i:end]),
            }
        )
    return {"lines": lines, "sections": sections, "steps": steps}


def snapshot(skill: Mapping[str, Any], library: str = "") -> dict:
    """Freeze a source revision; identical slugs in different libraries stay separate."""
    raw = str(skill.get("raw", ""))
    body = str(skill.get("body", raw))
    front = skill.get("frontmatter", {})
    if not front:
        match = re.match(r"^---\s*\n(.*?)\n---", raw, re.S)
        if match:
            try:
                import yaml

                parsed = yaml.safe_load(match[1])
                front = parsed if isinstance(parsed, dict) else {}
            except ImportError:
                pass
    path = str(skill.get("path", ""))
    directory = Path(path).name if path and not path.startswith("<") else ""
    slug = str(skill.get("slug") or directory or front.get("name") or "skill")
    fingerprint = hashlib.sha256(raw.encode()).hexdigest()
    identity = hashlib.sha256(f"{library}\0{slug}\0{fingerprint}".encode()).hexdigest()[:24]
    return {
        "id": identity,
        "slug": slug,
        "name": front.get("name", slug),
        "description": front.get("description", skill.get("description", "")),
        "library": library,
        "fingerprint": fingerprint,
        "raw": raw,
        "body": body,
        "filename": skill.get("filename", "SKILL.md"),
        "frontmatter": front,
        "resources": skill.get("resources", []),
        **source_map(raw),
    }


def _read_skill(path: str | Path, library: str) -> dict:
    path = Path(path)
    source = path / "SKILL.md" if path.is_dir() else path
    package = source.name == "SKILL.md"
    raw = source.read_text(encoding="utf-8")
    # Capture works standalone, including on machines without the agent's dependencies.
    body = raw
    front: dict = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.S)
    if match:
        body = match[2]
        try:
            import yaml

            parsed = yaml.safe_load(match[1])
            front = parsed if isinstance(parsed, dict) else {}
        except ImportError:
            front = dict(re.findall(r"^(name|description):\s*(.*)$", match[1], re.M))
    return snapshot(
        {
            "path": str(path),
            "slug": source.parent.name if package else source.stem,
            "raw": raw,
            "body": body,
            "filename": source.name,
            "frontmatter": front,
            "resources": sorted(
                str(p.relative_to(source.parent)) for p in source.parent.rglob("*") if p.is_file() and p != source
            )
            if package
            else [],
        },
        library,
    )


def _freeze(value: Any, library: str) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if "raw" in value and "body" in value and ("frontmatter" in value or "slug" in value):
            return snapshot(value, library)
        return {k: v if k in {"event", "events"} else _freeze(v, library) for k, v in value.items()}
    if isinstance(value, list):
        return [_freeze(v, library) for v in value]
    return value


def _sources(value: Any):
    if isinstance(value, dict):
        if "fingerprint" in value and "lines" in value:
            yield value
        else:
            for key, child in value.items():
                if key not in {"event", "events"}:
                    yield from _sources(child)
    elif isinstance(value, list):
        for child in value:
            yield from _sources(child)


class Recorder:
    """A synchronous ``observe_skills`` callback with an append-only SQLite journal."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "skills.sqlite3"
        self._lock = threading.RLock()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS journal (seq INTEGER PRIMARY KEY, stamp REAL, data TEXT NOT NULL)")

    def __call__(self, event: dict) -> None:
        library = str(event.get("library") or "")
        frozen = _freeze(event, library)
        dirs = event.get("skill_dirs")
        if dirs is None and event.get("skill_root"):
            root = Path(event["skill_root"])
            dirs = sorted(p for p in root.iterdir() if (p / "SKILL.md").is_file()) if root.is_dir() else []
        if dirs is not None:
            frozen["skills"] = [_read_skill(path, library) for path in dirs]
        frozen.setdefault("timestamp", time.time())
        with self._lock, closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.execute(
                "INSERT INTO journal(stamp,data) VALUES(?,?)",
                (frozen["timestamp"], json.dumps(frozen, ensure_ascii=False, default=str)),
            )

    def journal(self, after: int = 0, until: int | None = None) -> list[dict]:
        with self._lock, closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT seq,data FROM journal WHERE seq>? AND (? IS NULL OR seq<=?) ORDER BY seq",
                (after, until, until),
            ).fetchall()
        return [{"seq": seq, **json.loads(data)} for seq, data in rows]

    def state(self, until: int | None = None) -> dict:
        return project(self.journal(until=until))

    def export(self, path: str | Path) -> None:
        """Export a self-contained JSON recording, including all captured source text."""
        Path(path).write_text(json.dumps({"format": "skillify-observation-v1", "events": self.journal()}))

    def import_json(self, path: str | Path) -> None:
        """Import a portable recording into an empty archive without consulting source paths."""
        if self.journal():
            raise ValueError("Import requires an empty archive")
        recording = json.loads(Path(path).read_text())
        if recording.get("format") != "skillify-observation-v1":
            raise ValueError("Unknown recording format")
        with closing(sqlite3.connect(self.path)) as db, db:
            for event in recording["events"]:
                db.execute(
                    "INSERT INTO journal(seq,stamp,data) VALUES(?,?,?)",
                    (event["seq"], event["timestamp"], json.dumps(event)),
                )

    def import_skill(self, path: str | Path) -> None:
        """Inspect a Markdown file or skill package without inventing execution history."""
        if self.journal():
            raise ValueError("Import requires an empty archive")
        source = Path(path).resolve()
        if not source.is_dir() and source.suffix.lower() != ".md":
            raise ValueError("Expected a Markdown file or a directory containing SKILL.md")
        library = str(source if source.is_dir() else source.parent)
        skill = _read_skill(source, library)
        self(
            {
                "kind": "session.started",
                "title": source.name,
                "mode": "Document inspection",
                "note": "Source only: no execution, load, induction, or evaluation history was provided.",
            }
        )
        self({"kind": "document.imported", "library": library, "source": str(source), "skill": skill})
        self({"kind": "session.finished", "status": "completed"})

    def import_library(self, path: str | Path) -> None:
        """Inspect existing completed runs without unpickling arguments or calling models.

        Earlier temporary candidates and intermediate stages cannot be reconstructed
        from the library alone. This limitation is preserved in the session metadata.
        """
        if self.journal():
            raise ValueError("Import requires an empty archive")
        root = Path(path).resolve()
        if not (root / "versions").is_dir() or not (root / "records").is_dir():
            raise ValueError("Expected a skillify library containing versions/ and records/")
        library = str(root)
        events = []
        versions = {
            int(p.name): sorted(d for d in p.iterdir() if (d / "SKILL.md").is_file())
            for p in (root / "versions").iterdir()
            if p.is_dir() and p.name.isdigit()
        }

        def stamp(value):
            return datetime.fromisoformat(value).timestamp()

        history_path = root / "history.jsonl"
        history = (
            [json.loads(line) for line in history_path.read_text().splitlines() if line]
            if history_path.exists()
            else []
        )
        for entry in history:
            if entry.get("type") == "round":
                rid = f"import-round-{entry['round']}"
                events.append(
                    {
                        "kind": "round.started",
                        "round_id": rid,
                        "report": {
                            "round": entry["round"],
                            "window_calls": entry["window_calls"],
                            "version_before": entry["version_before"],
                        },
                        "timestamp": stamp(entry["started_at"]),
                    }
                )
                events.append(
                    {
                        "kind": "history.appended",
                        "round_id": rid,
                        "entry": entry,
                        "timestamp": stamp(entry["finished_at"] or entry["started_at"]),
                    }
                )
                version = entry["version_after"]
                if version in versions:
                    events.append(
                        {
                            "kind": "library.published",
                            "version": version,
                            "skill_dirs": versions[version],
                            "timestamp": stamp(entry["finished_at"] or entry["started_at"]),
                        }
                    )
            else:
                events.append({"kind": "history.appended", "entry": entry, "timestamp": stamp(entry["at"])})
        calls = root / "records/calls.jsonl"
        for line in calls.read_text().splitlines() if calls.exists() else []:
            record = json.loads(line)
            run_id = record["call_id"]
            trace_path = root / "records/events" / f"{run_id}.json"
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
            base = {"run_id": run_id, "phase": "call", "version": record["version"]}
            start = min((e["timestamp"] for e in trace), default=stamp(record["timestamp"]))
            events.append(
                {
                    **base,
                    "kind": "run.started",
                    "timestamp": start,
                    "skill_dirs": versions.get(record["version"], []),
                    "source_missing": record["version"] not in versions and record["version"] != 0,
                }
            )
            events.extend(
                {**base, "kind": "run.event", "timestamp": e["timestamp"], "event": e}
                for e in trace
                if e["kind"] not in _SKIP
            )
            events.append({**base, "kind": "call.recorded", "record": record, "timestamp": stamp(record["timestamp"])})
            events.append(
                {
                    **base,
                    "kind": "run.finished",
                    "status": "failed" if record.get("error") else "completed",
                    "outcome": record["outcome"],
                    "timestamp": stamp(record["timestamp"]),
                }
            )
        self(
            {
                "kind": "session.started",
                "title": root.parent.name,
                "mode": "Imported completed runs",
                "note": (
                    "Intermediate mining stages and discarded candidate text were not recorded. "
                    "Missing/pruned versions remain unavailable."
                ),
                "timestamp": min((e["timestamp"] for e in events), default=time.time()),
            }
        )
        for event in sorted(events, key=lambda e: e["timestamp"]):
            self({**event, "library": library})
        current = int((root / "CURRENT").read_text()) if (root / "CURRENT").exists() else 0
        self(
            {"kind": "library.current", "library": library, "version": current, "skill_dirs": versions.get(current, [])}
        )
        self({"kind": "session.finished", "status": "completed"})

    def import_legacy(self, path: str | Path) -> None:
        """Convert the earlier ScienceWorld recorder's factual journal, preserving its traces."""
        if self.journal():
            raise ValueError("Import requires an empty archive")
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        # A copy lets SQLite recover WAL without writing beside the original archive.
        with tempfile.TemporaryDirectory(prefix="skills-legacy-") as temporary:
            copied = Path(temporary) / source.name
            for file in source.parent.glob(source.name + "*"):
                shutil.copy2(file, Path(temporary) / file.name)
            with closing(sqlite3.connect(copied)) as db:
                old = [json.loads(row[0]) for row in db.execute("SELECT data FROM journal ORDER BY seq")]
        library = str(source.parent.parent / "library")
        skills, scores, version, metadata = {}, {}, 0, {}
        names = {
            "session_started": "session.started",
            "session_finished": "session.finished",
            "library_updated": "library.published",
            "run_started": "run.started",
            "run_event": "run.event",
            "run_finished": "run.finished",
            "stage_skipped": "stage.skipped",
            "skills_discovered": "mining.finished",
            "candidate_ready": "candidate.ready",
            "candidate_formatted": "candidate.formatted",
            "draft_created": "draft.created",
            "skills_testing": "gate.started",
        }
        for original in old:
            kind = original["kind"]
            if kind == "archive_snapshot":
                archive = original.get("snapshot", {})
                if isinstance(archive, str):
                    archive = json.loads(archive)
                for collection, key, value in archive.get("documents", []):
                    if collection == "meta" and key == "session":
                        metadata.update(json.loads(value) if isinstance(value, str) else value)
                continue
            e = {**original, "library": library, "kind": names.get(kind, kind), "legacy_kind": kind}
            if "phase" in e:
                e["phase"] = {"traffic": "call", "trial": "replay", "gate": "replay"}.get(e["phase"], e["phase"])
            elif not kind.startswith("run_"):
                e["phase"] = "learning"
            if "round" in e:
                e["round_id"] = f"legacy-round-{e['round']}"
            if kind == "session_started":
                e.update(
                    title="Recorded ScienceWorld pilot",
                    mode="Synthetic sample" if metadata.get("sample") else "Previously recorded model responses",
                    source=str(source),
                    source_metadata=metadata.copy(),
                )
                if metadata.get("history_complete") is False:
                    e["note"] = "The original archive reports incomplete history."
            elif kind == "library_updated":
                e["kind"] = "library.current" if e["version"] == version else "library.published"
                version = e["version"]
            elif kind == "run_started":
                e.update(function=e.get("label", "play"), inputs=e.get("arguments", {}))
            elif kind == "run_scored":
                scores[e["run_id"]] = e.get("outcome")
                continue
            elif kind == "run_finished":
                e["outcome"] = scores.get(e["run_id"])
            elif kind == "round_started":
                e.update(
                    kind="round.started",
                    report={"round": e["round"], "version_before": e["version"], "window_calls": e["window_calls"]},
                )
            elif kind == "round_finished":
                e.update(kind="history.appended", entry=e["report"], round_id=f"legacy-round-{e['report']['round']}")
            elif kind == "activity":
                e["kind"] = "stage.started" if e.get("status") == "running" else "stage.finished"
                e["stage"] = {
                    "segmentation": "segment",
                    "extraction": "extract",
                    "grouping": "group",
                    "merging": "merge",
                    "formatting": "format",
                }.get(e.get("stage"), e.get("stage"))
            elif kind in {"quality_result", "trial_result"}:
                e["kind"] = "candidate.scored" if kind == "quality_result" else "candidate.evaluated"
                if e["name"] in skills:
                    e["skill"] = skills[e["name"]]
                if kind == "trial_result":
                    e.update(admitted=e["passed"], reason=e["verdict"]["rationale"], scores=e["verdict"]["scores"])
                else:
                    e["stage"] = "prefilter"
            for skill in e.get("skills", []) + ([e["skill"]] if "skill" in e else []):
                if isinstance(skill, dict):
                    skills[skill["slug"]] = skill
            # Preserve other recorded facts (environment progress and jobs) for inspection.
            self(e)


def project(events: list[dict]) -> dict:
    """Reduce a journal prefix. Future source revisions never leak into earlier frames."""
    state: dict = {"session": {}, "libraries": {}, "skills": {}, "runs": {}, "rounds": {}, "activity": [], "seq": 0}
    round_sources: dict[str, set[str]] = {}
    round_calls: dict[str, list[str]] = {}
    environments: dict[str, dict] = {}
    observations: dict[tuple[str, str], str] = {}
    observation_focus: dict[tuple[str, str], list[str]] = {}
    for event in events:
        kind, seq = event["kind"], event["seq"]
        library = event.get("library", "")
        state["seq"] = seq
        state["timestamp"] = event["timestamp"]
        if library:
            state["libraries"].setdefault(library, {"name": Path(library).parent.name, "version": 0, "versions": {}})
        found = list(_sources(event))
        for skill in found:
            state["skills"].setdefault(skill["id"], {**skill, "status": "candidate", "first_seq": seq})
            if kind == "document.imported":
                state["skills"][skill["id"]]["status"] = "document"
        scope = event.get("round_id")
        if kind == "mining.started" and scope:
            round_calls[scope] = event.get("call_ids", [])
        if scope and not kind.startswith(("run.", "library.")):
            round_sources.setdefault(scope, set()).update(s["id"] for s in found)
            for skill in found:
                state["skills"][skill["id"]].setdefault("source_call_ids", round_calls.get(scope, []))
        if kind.startswith("session."):
            state["session"].update({k: v for k, v in event.items() if k not in {"seq", "kind"}})
        environment_id = event.get("environment_id")
        if kind in {"environment_progress", "environment.progress"} and environment_id:
            observation = event.get("observation", "")
            observation_key = (environment_id, event.get("action", ""))
            previous = observations.get(observation_key)
            progress = dict(event)
            if isinstance(observation, str):
                observations[observation_key] = observation
                if previous is not None:
                    progress["changed_lines"] = [
                        line for line in observation.splitlines() if line.strip() and line not in previous.splitlines()
                    ]
                    progress["observation_unchanged"] = observation == previous
                    if progress["changed_lines"]:
                        observation_focus[observation_key] = progress["changed_lines"]
                    progress["observation_focus"] = [
                        line for line in observation_focus.get(observation_key, []) if line in observation.splitlines()
                    ]
            environments[environment_id] = progress
            for existing_run in state["runs"].values():
                if existing_run.get("environment_id") == environment_id:
                    existing_run["progress"] = progress
        rid = event.get("run_id")
        if rid:
            run = state["runs"].setdefault(
                rid, {"id": rid, "events": [], "skills": [], "status": "running", "started_seq": seq}
            )
            for key in (
                "phase",
                "function",
                "library",
                "version",
                "arm",
                "repeat",
                "source_call_id",
                "attempt",
                "inputs",
                "thread_id",
                "source_missing",
                "environment_id",
                "job_id",
                "task_id",
                "candidate",
                "activity_stage",
            ):
                if key in event:
                    run[key] = event[key]
            if kind == "run.started":
                run.update(skills=[s["id"] for s in found], status="running")
                run.setdefault("started", event["timestamp"])
                if environment_id in environments:
                    run["progress"] = environments[environment_id]
                if (
                    library
                    and event.get("version") is not None
                    and event.get("phase") != "replay"
                    and (found or event.get("phase") == "call")
                ):
                    lib = state["libraries"][library]
                    if not lib["versions"]:
                        lib["version"] = event["version"]
                    lib["versions"].setdefault(str(event["version"]), list(run["skills"]))
            if kind in {"run.event", "run.events"}:
                incoming = [event["event"]] if kind == "run.event" else event["events"]
                existing = {e.get("id") for e in run["events"]}
                run["events"].extend(e for e in incoming if e.get("id") not in existing and e.get("kind") not in _SKIP)
            if kind == "run.finished":
                run.update(
                    status=event.get("status", "completed"),
                    outcome=event.get("outcome"),
                    error=event.get("error"),
                    finished_seq=seq,
                )
            if kind == "call.recorded":
                run.update(record=event["record"], outcome=event["record"]["outcome"])
        round_id = event.get("round_id")
        if kind == "round.started":
            round_id = round_id or f"{library}:{event['report']['round']}"
            state["rounds"][round_id] = {
                "id": round_id,
                "library": library,
                **event["report"],
                "status": "running",
                "started_seq": seq,
            }
        if kind == "history.appended":
            entry = event["entry"]
            if entry.get("type") == "round":
                round_id = round_id or f"{library}:{entry['round']}"
                state["rounds"].setdefault(round_id, {"id": round_id, "library": library}).update(entry)
                for sid in round_sources.get(round_id, set()):
                    skill = state["skills"][sid]
                    for field, status in (("admitted", "stored"), ("revised", "stored"), ("rejected", "rejected")):
                        if skill["slug"] in entry.get(field, {}):
                            skill.update(status=status, verdict=entry[field][skill["slug"]], round=entry["round"])
            elif entry.get("type") == "rollback" and library:
                state["libraries"][library]["version"] = entry["to"]
        if kind == "round.cancelled" and round_id in state["rounds"]:
            state["rounds"][round_id]["status"] = "cancelled"
        if kind == "candidate.excluded":
            for skill in found:
                target = state["skills"][skill["id"]]
                if target["status"] == "stored":
                    continue
                target.update(
                    status="rejected", verdict={"stage": event.get("stage"), "rationale": event.get("reason")}
                )
        if kind in {"candidate.scored", "candidate.evaluated"}:
            for skill in found:
                state["skills"][skill["id"]]["verdict"] = {k: v for k, v in event.items() if k not in {"skill", "seq"}}
                if kind == "candidate.evaluated":
                    state["skills"][skill["id"]]["status"] = "accepted" if event["admitted"] else "rejected"
        if kind == "gate.finished":
            for verdict in event["verdicts"]:
                skill = verdict["candidate"]["skill"]
                state["skills"][skill["id"]].update(
                    status="accepted" if verdict["admitted"] else "rejected",
                    verdict={k: v for k, v in verdict.items() if k != "candidate"},
                )
        if kind == "skill.staged":
            for skill in found:
                state["skills"][skill["id"]]["status"] = "staged"
        if kind in {"library.published", "library.current"} and library:
            lib = state["libraries"][library]
            lib["version"] = event["version"]
            lib["versions"][str(event["version"])] = [s["id"] for s in found]
            for skill in found:
                state["skills"][skill["id"]]["status"] = "stored"
                for report in state["rounds"].values():
                    if report.get("library") == library and report.get("version_after") == event["version"]:
                        verdict = report.get("admitted", {}).get(skill["slug"]) or report.get("revised", {}).get(
                            skill["slug"]
                        )
                        if verdict:
                            state["skills"][skill["id"]]["verdict"] = verdict
        if kind not in {"run.event", "run.events", "call.recorded"}:
            draft = event.get("draft") or next(reversed(event.get("drafts") or []), None)
            if kind == "draft.created":
                draft = draft or event
            if isinstance(draft, dict) and isinstance(draft.get("body"), str):
                event = {**event, "draft_map": source_map(draft["body"])}
            state["activity"].append(event)
    for run in state["runs"].values():
        run["timeline"] = timeline(
            run["events"], [state["skills"][sid] for sid in run["skills"] if sid in state["skills"]]
        )
    return state


_STOP = set("the a an to of and or in on with from this that it is be use then for as at by".split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP


def _matches(action: str, loaded: list[dict]) -> list[dict]:
    tokens = _tokens(action)
    found = []
    for skill in loaded:
        for step in skill["steps"]:
            overlap = tokens & _tokens(step["text"])
            if len(overlap) >= 2 and len(overlap) / max(1, len(tokens)) >= 0.65:
                found.append(
                    {
                        "skill_id": skill["id"],
                        "slug": skill["slug"],
                        "line_start": step["line_start"],
                        "line_end": step["line_end"],
                        "method": "inferred",
                        "basis": "Shared words: " + ", ".join(sorted(overlap)),
                    }
                )
    return found


def timeline(events: list[dict], skills: list[dict]) -> list[dict]:
    """Pair actual tool results with requests; only successful, source-matching loads count."""
    by_slug = {s["slug"]: s for s in skills}
    pending, loaded, rows = {}, {}, []
    # A completed trace can supply early events missed by a live subscription.
    for event in sorted(events, key=lambda e: e.get("timestamp", 0)):
        kind = event.get("kind")
        if kind == "started":
            pending.clear()
            loaded.clear()
        elif kind == "tool_call":
            tool, args = event["tool_name"], event.get("arguments", {})
            text = str(args.get("skill", "")) if tool == "Skill" else str(args.get("action", json.dumps(args)))
            row = {
                "id": event["id"],
                "timestamp": event["timestamp"],
                "type": "load" if tool == "Skill" else "action",
                "tool": tool,
                "text": text,
                "arguments": args,
                "status": "running",
                "confirmed": False,
                "matches": [] if tool == "Skill" else _matches(text, list(loaded.values())),
            }
            pending[event.get("tool_use_id")] = row
            rows.append(row)
        elif kind == "tool_result" and event.get("tool_use_id") in pending:
            row = pending[event["tool_use_id"]]
            content = "\n".join(str(b.get("text", b.get("json", ""))) for b in event.get("content", []))
            row.update(status=event.get("status"), result=content, finished=event["timestamp"])
            if row["type"] == "load":
                skill = by_slug.get(row["text"].strip())
                expected = f"# {skill['slug']}\n\n{skill['body']}".strip() if skill else None
                if row["status"] == "success" and expected is not None and content.strip() == expected:
                    row.update(confirmed=True, skill_id=skill["id"])
                    loaded[skill["id"]] = skill
        elif kind in {"message_user", "message_assistant_complete", "failed", "cancelled"}:
            text = (
                event.get("text")
                or event.get("error")
                or "\n".join(b.get("text", "") for b in event.get("content", []) if isinstance(b, dict))
            )
            if text:
                rows.append(
                    {
                        "id": event["id"],
                        "timestamp": event["timestamp"],
                        "type": "message" if kind.startswith("message") else "error",
                        "text": text,
                        "matches": [],
                    }
                )
    return rows


def replay(events: list[dict]) -> dict:
    """Select recorded milestones for interactive playback, in journal order."""
    points: list[dict] = []
    requests: dict[tuple[str | None, str | None], str] = {}
    for event in events:
        kind, run_id = event["kind"], event.get("run_id")
        point = {key: event[key] for key in ("seq", "kind", "stage", "run_id") if key in event}
        if kind in {"run.event", "run.events"}:
            trace = [event["event"]] if kind == "run.event" else event["events"]
            for item in trace:
                key = (run_id, item.get("tool_use_id"))
                if item["kind"] == "tool_call":
                    requests[key] = item["id"]
                    point["entry_id"] = item["id"]
                elif item["kind"] == "tool_result" and key in requests:
                    point["entry_id"] = requests[key]
                elif item["kind"] in {"message_user", "message_assistant_complete", "failed", "cancelled"}:
                    point["entry_id"] = item["id"]
            if "entry_id" not in point:
                continue
        elif kind == "call.recorded":
            continue
        source = next(iter(_sources(event)), None)
        if source:
            point["skill_id"] = source["id"]
        point.update(at=len(points) * 2, duration=2)
        points.append(point)
    return {"duration": len(points) * 2, "points": points, "last_seq": events[-1]["seq"] if events else 0}
