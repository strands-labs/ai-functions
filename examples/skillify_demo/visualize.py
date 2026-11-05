"""Capture ScienceWorld once and inspect its recorded model responses.

From this directory:
    python visualize.py capture -- --smoke --out ./out-capture
    python visualize.py view --data ./out-capture/inspection

Only ``capture`` runs the demo and needs its model credentials and environment.
``view`` replays saved facts without running an agent or simulator.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Support both direct scripts and module invocation without installing the viewer.
for directory in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("capture", "view"))
    args, remaining = parser.parse_known_args(argv)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if args.command == "view":
        from skills_viz.viewer import main as view

        view(remaining)
        return
    import main as demo
    from skills_viz.capture import Recorder

    from ai_functions.experimental.skills import observe_skills, skill_event

    options = demo._parse_args(remaining)
    destination = Path(options.out).resolve() / "inspection"
    if (destination / "skills.sqlite3").exists():
        parser.error(f"{destination} already contains a capture; use a fresh --out")
    recorder = Recorder(destination)
    with observe_skills(recorder):
        skill_event("session.started", title="ScienceWorld · skillify", mode="Real model capture", config=vars(options))
        try:
            asyncio.run(demo.main(options))
        except BaseException as exc:
            skill_event("session.finished", status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            skill_event("session.finished", status="completed")
    print(f"Recording: {destination}\nInspect: python visualize.py view --data {destination}", flush=True)


if __name__ == "__main__":
    main()
