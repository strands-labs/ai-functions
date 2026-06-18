"""ai_functions CLI — the ``ai-functions`` command installed by the package.

This subpackage holds the Typer app that powers the ``ai_functions`` console
script. The public API of the package does NOT re-export CLI internals
— user code should go through :mod:`ai_functions.discovery` and
:mod:`ai_functions.runner` instead. The only name exported from this module is
:func:`main`, which is the entry point referenced from
``pyproject.toml`` under ``[project.scripts]``.

Command surface (verbs modelled on ``docker`` / ``kubectl``):

- ``ai-functions server``         — start the coordinator endpoint in the
                             foreground; publish a runtime file; clean
                             up on exit.
- ``ai-functions server stop``    — read the runtime file, ask the server to
                             shut down.
- ``ai-functions server status``  — show the advertised URL, PID, uptime.
- ``ai-functions ps``             — list registered threads (one row per
                             :class:`ThreadInfo`).
- ``ai-functions attach <tid>``   — open the Textual TUI for a thread: live
                             event feed, input box for chat-shaped
                             threads (see :mod:`ai_functions.cli.attach`).
- ``ai-functions logs <tid>``     — non-TUI event dump; honours ``--follow``
                             and ``--since``.
- ``ai-functions notify <tid> <text>`` — one-shot ``notify`` (side channel,
                             no cycle).
- ``ai-functions submit <tid> <text>`` — ``submit`` one string prompt, block
                             until the cycle resolves, print the result
                             (``--json`` adds token usage and timing).
- ``ai-functions kill <tid>``     — ``terminate`` by default, ``--now`` for
                             ``terminate_now``.
- ``ai-functions run <script>``   — execute a user script whose module exposes
                             a main :class:`Spawnable` attribute (see
                             :mod:`ai_functions.cli.run_cmd`).

Every command reads the target coordinator URL from, in order:
``--url`` flag, ``AI_FUNCTIONS_COORDINATOR_URL`` env var, the runtime file.
"""

from __future__ import annotations


def main() -> int:
    """CLI entry point installed as the ``ai_functions`` console script.

    Returns:
        Process exit code. ``0`` on success; non-zero for user errors,
        discovery failures, or command-specific failures. The concrete
        mapping is:

        - ``0`` — command succeeded.
        - ``1`` — generic user error (bad argument, unknown thread id).
        - ``2`` — no coordinator could be discovered.
        - ``130`` — interrupted by the user (Ctrl-C).

    Ensures:
        Never raises; exceptions are caught and translated into exit
        codes + stderr messages formatted for human readers.
    """
    ...
