"""Host a persistent agent for the ``ai-functions`` CLI to drive.

The earlier examples create a handle and drive it from the same script.
This one instead *hosts* an agent and then blocks, so other processes —
in particular the ``ai-functions`` CLI — can discover it, send it
prompts, and watch its events.

The module exposes a single :class:`~ai_functions.protocols.Spawnable`
as ``main``, which is what ``ai-functions run`` looks for by default.

Run it as a hosted agent::

    # Terminal 1 — start a coordinator (the shared meeting point).
    ai-functions server

    # Terminal 2 — host this agent on that coordinator.
    ai-functions run examples/09_serve_cli.py

    # Terminal 3 — find the thread, talk to it, watch it.
    ai-functions ps
    ai-functions submit  thread-<id> "What is an octopus?"
    ai-functions attach  thread-<id>          # interactive TUI

``ai-functions run`` starts no initial cycle: the thread sits
``not_started`` until a peer drives it via ``submit`` (one blocking
cycle) or ``attach`` (type a prompt, press Enter).

Run it directly instead, and it behaves the same way via
:func:`ai_functions.serve` — it discovers the coordinator, registers the
agent, and blocks until Ctrl-C::

    ai-functions server                  # terminal 1
    python examples/09_serve_cli.py      # terminal 2 (equivalent to `run`)
"""

import ai_functions
from ai_functions import ai_function


@ai_function(str, structured_output=False)
def assistant(message: str) -> str:
    """{message}"""


# ``ai-functions run examples/09_serve_cli.py`` hosts this object. The
# attribute name ``main`` is the default ``--attr`` that ``run`` resolves.
main = assistant


if __name__ == "__main__":
    # Equivalent to `ai-functions run` on this file: register the agent on
    # the discovered coordinator and block until Ctrl-C. No initial cycle
    # is started, so the thread waits for a peer to drive it.
    ai_functions.serve(main, thread_name="assistant")
