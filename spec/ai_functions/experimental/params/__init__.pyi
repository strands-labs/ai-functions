"""Local parameters: trainable values that live in the process, declared on a signature.

A local parameter is a value that lives in the process, is declared on an ``@ai_function`` or
``@ai_method`` signature, is a gradient target for ``TextGradOptimizer``, and is not a memory entry.
It is the second answer to "where can a trainable value come from": ``methods/recall.py`` fills a
parameter from a ``MemoryBackend``; this package fills it from a default on the signature and keeps
it in a :class:`Local` host.

    @ai_function
    def research(query: str, instructions: Trainable[str] = "Find relevant facts.") -> str:
        '''Instructions: {instructions}. Research: {query}.'''

    bound = Bind(research)
    result = await bound.trace(query="cats")
    await TextGradOptimizer(model=m).step(result, "Cite sources.", backends=[bound.local])

One annotation and one default give a trainable parameter with sensible defaults; every knob is
optional (``Annotated[str, Trainable(reducer=ORDERED, conflict="merge")]``). The combining rule
is a :class:`FeedbackReducer` declared per parameter (:mod:`.reducers`, FR1 to FR4); versions and
the conflict policy keep two learners on one parameter from losing a write (:mod:`.local`, LP1 to
LP8). Nothing here touches a memory backend, and the core learns no new class: the host joins the
backward pass through the structural ``ParameterHost`` seam.
"""

from .local import (
    Bind,
    Conflict,
    Local,
    Param,
    ParamWiringError,
    Trainable,
    VersionConflictError,
    trainable_params,
)
from .reducers import (
    NUMERIC_ADD,
    ORDERED,
    CallableReducer,
    FeedbackReducer,
    ParamsError,
    ReducerError,
    as_reducer,
    default_reducer,
)

__all__ = [
    "Bind",
    "CallableReducer",
    "Conflict",
    "FeedbackReducer",
    "Local",
    "NUMERIC_ADD",
    "ORDERED",
    "Param",
    "ParamWiringError",
    "ParamsError",
    "ReducerError",
    "Trainable",
    "VersionConflictError",
    "as_reducer",
    "default_reducer",
    "trainable_params",
]
