"""Plain Python workflow as a Spawnable.

Any class that satisfies the ``Thread`` protocol (``name``, ``execute``,
``notify``, ``serialize_result``, ``deserialize_result``, ``fork``,
``teardown``) and the ``Spawnable`` protocol (``to_thread`` returning such
a thread, plus ``input_shape``) can be spawned by a worker. The workflow
uses ``ctx.coordinator`` to spawn AI children; their token usage rolls up
to the parent. The coordinator never inspects the type — it just calls
``to_thread()`` and drives the result.
"""

import asyncio
from typing import Self

from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.ai_thread import AIFunctionError
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import InputShape, ThreadContext


class AnalysisReport(BaseModel):
    topic: str
    sections: list[str]
    word_count: int


@ai_function(list[str])
def outline_generator(topic: str):
    """Generate 3 section titles for a report about: {topic}"""


@ai_function(str)
def section_writer(title: str, topic: str):
    """Write a short section titled '{title}' for a report about: {topic}"""


class ReportWorkflow:
    """Plain Python orchestration — no LLM in this layer.

    Only the outline generation and section writing use AI. The runtime
    manages everything as threads with proper parent-child relationships
    and token rollup.
    """

    name: str = "report_workflow"
    input_shape: InputShape = InputShape.STRUCTURED

    def to_thread(self) -> Self:
        # Already a live thread instance; the runtime calls this at spawn.
        return self

    async def execute(self, ctx: ThreadContext, topic: str) -> AnalysisReport:
        coord = ctx.coordinator

        outline_h = await coord.spawn(
            outline_generator,
            parent_id=ctx.thread_id,
            thread_name="outline",
        )
        try:
            sections = await outline_h.run(topic=topic)
        finally:
            await outline_h.terminate_now()

        writer_handles = [
            await coord.spawn(
                section_writer,
                parent_id=ctx.thread_id,
                thread_name=f"writer-{i}",
            )
            for i in range(len(sections))
        ]
        try:
            written = await asyncio.gather(
                *[h.run(title=title, topic=topic) for h, title in zip(writer_handles, sections, strict=True)]
            )
        finally:
            await asyncio.gather(*[h.terminate_now() for h in writer_handles])

        full_text = "\n\n".join(written)
        return AnalysisReport(
            topic=topic,
            sections=sections,
            word_count=len(full_text.split()),
        )

    async def notify(self, text: str) -> None:
        del text

    async def fork(self) -> Self:
        raise NotImplementedError

    async def teardown(self) -> None:
        pass

    def serialize_result(self, result: AnalysisReport) -> str:
        return result.model_dump_json()

    def deserialize_result(self, payload: str) -> AnalysisReport:
        try:
            return AnalysisReport.model_validate_json(payload)
        except Exception as e:  # noqa: BLE001 — protocol contract raises AIFunctionError
            raise AIFunctionError(f"Failed to decode AnalysisReport: {e}") from e


async def main() -> None:
    coord = InMemoryCoordinator()
    worker = LocalWorker(coord)
    await worker.register()

    handle = await worker.spawn_locally(ReportWorkflow(), thread_name="report-workflow")
    try:
        report = await handle.run(topic="renewable energy")
    finally:
        await handle.terminate_now()
        await worker.close()

    print(f"Report on: {report.topic}")
    print(f"Sections: {report.sections}")
    print(f"Word count: {report.word_count}")


if __name__ == "__main__":
    asyncio.run(main())
