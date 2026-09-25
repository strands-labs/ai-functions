import textwrap

from example_helpers import models
from example_helpers.utils import display
from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult


class MeetingSummary(BaseModel):
    attendees: list[str]
    summary: str
    action_items: list[str]


# Post-conditions take the AI function's result to validate and, optionally, any
# of its original arguments (e.g., `max_length`).
def check_length(response: MeetingSummary, max_length: int):
    """Post-condition: summary must be less than 50 words."""
    length = len(response.summary.split())
    assert length <= max_length, f"Summary has {length} words, but must be {max_length} or fewer"


# A post-condition can itself be an AI Function returning `PostConditionResult`.
@ai_function(model=models.small)
def check_style(response: MeetingSummary) -> PostConditionResult:
    """
    Check if the summary below satisfies the following criteria:
    - It must use bullet points
    - It must provide the reader with the necessary context

    <summary>
    {response.summary}
    </summary>
    """


# max_attempts caps how many times the agent retries to satisfy the
# post-conditions before an exception is raised.
@ai_function(model=models.medium, post_conditions=[check_length, check_style], max_attempts=5)
def summarize_meeting(transcripts: str, max_length: int = 50) -> MeetingSummary:
    """
    Write a summary of the following meeting in less than {max_length} words.
    <transcripts>
    {transcripts}
    </transcripts>
    """


if __name__ == "__main__":
    transcripts = textwrap.dedent("""\
    Sarah: Alright, let's get started. We're three weeks out from the beta launch.
           Marcus, where are we on the authentication module?
    Marcus: We're about 80% done. The OAuth integration is working but I'm still
            debugging an issue with session timeouts. I should have it wrapped up by Friday,
            and then I'll push it to staging for QA. Oh, and Lisa—once it's there, you'll be
            able to do your screen recordings.
    Lisa: Great, that works. Priya, I'm still waiting on those product screenshots
          for the landing page. The hero image especially—I can't finalize the demo video
          without it. Any chance you could get those to me by Wednesday?
    Priya: Wednesday is tight but I can make it work. I'll also need to update the
           style guide with the new color palette, so I'll bundle that in and send everything
           over together. Actually Sarah, can you remind me—are we using the blue from the
           original mockups or the darker navy we discussed last week?
    Sarah: The darker navy. Good catch. Alright, this all sounds like it's moving.
           I'll set up a check-in for Monday morning so we can catch any blockers before they
           snowball. Marcus, just flag me if the timeout issue turns into something bigger.
           Otherwise, I think we're in decent shape.
    """)

    meeting_summary = summarize_meeting.run_sync(transcripts=transcripts)

    display("Meeting Summary", meeting_summary.model_dump_json(indent=2), lang="json")
