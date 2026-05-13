"""Pydantic schemas for the feedback API."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


ChosenPosition = Literal["a", "b", "tie", "neither"]
ChosenModel = Literal["potpie", "copilot", "tie", "neither"]


class ComparisonSummary(BaseModel):
    """Lightweight list entry shown in the voter's progress view."""

    id: str
    question: str
    response_count: int


class ComparisonDetail(BaseModel):
    """Single comparison sent to the UI - model labels are NOT included.

    The voter receives an opaque `ballot_id`; the underlying randomization
    seed never crosses the wire so the client cannot reverse-engineer which
    response is Potpie vs Copilot from the seed parity.
    """

    id: str
    question: str
    ballot_id: str
    response_a: str
    response_b: str


class FeedbackVoteCreate(BaseModel):
    comparison_id: str
    chosen_position: ChosenPosition
    # Opaque ballot the server issued in `GET /comparisons/{id}`. The server
    # resolves this back to `(comparison_id, seed)` to decode chosen_model
    # without ever revealing it to the voter.
    ballot_id: str = Field(min_length=1, max_length=64)

    # Voter identity captured on the page.
    voter_name: str = Field(min_length=1, max_length=255)
    voter_email: EmailStr
    voter_role: Optional[str] = Field(default=None, max_length=64)

    # Optional quality signals.
    confidence: Optional[int] = Field(default=None, ge=1, le=5)
    reason_tags: Optional[List[str]] = None
    comment: Optional[str] = None
    time_on_page_ms: Optional[int] = Field(default=None, ge=0)

    session_id: str = Field(min_length=1, max_length=64)


class FeedbackVoteResponse(BaseModel):
    """Vote acknowledgement returned to the voter.

    Intentionally omits `chosen_model` so the voter can't correlate response
    style with model identity once they've cast a vote.
    """

    id: str
    comparison_id: str
    chosen_position: ChosenPosition
    confidence: Optional[int] = None
    submitted_at: datetime


class MyVotesResponse(BaseModel):
    """Returned to the page so it can skip already-voted comparisons."""

    votes: List[FeedbackVoteResponse]


class ModelCounts(BaseModel):
    """Tally of votes broken down by which model won."""

    potpie: int = 0
    copilot: int = 0
    tie: int = 0
    neither: int = 0
    total: int = 0


class ComparisonStats(BaseModel):
    """Per-question vote breakdown surfaced on the admin dashboard."""

    comparison_id: str
    question: str
    counts: ModelCounts
    avg_confidence: Optional[float] = None


class AdminSummary(BaseModel):
    """Aggregate snapshot at the top of the admin dashboard."""

    total_votes: int
    unique_voters: int
    by_model: ModelCounts
    by_comparison: List[ComparisonStats]


class AdminVoteRow(BaseModel):
    """Full per-vote detail shown in the admin votes table."""

    id: str
    comparison_id: str
    chosen_model: ChosenModel
    chosen_position: ChosenPosition
    presentation_seed: int
    voter_name: str
    voter_email: str
    voter_role: Optional[str] = None
    confidence: Optional[int] = None
    reason_tags: Optional[List[str]] = None
    comment: Optional[str] = None
    time_on_page_ms: Optional[int] = None
    session_id: str
    client_user_agent: Optional[str] = None
    submitted_at: datetime


class AdminVotesResponse(BaseModel):
    votes: List[AdminVoteRow]
