"""Pydantic schemas for analytics API."""

from datetime import date, datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tokens-by-day (primary endpoint)
# ---------------------------------------------------------------------------

class TokensByDay(BaseModel):
    """Token usage for a single day, grouped by project."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    project_id: Optional[str] = Field(
        None, description="Project ID (null for spans without a project)"
    )
    total_tokens: int = Field(..., description="Sum of input + output tokens")


# ---------------------------------------------------------------------------
# Main analytics response
# ---------------------------------------------------------------------------

class DailyCost(BaseModel):
    """Daily cost data point."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    cost: float = Field(..., description="Total cost for the day")
    run_count: int = Field(..., description="Number of LLM calls")
    tokens: int = Field(default=0, description="Total tokens used")


class ConversationStat(BaseModel):
    """Daily conversation statistics."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    count: int = Field(..., description="Number of conversations")
    avg_messages: float = Field(
        ..., description="Average messages per conversation"
    )


class AnalyticsSummary(BaseModel):
    """Summary statistics for the period."""

    total_cost: float = Field(..., description="Total estimated LLM cost")
    total_llm_calls: int = Field(..., description="Total number of LLM calls")
    total_tokens: int = Field(..., description="Total tokens used (input + output)")
    avg_duration_ms: float = Field(..., description="Average duration in milliseconds")
    success_rate: float = Field(..., description="Success rate (0-1)")


class AnalyticsPeriod(BaseModel):
    """Time period for analytics data."""

    start: date = Field(..., description="Start date (inclusive)")
    end: date = Field(..., description="End date (inclusive)")
    days: int = Field(..., description="Number of days in the range")


class UserAnalyticsResponse(BaseModel):
    """Response model for user analytics."""

    user_id: str = Field(..., description="User ID")
    period: AnalyticsPeriod = Field(..., description="Time period")
    summary: AnalyticsSummary = Field(..., description="Summary statistics")
    daily_costs: List[DailyCost] = Field(
        ..., description="Daily cost breakdown"
    )
    agent_runs_by_outcome: Dict[str, int] = Field(
        ..., description="Agent runs grouped by outcome"
    )
    conversation_stats: List[ConversationStat] = Field(
        ..., description="Daily conversation statistics"
    )


# ---------------------------------------------------------------------------
# Raw span data
# ---------------------------------------------------------------------------

class RawSpan(BaseModel):
    """Raw Logfire span data."""

    start_timestamp: datetime
    span_name: Optional[str] = None
    duration_ms: Optional[float] = None
    attributes: Optional[Dict] = None
