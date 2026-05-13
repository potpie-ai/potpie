"""Controller layer for the feedback API."""

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.feedback.feedback_schema import (
    AdminSummary,
    AdminVoteRow,
    ComparisonDetail,
    ComparisonSummary,
    FeedbackVoteCreate,
    FeedbackVoteResponse,
)
from app.modules.feedback.feedback_service import FeedbackService


class FeedbackController:
    @staticmethod
    async def list_comparisons() -> List[ComparisonSummary]:
        return await FeedbackService.list_comparisons()

    @staticmethod
    async def get_comparison_detail(comparison_id: str) -> ComparisonDetail:
        return await FeedbackService.get_comparison_detail(comparison_id)

    @staticmethod
    async def submit_vote(
        session: AsyncSession,
        payload: FeedbackVoteCreate,
        client_user_agent: Optional[str],
    ) -> FeedbackVoteResponse:
        return await FeedbackService.submit_vote(session, payload, client_user_agent)

    @staticmethod
    async def get_votes_for_email(
        session: AsyncSession, voter_email: str
    ) -> List[FeedbackVoteResponse]:
        return await FeedbackService.get_votes_for_email(session, voter_email)

    @staticmethod
    async def get_admin_summary(session: AsyncSession) -> AdminSummary:
        return await FeedbackService.get_admin_summary(session)

    @staticmethod
    async def list_all_votes(
        session: AsyncSession, limit: int = 500
    ) -> List[AdminVoteRow]:
        return await FeedbackService.list_all_votes(session, limit=limit)
