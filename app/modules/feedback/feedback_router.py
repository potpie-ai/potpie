"""FastAPI router for the blind feedback page (no auth)."""

from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db
from app.modules.feedback.feedback_controller import FeedbackController
from app.modules.feedback.feedback_schema import (
    AdminSummary,
    AdminVoteRow,
    ComparisonDetail,
    ComparisonSummary,
    FeedbackVoteCreate,
    FeedbackVoteResponse,
)
from app.modules.feedback.feedback_service import verify_admin_password

router = APIRouter()


async def _require_admin(
    x_admin_password: Optional[str] = Header(default=None, alias="X-Admin-Password"),
) -> None:
    """Dependency: enforce a constant-time check against FEEDBACK_ADMIN_PASSWORD."""
    verify_admin_password(x_admin_password)


class FeedbackAPI:
    @staticmethod
    @router.get("/feedback/comparisons", response_model=List[ComparisonSummary])
    async def list_comparisons() -> List[ComparisonSummary]:
        """List every comparison available for voting (questions only, no answers)."""
        return await FeedbackController.list_comparisons()

    @staticmethod
    @router.get(
        "/feedback/comparisons/{comparison_id}", response_model=ComparisonDetail
    )
    async def get_comparison_detail(
        comparison_id: str,
    ) -> ComparisonDetail:
        return await FeedbackController.get_comparison_detail(comparison_id)

    @staticmethod
    @router.post("/feedback/votes", response_model=FeedbackVoteResponse)
    async def submit_vote(
        payload: FeedbackVoteCreate,
        request: Request,
        async_db: AsyncSession = Depends(get_async_db),
    ) -> FeedbackVoteResponse:
        client_user_agent = request.headers.get("user-agent")
        return await FeedbackController.submit_vote(
            async_db, payload, client_user_agent
        )

    @staticmethod
    @router.get("/feedback/votes", response_model=List[FeedbackVoteResponse])
    async def get_my_votes(
        voter_email: str = Query(..., description="Voter email to look up votes for"),
        async_db: AsyncSession = Depends(get_async_db),
    ) -> List[FeedbackVoteResponse]:
        if not voter_email.strip():
            raise HTTPException(status_code=400, detail="voter_email is required")
        return await FeedbackController.get_votes_for_email(async_db, voter_email)

    # ---- Admin (password-gated) -----------------------------------------------

    @staticmethod
    @router.get(
        "/feedback/admin/summary",
        response_model=AdminSummary,
        dependencies=[Depends(_require_admin)],
    )
    async def admin_summary(
        async_db: AsyncSession = Depends(get_async_db),
    ) -> AdminSummary:
        return await FeedbackController.get_admin_summary(async_db)

    @staticmethod
    @router.get(
        "/feedback/admin/votes",
        response_model=List[AdminVoteRow],
        dependencies=[Depends(_require_admin)],
    )
    async def admin_votes(
        limit: int = Query(default=500, ge=1, le=2000),
        async_db: AsyncSession = Depends(get_async_db),
    ) -> List[AdminVoteRow]:
        return await FeedbackController.list_all_votes(async_db, limit=limit)
