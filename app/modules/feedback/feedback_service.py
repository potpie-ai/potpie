"""Business logic for blind Potpie-vs-Copilot feedback collection."""

from __future__ import annotations

import os
import secrets
import uuid
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.feedback.ballot_store import get_ballot_store
from app.modules.feedback.comparison_data import (
    get_all_comparisons,
    get_comparison,
    model_for_position,
    shuffled_responses,
)
from app.modules.feedback.feedback_model import FeedbackVote
from app.modules.feedback.feedback_schema import (
    AdminSummary,
    AdminVoteRow,
    ComparisonDetail,
    ComparisonStats,
    ComparisonSummary,
    FeedbackVoteCreate,
    FeedbackVoteResponse,
    ModelCounts,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


DEFAULT_ADMIN_PASSWORD = "potpie-eval-2026"
LOCAL_ENV_NAMES = {"local", "dev", "development", "test", "testing"}


def _is_local_environment() -> bool:
    if os.getenv("isDevelopmentMode", "").strip().lower() == "enabled":
        return True
    env_name = (
        os.getenv("ENVIRONMENT")
        or os.getenv("ENV")
        or os.getenv("LOGFIRE_ENVIRONMENT")
        or ""
    ).strip().lower()
    return env_name in LOCAL_ENV_NAMES


def _admin_password() -> str:
    """Read the admin password from env, using the default only for local/dev."""
    configured = os.getenv("FEEDBACK_ADMIN_PASSWORD")
    if configured:
        return configured
    if _is_local_environment():
        return DEFAULT_ADMIN_PASSWORD
    raise HTTPException(
        status_code=503,
        detail="FEEDBACK_ADMIN_PASSWORD is required for this environment",
    )


def verify_admin_password(provided: Optional[str]) -> None:
    """Constant-time check; raises 401 on mismatch."""
    expected = _admin_password()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid admin password")


class FeedbackService:
    @staticmethod
    async def list_comparisons() -> List[ComparisonSummary]:
        return [
            ComparisonSummary(
                id=entry["id"],
                question=entry["question"],
                response_count=len(entry["responses"]),
            )
            for entry in get_all_comparisons()
        ]

    @staticmethod
    async def get_comparison_detail(comparison_id: str) -> ComparisonDetail:
        entry = get_comparison(comparison_id)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown comparison: {comparison_id}"
            )
        # Server-side: pick a fresh seed, stash it behind an opaque ballot id.
        # The voter never sees the seed, only the ballot id, so they can't
        # reverse-engineer the response ordering.
        ballot_id, seed = get_ballot_store().issue(comparison_id)
        responses = shuffled_responses(comparison_id, seed)
        return ComparisonDetail(
            id=entry["id"],
            question=entry["question"],
            ballot_id=ballot_id,
            response_a=responses["response_a"],
            response_b=responses["response_b"],
        )

    @staticmethod
    async def submit_vote(
        session: AsyncSession,
        payload: FeedbackVoteCreate,
        client_user_agent: Optional[str],
    ) -> FeedbackVoteResponse:
        if get_comparison(payload.comparison_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown comparison: {payload.comparison_id}",
            )

        # Resolve the opaque ballot back to its presentation seed. If the
        # ballot is malformed, forged, expired, or issued for another question,
        # we refuse the vote so a voter can't choose their own ordering.
        presentation_seed = get_ballot_store().redeem(
            payload.comparison_id, payload.ballot_id
        )
        if presentation_seed is None:
            raise HTTPException(
                status_code=400,
                detail="Ballot has expired. Reload the question and vote again.",
            )

        try:
            chosen_model = model_for_position(
                payload.comparison_id,
                payload.chosen_position,
                presentation_seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        vote_id = uuid.uuid4().hex
        normalized_email = payload.voter_email.lower().strip()

        stmt = pg_insert(FeedbackVote).values(
            id=vote_id,
            comparison_id=payload.comparison_id,
            chosen_model=chosen_model,
            chosen_position=payload.chosen_position,
            presentation_seed=presentation_seed,
            voter_name=payload.voter_name.strip(),
            voter_email=normalized_email,
            voter_role=payload.voter_role,
            confidence=payload.confidence,
            reason_tags=payload.reason_tags,
            comment=payload.comment,
            time_on_page_ms=payload.time_on_page_ms,
            session_id=payload.session_id,
            client_user_agent=client_user_agent,
        )
        # Upsert: a voter may revise their vote on the same comparison.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_feedback_votes_email_comparison",
            set_={
                "chosen_model": stmt.excluded.chosen_model,
                "chosen_position": stmt.excluded.chosen_position,
                "presentation_seed": stmt.excluded.presentation_seed,
                "voter_name": stmt.excluded.voter_name,
                "voter_role": stmt.excluded.voter_role,
                "confidence": stmt.excluded.confidence,
                "reason_tags": stmt.excluded.reason_tags,
                "comment": stmt.excluded.comment,
                "time_on_page_ms": stmt.excluded.time_on_page_ms,
                "session_id": stmt.excluded.session_id,
                "client_user_agent": stmt.excluded.client_user_agent,
                "submitted_at": stmt.excluded.submitted_at,
            },
        ).returning(FeedbackVote)

        try:
            result = await session.execute(stmt)
            row = result.scalar_one()
            await session.commit()
        except SQLAlchemyError as exc:
            await session.rollback()
            logger.exception("Failed to persist feedback vote")
            raise HTTPException(
                status_code=500, detail="Failed to record vote"
            ) from exc

        # Deliberately do NOT include `chosen_model` in the response - the
        # voter must stay blind to which model they just chose.
        return FeedbackVoteResponse(
            id=row.id,
            comparison_id=row.comparison_id,
            chosen_position=row.chosen_position,
            confidence=row.confidence,
            submitted_at=row.submitted_at,
        )

    @staticmethod
    async def get_admin_summary(session: AsyncSession) -> AdminSummary:
        """Aggregate snapshot for the admin dashboard."""
        # Per-(comparison, model) counts in one query.
        bucket_stmt = select(
            FeedbackVote.comparison_id,
            FeedbackVote.chosen_model,
            func.count(FeedbackVote.id).label("votes"),
            func.avg(FeedbackVote.confidence).label("avg_conf"),
        ).group_by(FeedbackVote.comparison_id, FeedbackVote.chosen_model)
        bucket_rows = (await session.execute(bucket_stmt)).all()

        total_votes = sum(int(row.votes) for row in bucket_rows)

        unique_voters_stmt = select(
            func.count(func.distinct(FeedbackVote.voter_email))
        )
        unique_voters = int(
            (await session.execute(unique_voters_stmt)).scalar() or 0
        )

        # Build per-comparison structure for every configured comparison, even
        # those with zero votes, so the dashboard shows a stable row order.
        per_comparison: Dict[str, ComparisonStats] = {}
        confidence_acc: Dict[str, list] = {}
        for entry in get_all_comparisons():
            per_comparison[entry["id"]] = ComparisonStats(
                comparison_id=entry["id"],
                question=entry["question"],
                counts=ModelCounts(),
            )
            confidence_acc[entry["id"]] = []

        # Overall and per-question tallies.
        overall = ModelCounts()
        for row in bucket_rows:
            cid = row.comparison_id
            model = row.chosen_model
            votes = int(row.votes)
            avg_conf = float(row.avg_conf) if row.avg_conf is not None else None

            if cid not in per_comparison:
                # Vote for a comparison that no longer exists in code (id
                # changed). Surface it anyway so we don't silently drop data.
                per_comparison[cid] = ComparisonStats(
                    comparison_id=cid,
                    question=f"(unknown comparison: {cid})",
                    counts=ModelCounts(),
                )
                confidence_acc[cid] = []

            counts = per_comparison[cid].counts
            if model in ("potpie", "copilot", "tie", "neither"):
                setattr(counts, model, getattr(counts, model) + votes)
                setattr(overall, model, getattr(overall, model) + votes)
            counts.total += votes
            overall.total += votes
            if avg_conf is not None:
                confidence_acc[cid].append((avg_conf, votes))

        # Weighted average confidence per comparison.
        for cid, samples in confidence_acc.items():
            total_weight = sum(weight for _, weight in samples)
            if total_weight > 0:
                weighted = sum(value * weight for value, weight in samples)
                per_comparison[cid].avg_confidence = round(
                    weighted / total_weight, 2
                )

        return AdminSummary(
            total_votes=total_votes,
            unique_voters=unique_voters,
            by_model=overall,
            by_comparison=list(per_comparison.values()),
        )

    @staticmethod
    async def list_all_votes(
        session: AsyncSession,
        limit: int = 500,
    ) -> List[AdminVoteRow]:
        stmt = (
            select(FeedbackVote)
            .order_by(FeedbackVote.submitted_at.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).scalars().all()
        return [
            AdminVoteRow(
                id=row.id,
                comparison_id=row.comparison_id,
                chosen_model=row.chosen_model,
                chosen_position=row.chosen_position,
                presentation_seed=row.presentation_seed,
                voter_name=row.voter_name,
                voter_email=row.voter_email,
                voter_role=row.voter_role,
                confidence=row.confidence,
                reason_tags=row.reason_tags,
                comment=row.comment,
                time_on_page_ms=row.time_on_page_ms,
                session_id=row.session_id,
                client_user_agent=row.client_user_agent,
                submitted_at=row.submitted_at,
            )
            for row in rows
        ]

    @staticmethod
    async def get_votes_for_email(
        session: AsyncSession, voter_email: str
    ) -> List[FeedbackVoteResponse]:
        normalized_email = voter_email.lower().strip()
        stmt = (
            select(FeedbackVote)
            .where(FeedbackVote.voter_email == normalized_email)
            .order_by(FeedbackVote.submitted_at.desc())
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        # `chosen_model` is intentionally not surfaced here - if a voter
        # could see which model their earlier "A" choice mapped to they
        # would unmask the entire study.
        return [
            FeedbackVoteResponse(
                id=row.id,
                comparison_id=row.comparison_id,
                chosen_position=row.chosen_position,
                confidence=row.confidence,
                submitted_at=row.submitted_at,
            )
            for row in rows
        ]
