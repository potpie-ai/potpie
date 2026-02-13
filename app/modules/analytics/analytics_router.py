"""FastAPI router for analytics endpoints."""

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.modules.analytics.analytics_service import AnalyticsService
from app.modules.analytics.schemas import (
    RawSpan,
    TokensByDay,
    UserAnalyticsResponse,
)
from app.modules.auth.auth_service import AuthService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared date-range query parameters
# ---------------------------------------------------------------------------
# Best practice: accept ISO dates (YYYY-MM-DD) with sensible defaults.
# FastAPI automatically parses `date` query params from strings and returns
# a 422 with a clear message for invalid formats (e.g. "2026-02-31").
# ---------------------------------------------------------------------------

_START_DATE_QUERY = Query(
    default=None,
    description="Start date (inclusive, YYYY-MM-DD). Defaults to 30 days before end_date.",
    examples=["2026-01-01"],
)
_END_DATE_QUERY = Query(
    default=None,
    description="End date (inclusive, YYYY-MM-DD). Defaults to today.",
    examples=["2026-02-12"],
)


def _validate_date_range(
    start_date: Optional[date], end_date: Optional[date]
) -> None:
    """Raise 422 if the caller-supplied dates are invalid."""
    today = date.today()

    # Reject future dates
    if end_date and end_date > today:
        raise HTTPException(
            status_code=422,
            detail="end_date cannot be in the future.",
        )
    if start_date and start_date > today:
        raise HTTPException(
            status_code=422,
            detail="start_date cannot be in the future.",
        )

    # start must be <= end
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must be on or before end_date.",
        )

    # Max 30-day range
    if start_date and end_date and (end_date - start_date).days > 30:
        raise HTTPException(
            status_code=422,
            detail="Date range must not exceed 30 days.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class AnalyticsAPI:
    """Analytics API endpoints."""

    # ---- Tokens by day (primary endpoint) ----

    @staticmethod
    @router.get(
        "/analytics/tokens-by-day",
        response_model=List[TokensByDay],
        summary="Get token usage by day",
        description="Returns total tokens per day, grouped by project, for the "
        "authenticated user within a custom date range.",
    )
    async def get_tokens_by_day(
        start_date: Optional[date] = _START_DATE_QUERY,
        end_date: Optional[date] = _END_DATE_QUERY,
        user=Depends(AuthService.check_auth),
    ):
        _validate_date_range(start_date, end_date)
        user_id = user.get("user_id")
        try:
            service = AnalyticsService()
            rows = service.get_tokens_by_day(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            return [TokensByDay(**r) for r in rows]
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"ValueError in tokens-by-day: {error_msg}")
            if "LOGFIRE_READ_TOKEN" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Analytics service not properly configured.",
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process tokens by day: {error_msg}",
            )
        except Exception as e:
            logger.exception(f"Error fetching tokens by day: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve tokens by day: {str(e)}",
            )

    # ---- Aggregated analytics ----

    @staticmethod
    @router.get(
        "/analytics/summary",
        response_model=UserAnalyticsResponse,
        summary="Get user analytics summary",
        description="Retrieve aggregated analytics data for the authenticated user "
        "from Logfire, including LLM costs, agent runs, and conversation "
        "statistics within a custom date range.",
    )
    async def get_user_analytics(
        start_date: Optional[date] = _START_DATE_QUERY,
        end_date: Optional[date] = _END_DATE_QUERY,
        user=Depends(AuthService.check_auth),
    ):
        _validate_date_range(start_date, end_date)
        user_id = user.get("user_id")
        logger.info(
            f"User {user_id} requesting analytics ({start_date} to {end_date})"
        )

        try:
            service = AnalyticsService()
            analytics_data = await service.get_user_analytics(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            return analytics_data
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"ValueError in analytics: {error_msg}")
            if "LOGFIRE_READ_TOKEN" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Analytics service not properly configured. "
                    "Please contact support.",
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process analytics data: {error_msg}",
            )
        except Exception as e:
            logger.exception(f"Error fetching analytics: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve analytics data: {str(e)}",
            )

    # ---- Raw span data ----

    @staticmethod
    @router.get(
        "/analytics/raw",
        response_model=List[RawSpan],
        summary="Get raw span data",
        description="Retrieve raw Logfire span data for the authenticated user "
        "within a custom date range. Useful for debugging or custom analysis.",
    )
    async def get_raw_spans(
        start_date: Optional[date] = _START_DATE_QUERY,
        end_date: Optional[date] = _END_DATE_QUERY,
        limit: int = Query(
            default=100,
            ge=1,
            le=1000,
            description="Maximum number of spans to return (1-1000)",
        ),
        user=Depends(AuthService.check_auth),
    ):
        _validate_date_range(start_date, end_date)
        user_id = user.get("user_id")
        logger.info(
            f"User {user_id} requesting raw spans "
            f"({start_date} to {end_date}, limit {limit})"
        )

        try:
            service = AnalyticsService()
            spans = await service.get_raw_spans(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
            return spans
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"ValueError in raw spans: {error_msg}")
            if "LOGFIRE_READ_TOKEN" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Analytics service not properly configured. "
                    "Please contact support.",
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process raw span data: {error_msg}",
            )
        except Exception as e:
            logger.exception(f"Error fetching raw spans: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve raw span data: {str(e)}",
            )

    # ---- Debug: raw Logfire response ----

    @staticmethod
    @router.get(
        "/analytics/debug/raw",
        summary="Get raw Logfire response (debug)",
        description="Returns the raw Logfire API response (column- and row-oriented) "
        "for a minimal query. Use to inspect exact keys and structure.",
    )
    async def get_debug_raw(
        start_date: Optional[date] = _START_DATE_QUERY,
        end_date: Optional[date] = _END_DATE_QUERY,
        user=Depends(AuthService.check_auth),
    ):
        _validate_date_range(start_date, end_date)
        user_id = user.get("user_id")
        try:
            service = AnalyticsService()
            raw = service.get_raw_logfire_response(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
            )
            return raw
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Configuration error: {error_msg}")
            if "LOGFIRE_READ_TOKEN" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="Analytics service not properly configured.",
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process debug data: {error_msg}",
            )
        except Exception as e:
            logger.exception(f"Error fetching raw Logfire response: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve raw response: {str(e)}",
            )
