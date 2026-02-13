"""Service for querying Logfire analytics data."""

import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from logfire.query_client import LogfireQueryClient

from app.modules.analytics.schemas import (
    AnalyticsPeriod,
    AnalyticsSummary,
    ConversationStat,
    DailyCost,
    RawSpan,
    UserAnalyticsResponse,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class AnalyticsService:
    """Service for querying and aggregating Logfire data."""

    def __init__(self):
        """Initialize with Logfire read token from environment."""
        self.read_token = os.getenv("LOGFIRE_READ_TOKEN")
        if not self.read_token:
            logger.warning("LOGFIRE_READ_TOKEN not found in environment")

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        """Escape single quotes for safe SQL string literal."""
        return value.replace("'", "''")

    @staticmethod
    def _extract_rows(result) -> list:
        """Extract row list from query_json_rows() response.

        The Logfire client returns {"columns": [...], "rows": [...]}
        rather than a plain list. This helper handles both formats defensively.
        """
        if isinstance(result, dict) and "rows" in result:
            return result["rows"]
        if isinstance(result, list):
            return result
        return []

    def get_raw_logfire_response(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict:
        """
        Run a minimal query and return the raw Logfire API response (no transformation).
        Used for debugging to see exact keys and structure.
        """
        if not self.read_token:
            raise ValueError("LOGFIRE_READ_TOKEN not configured")

        sd, ed, start_dt, _ = self._resolve_date_range(start_date, end_date)
        user_id_escaped = self._escape_sql_string(user_id)
        query = f"""
        SELECT
            start_timestamp,
            attributes->>'user_id' as user_id,
            attributes->>'project_id' as project_id,
            attributes->>'gen_ai.usage.input_tokens' as input_tokens,
            attributes->>'gen_ai.usage.output_tokens' as output_tokens,
            attributes->>'openinference.span.kind' as span_kind
        FROM records
        WHERE attributes->>'user_id' = '{user_id_escaped}'
          AND start_timestamp >= '{sd.isoformat()}'
          AND start_timestamp <= '{ed.isoformat()}T23:59:59Z'
        LIMIT 20
        """

        with LogfireQueryClient(read_token=self.read_token) as client:
            try:
                column_oriented = client.query_json(
                    sql=query,
                    min_timestamp=start_dt,
                    limit=20,
                )
            except Exception as e:
                column_oriented = {"_error": str(e)}
                logger.exception("Debug query_json failed: %s", e)

            try:
                row_oriented = client.query_json_rows(
                    sql=query,
                    min_timestamp=start_dt,
                    limit=20,
                )
            except Exception as e:
                row_oriented = {"_error": str(e)}
                logger.exception("Debug query_json_rows failed: %s", e)

        return {
            "column_oriented": column_oriented,
            "row_oriented": row_oriented,
        }

    # ------------------------------------------------------------------
    # Date-range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_date_range(
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> tuple:
        """Return (start_date, end_date, start_datetime_utc, end_datetime_utc).

        Defaults:
            end_date   → today
            start_date → 30 days before end_date
        The end datetime is set to 23:59:59.999999 of end_date (inclusive).
        """
        today = date.today()
        if end_date is None:
            end_date = today
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        # Clamp: start must not be after end
        if start_date > end_date:
            start_date = end_date
        start_dt = datetime(
            start_date.year, start_date.month, start_date.day,
            tzinfo=timezone.utc,
        )
        end_dt = datetime(
            end_date.year, end_date.month, end_date.day,
            23, 59, 59, 999999,
            tzinfo=timezone.utc,
        )
        return start_date, end_date, start_dt, end_dt

    # ------------------------------------------------------------------
    # Tokens by day (primary endpoint)
    # ------------------------------------------------------------------

    def get_tokens_by_day(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict]:
        """
        Get total tokens per day per project for a user.

        Args:
            user_id: The user ID to query
            start_date: Start of the range (inclusive, default 30 days ago)
            end_date: End of the range (inclusive, default today)

        Returns:
            List of dicts with keys: date, project_id, total_tokens
        """
        if not self.read_token:
            raise ValueError("LOGFIRE_READ_TOKEN not configured")

        sd, ed, start_dt, end_dt = self._resolve_date_range(start_date, end_date)
        user_id_escaped = self._escape_sql_string(user_id)

        query = f"""
        SELECT
            date_trunc('day', start_timestamp)          AS day,
            attributes->>'project_id'                    AS project_id,
            SUM(
                COALESCE((attributes->>'gen_ai.usage.input_tokens')::numeric, 0)
                + COALESCE((attributes->>'gen_ai.usage.output_tokens')::numeric, 0)
            ) AS total_tokens
        FROM records
        WHERE attributes->>'user_id' = '{user_id_escaped}'
          AND start_timestamp >= '{sd.isoformat()}'
          AND start_timestamp <= '{ed.isoformat()}T23:59:59Z'
          AND attributes->>'openinference.span.kind' = 'LLM'
        GROUP BY 1, 2
        ORDER BY 1 DESC
        LIMIT 1000
        """

        with LogfireQueryClient(read_token=self.read_token) as client:
            try:
                raw = client.query_json_rows(
                    sql=query,
                    min_timestamp=start_dt,
                    limit=1000,
                )
            except Exception as e:
                logger.error("Error querying tokens by day: %s", e)
                return []

        rows = self._extract_rows(raw)
        result = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            day_val = row.get("day")
            if day_val is not None and hasattr(day_val, "isoformat"):
                day_str = day_val.isoformat()[:10]
            else:
                day_str = str(day_val)[:10] if day_val else ""
            result.append({
                "date": day_str,
                "project_id": row.get("project_id"),
                "total_tokens": int(float(row.get("total_tokens") or 0)),
            })
        return result

    def get_user_analytics(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> UserAnalyticsResponse:
        """
        Get analytics data for a user over the specified date range.

        Args:
            user_id: The user ID to query
            start_date: Start of the range (inclusive, default 30 days ago)
            end_date: End of the range (inclusive, default today)

        Returns:
            UserAnalyticsResponse with aggregated analytics data
        """
        if not self.read_token:
            raise ValueError("LOGFIRE_READ_TOKEN not configured")

        sd, ed, start_dt, end_dt = self._resolve_date_range(start_date, end_date)
        days = (ed - sd).days + 1  # inclusive

        logger.info(
            f"Fetching analytics for user {user_id} from {sd} to {ed} ({days} days)"
        )

        # Query Logfire for different data types
        with LogfireQueryClient(read_token=self.read_token) as client:
            cost_data = self._get_cost_data(client, user_id, start_dt, sd, ed)
            agent_data = self._get_agent_data(client, user_id, start_dt, sd, ed)
            conversation_data = self._get_conversation_data(
                client, user_id, start_dt, sd, ed
            )

        return self._aggregate_analytics(
            user_id=user_id,
            start_date=sd,
            end_date=ed,
            days=days,
            cost_data=cost_data,
            agent_data=agent_data,
            conversation_data=conversation_data,
        )

    def _get_cost_data(
        self,
        client: LogfireQueryClient,
        user_id: str,
        start_dt: datetime,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """Query LLM usage data from LLM spans within a date range."""
        user_id_escaped = self._escape_sql_string(user_id)
        query = f"""
        SELECT 
            start_timestamp,
            attributes->>'gen_ai.usage.input_tokens' as input_tokens,
            attributes->>'gen_ai.usage.output_tokens' as output_tokens,
            attributes->>'gen_ai.response.model' as model,
            attributes->>'outcome' as outcome,
            span_name
        FROM records
        WHERE 
            attributes->>'openinference.span.kind' = 'LLM'
            AND attributes->>'user_id' = '{user_id_escaped}'
            AND start_timestamp >= '{start_date.isoformat()}'
            AND start_timestamp <= '{end_date.isoformat()}T23:59:59Z'
        ORDER BY start_timestamp DESC
        LIMIT 10000
        """

        try:
            raw = client.query_json_rows(
                sql=query,
                min_timestamp=start_dt,
                limit=10000,
            )
            rows = self._extract_rows(raw)
            if not rows:
                return []
            result = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ts = row.get("start_timestamp") or row.get("Start_Timestamp")
                if ts is None:
                    continue
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                else:
                    ts = str(ts)
                result.append({
                    "start_timestamp": ts,
                    "input_tokens": row.get("input_tokens") or row.get("Input_Tokens"),
                    "output_tokens": row.get("output_tokens") or row.get("Output_Tokens"),
                    "model": row.get("model") or row.get("Model"),
                    "outcome": row.get("outcome") or row.get("Outcome"),
                    "span_name": row.get("span_name") or row.get("Span_Name"),
                })
            logger.info("Retrieved %s LLM usage records", len(result))
            return result
        except Exception as e:
            logger.error("Error querying cost data: %s", e)
            return []

    def _get_agent_data(
        self,
        client: LogfireQueryClient,
        user_id: str,
        start_dt: datetime,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """Query agent/LLM execution data within a date range."""
        user_id_escaped = self._escape_sql_string(user_id)
        query = f"""
        SELECT 
            start_timestamp,
            attributes->>'agent_id' as agent_id,
            attributes->>'run_id' as run_id,
            attributes->>'conversation_id' as conversation_id,
            span_name,
            duration_ms
        FROM records
        WHERE 
            attributes->>'user_id' = '{user_id_escaped}'
            AND start_timestamp >= '{start_date.isoformat()}'
            AND start_timestamp <= '{end_date.isoformat()}T23:59:59Z'
        ORDER BY start_timestamp DESC
        LIMIT 10000
        """

        try:
            raw = client.query_json_rows(
                sql=query,
                min_timestamp=start_dt,
                limit=10000,
            )
            rows = self._extract_rows(raw)
            if not rows:
                return []
            result = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                result.append({
                    "start_timestamp": row.get("start_timestamp") or row.get("Start_Timestamp"),
                    "agent_id": row.get("agent_id") or row.get("Agent_Id"),
                    "run_id": row.get("run_id") or row.get("Run_Id"),
                    "conversation_id": row.get("conversation_id") or row.get("Conversation_Id"),
                    "span_name": row.get("span_name") or row.get("Span_Name"),
                    "duration_ms": row.get("duration_ms") or row.get("Duration_Ms"),
                })
            logger.info("Retrieved %s execution records", len(result))
            return result
        except Exception as e:
            logger.error("Error querying agent data: %s", e)
            return []

    def _get_conversation_data(
        self,
        client: LogfireQueryClient,
        user_id: str,
        start_dt: datetime,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """Query conversation statistics within a date range."""
        user_id_escaped = self._escape_sql_string(user_id)
        query = f"""
        SELECT 
            start_timestamp,
            attributes->>'conversation_id' as conversation_id,
            span_name
        FROM records
        WHERE 
            attributes->>'user_id' = '{user_id_escaped}'
            AND attributes->>'conversation_id' IS NOT NULL
            AND start_timestamp >= '{start_date.isoformat()}'
            AND start_timestamp <= '{end_date.isoformat()}T23:59:59Z'
        ORDER BY start_timestamp DESC
        LIMIT 10000
        """

        try:
            raw = client.query_json_rows(
                sql=query,
                min_timestamp=start_dt,
                limit=10000,
            )
            rows = self._extract_rows(raw)
            if not rows:
                return []
            result = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                result.append({
                    "start_timestamp": row.get("start_timestamp") or row.get("Start_Timestamp"),
                    "conversation_id": row.get("conversation_id") or row.get("Conversation_Id"),
                    "span_name": row.get("span_name") or row.get("Span_Name"),
                })
            logger.info("Retrieved %s conversation records", len(result))
            return result
        except Exception as e:
            logger.error("Error querying conversation data: %s", e)
            return []

    def _aggregate_analytics(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
        days: int,
        cost_data: List[Dict],
        agent_data: List[Dict],
        conversation_data: List[Dict],
    ) -> UserAnalyticsResponse:
        """Aggregate raw data into analytics response."""

        # Aggregate daily costs from token usage
        daily_costs_map = defaultdict(lambda: {"cost": 0.0, "run_count": 0, "tokens": 0})
        total_cost = 0.0
        total_tokens = 0
        outcomes_count = defaultdict(int)

        for record in cost_data:
            try:
                # Parse the timestamp
                timestamp = datetime.fromisoformat(
                    record['start_timestamp'].replace('Z', '+00:00')
                )
                date_key = timestamp.strftime("%Y-%m-%d")

                # Parse tokens and estimate cost
                # Using rough pricing: $0.50 per 1M input tokens, $1.50 per 1M output tokens
                # (Adjust based on your actual model pricing)
                input_tokens = float(record.get('input_tokens') or 0)
                output_tokens = float(record.get('output_tokens') or 0)
                
                # Calculate cost (price per million tokens)
                input_cost = (input_tokens / 1_000_000) * 0.50
                output_cost = (output_tokens / 1_000_000) * 1.50
                record_cost = input_cost + output_cost
                
                total_cost += record_cost
                total_tokens += input_tokens + output_tokens
                daily_costs_map[date_key]["cost"] += record_cost
                daily_costs_map[date_key]["tokens"] += input_tokens + output_tokens
                daily_costs_map[date_key]["run_count"] += 1

                # Track outcomes – value may be None even when key exists
                outcome = record.get('outcome') or 'success'
                outcomes_count[str(outcome)] += 1
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid cost record: {e}")
                continue

        # Convert daily costs to list
        daily_costs = [
            DailyCost(
                date=date,
                cost=round(data["cost"], 4),
                run_count=data["run_count"],
                tokens=int(data["tokens"]),
            )
            for date, data in sorted(daily_costs_map.items())
        ]

        # Calculate LLM call stats
        total_llm_calls = sum(outcomes_count.values())
        if total_llm_calls == 0:
            total_llm_calls = len(cost_data)  # Count all LLM records
        success_count = outcomes_count.get('success', total_llm_calls)  # Default all to success
        success_rate = success_count / total_llm_calls if total_llm_calls > 0 else 1.0

        # Calculate average duration
        durations = [
            float(record.get('duration_ms', 0) or 0)
            for record in agent_data
            if record.get('duration_ms')
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        # Aggregate conversation stats
        conv_by_date = defaultdict(set)
        for record in conversation_data:
            try:
                timestamp = datetime.fromisoformat(
                    record['start_timestamp'].replace('Z', '+00:00')
                )
                date_key = timestamp.strftime("%Y-%m-%d")
                conv_id = record.get('conversation_id')
                if conv_id:
                    conv_by_date[date_key].add(conv_id)
            except (ValueError, TypeError):
                continue

        conversation_stats = [
            ConversationStat(
                date=date,
                count=len(conv_ids),
                avg_messages=len(conv_ids),  # Simplified - could calculate actual message count
            )
            for date, conv_ids in sorted(conv_by_date.items())
        ]

        # Build response
        return UserAnalyticsResponse(
            user_id=user_id,
            period=AnalyticsPeriod(
                start=start_date,
                end=end_date,
                days=days,
            ),
            summary=AnalyticsSummary(
                total_cost=round(total_cost, 4),
                total_llm_calls=total_llm_calls,
                total_tokens=int(total_tokens),
                avg_duration_ms=round(avg_duration, 2),
                success_rate=round(success_rate, 4),
            ),
            daily_costs=daily_costs,
            agent_runs_by_outcome=dict(outcomes_count) if outcomes_count else {"success": total_llm_calls},
            conversation_stats=conversation_stats,
        )

    def get_raw_spans(
        self,
        user_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
    ) -> List[RawSpan]:
        """
        Get raw span data for a user (useful for debugging or custom analysis).

        Args:
            user_id: The user ID to query
            start_date: Start of the range (inclusive, default 30 days ago)
            end_date: End of the range (inclusive, default today)
            limit: Maximum number of spans to return

        Returns:
            List of raw spans
        """
        if not self.read_token:
            raise ValueError("LOGFIRE_READ_TOKEN not configured")

        sd, ed, start_dt, _ = self._resolve_date_range(start_date, end_date)
        user_id_escaped = self._escape_sql_string(user_id)
        query = f"""
        SELECT 
            start_timestamp,
            span_name,
            duration_ms,
            attributes
        FROM records
        WHERE 
            attributes->>'user_id' = '{user_id_escaped}'
            AND start_timestamp >= '{sd.isoformat()}'
            AND start_timestamp <= '{ed.isoformat()}T23:59:59Z'
        ORDER BY start_timestamp DESC
        LIMIT {min(limit, 10000)}
        """

        with LogfireQueryClient(read_token=self.read_token) as client:
            try:
                raw = client.query_json_rows(
                    sql=query,
                    min_timestamp=start_dt,
                    limit=limit,
                )
                rows = self._extract_rows(raw)
                return [RawSpan(**row) for row in rows]
            except Exception as e:
                logger.error(f"Error querying raw spans: {e}")
                return []
