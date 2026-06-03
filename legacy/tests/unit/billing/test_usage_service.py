"""
Unit tests for UsageReportingService (app/modules/billing/usage_service.py)

Tests cover:
- report_message_usage (async)
- report_message_usage_sync
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.modules.billing.usage_service import UsageReportingService


pytestmark = pytest.mark.unit


class TestReportMessageUsage:
    """Tests for report_message_usage async method"""

    @pytest.fixture
    def user_args(self):
        """Standard arguments for usage reporting"""
        return {
            "user_id": "test-user-123",
            "dodo_customer_id": "dodo-cust-456",
            "conversation_id": "conv-789",
        }

    @pytest.fixture
    def expected_payload(self, user_args):
        """Expected JSON payload sent to the API"""
        return {
            "user_id": user_args["user_id"],
            "dodo_customer_id": user_args["dodo_customer_id"],
            "event_type": "message",
            "resource_id": user_args["conversation_id"],
            "resource_type": "conversation",
        }

    @pytest.mark.asyncio
    async def test_report_message_usage_success(self, user_args, expected_payload):
        """Test successful usage report returns 200 response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "usage_id": "usage-123"}
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "success", "usage_id": "usage-123"}
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args.kwargs
            assert call_kwargs["json"] == expected_payload
            assert call_kwargs["timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_report_message_usage_http_400(self, user_args):
        """Test HTTP 400 error returns error status"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "error", "error": "HTTP 400"}

    @pytest.mark.asyncio
    async def test_report_message_usage_http_401(self, user_args):
        """Test HTTP 401 error returns error status"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "error", "error": "HTTP 401"}

    @pytest.mark.asyncio
    async def test_report_message_usage_http_500(self, user_args):
        """Test HTTP 500 error returns error status"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "error", "error": "HTTP 500"}

    @pytest.mark.asyncio
    async def test_report_message_usage_connect_error(self, user_args):
        """Test ConnectError returns service unavailable error"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {
                "status": "error",
                "error": "stripe-potpie service unavailable",
            }

    @pytest.mark.asyncio
    async def test_report_message_usage_generic_exception(self, user_args):
        """Test generic Exception returns error with message"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = ValueError("Something went wrong")

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "error", "error": "Something went wrong"}

    @pytest.mark.asyncio
    async def test_report_message_usage_timeout(self, user_args):
        """Test Timeout exception is caught and handled"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            # Timeout is a subclass of Exception, so it should be caught
            assert result == {"status": "error", "error": "Request timed out"}

    @pytest.mark.asyncio
    async def test_report_message_usage_payload_verification(
        self, user_args, expected_payload
    ):
        """Test correct JSON payload is sent to the API"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            await UsageReportingService.report_message_usage(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args

            # call_args[0] contains positional args (url is first)
            # call_args.kwargs contains keyword args (json, timeout)
            call_kwargs = call_args.kwargs

            # Verify payload structure
            assert call_kwargs["json"]["user_id"] == expected_payload["user_id"]
            assert (
                call_kwargs["json"]["dodo_customer_id"]
                == expected_payload["dodo_customer_id"]
            )
            assert call_kwargs["json"]["event_type"] == expected_payload["event_type"]
            assert call_kwargs["json"]["resource_id"] == expected_payload["resource_id"]
            assert (
                call_kwargs["json"]["resource_type"]
                == expected_payload["resource_type"]
            )

            # Verify URL (positional arg) and timeout
            assert "/dodo/usage/report" in call_args[0][0]
            assert call_kwargs["timeout"] == 5.0


class TestReportMessageUsageSync:
    """Tests for report_message_usage_sync method"""

    @pytest.fixture
    def user_args(self):
        """Standard arguments for usage reporting"""
        return {
            "user_id": "test-user-123",
            "dodo_customer_id": "dodo-cust-456",
            "conversation_id": "conv-789",
        }

    def test_report_message_usage_sync_loop_running(self, user_args):
        """Test returns skipped when called from within a running event loop"""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            result = UsageReportingService.report_message_usage_sync(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {
                "status": "skipped",
                "reason": "called from async context",
            }
            mock_loop.run_until_complete.assert_not_called()

    def test_report_message_usage_sync_new_loop_created(self, user_args):
        """Test creates new loop when get_event_loop raises RuntimeError"""
        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = {"status": "success"}

        def get_event_loop_raises():
            raise RuntimeError("no running event loop")

        with (
            patch(
                "asyncio.get_event_loop", side_effect=get_event_loop_raises
            ) as mock_get_loop,
            patch(
                "asyncio.new_event_loop", return_value=mock_new_loop
            ) as mock_create_loop,
            patch("asyncio.set_event_loop") as mock_set_loop,
        ):
            result = UsageReportingService.report_message_usage_sync(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "success"}
            mock_create_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_new_loop)
            mock_new_loop.run_until_complete.assert_called_once()
            mock_new_loop.close.assert_called_once()

    def test_report_message_usage_sync_delegates_to_async(self, user_args):
        """Test successful delegation to async method when loop exists"""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = {"status": "success"}

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch(
                "app.modules.billing.usage_service.UsageReportingService.report_message_usage",
                new_callable=AsyncMock,
                return_value={"status": "success"},
            ) as mock_async,
        ):
            result = UsageReportingService.report_message_usage_sync(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "success"}
            mock_async.assert_called_once_with(
                user_args["user_id"],
                user_args["dodo_customer_id"],
                user_args["conversation_id"],
            )
            mock_loop.run_until_complete.assert_called_once()

    def test_report_message_usage_sync_runtime_error_no_loop(self, user_args):
        """Test RuntimeError is caught when no loop exists (Python 3.10+ behavior)"""
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.side_effect = RuntimeError("no running event loop")

        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = {"status": "success"}

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop) as mock_get_loop,
            patch(
                "asyncio.new_event_loop", return_value=mock_new_loop
            ) as mock_create_loop,
            patch("asyncio.set_event_loop") as mock_set_loop,
        ):
            result = UsageReportingService.report_message_usage_sync(
                user_id=user_args["user_id"],
                dodo_customer_id=user_args["dodo_customer_id"],
                conversation_id=user_args["conversation_id"],
            )

            assert result == {"status": "success"}
            mock_create_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_new_loop)
            mock_new_loop.close.assert_called_once()


class TestUsageReportingServiceIntegration:
    """Integration tests verifying async and sync methods work together"""

    @pytest.fixture
    def user_args(self):
        """Standard arguments for usage reporting"""
        return {
            "user_id": "test-user-123",
            "dodo_customer_id": "dodo-cust-456",
            "conversation_id": "conv-789",
        }

    @pytest.mark.asyncio
    async def test_async_returns_correct_structure(self, user_args):
        """Test async method returns expected structure for success"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"usage_id": "usage-abc"}
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch(
            "app.modules.billing.usage_service.httpx.AsyncClient"
        ) as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await UsageReportingService.report_message_usage(**user_args)

            assert "status" in result or "usage_id" in result

    def test_sync_returns_same_structure_as_async(self, user_args):
        """Test sync method returns same structure as async (when successful)"""
        expected_result = {"status": "success", "usage_id": "usage-xyz"}

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = expected_result

        with (
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch(
                "app.modules.billing.usage_service.UsageReportingService.report_message_usage",
                new_callable=AsyncMock,
                return_value=expected_result,
            ),
        ):
            result = UsageReportingService.report_message_usage_sync(**user_args)

            assert result == expected_result
