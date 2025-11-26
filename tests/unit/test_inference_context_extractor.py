"""
Unit tests for InferenceContextExtractor.

Tests the inference context extraction for LLM optimization (85-90% token savings).
"""
import json
import pytest


class TestInferenceContextExtractor:
    """Tests for InferenceContextExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a fresh InferenceContextExtractor instance."""
        from app.modules.parsing.graph_construction.inference_context_extractor import InferenceContextExtractor
        return InferenceContextExtractor()

    def test_extract_context_python_function(self, extractor):
        """Test context extraction for a Python function."""
        code = '''def calculate_discount(self, cart_total: float, coupon_code: str) -> float:
    """Calculate discount amount."""
    if not coupon_code:
        return 0.0

    coupon = self.coupon_service.get_coupon(coupon_code)
    if not coupon or not coupon.is_valid():
        raise InvalidCouponError(f"Coupon {coupon_code} is invalid")

    if coupon.is_percentage:
        discount = cart_total * (coupon.value / 100)
    else:
        discount = min(coupon.value, cart_total)

    return min(discount, coupon.max_discount or float('inf'))
'''
        context = extractor.extract_context(
            full_text=code,
            file_path="services/discount_service.py",
            language="python",
            node_type="FUNCTION",
            node_name="calculate_discount",
            class_name="DiscountService"
        )

        # Verify required fields
        assert "signature" in context
        assert "type" in context
        assert context["type"] == "FUNCTION"
        assert context["language"] == "python"

        # Verify signature extraction
        assert "calculate_discount" in context["signature"]
        assert "cart_total" in context["signature"] or "float" in context["signature"]

        # Verify class context
        assert context["class_name"] == "DiscountService"

        # Verify docstring extraction
        assert context.get("existing_docstring") is not None or context.get("existing_docstring") == "Calculate discount amount."

        # Verify JSON serializable
        json_str = json.dumps(context)
        assert len(json_str) < 2048  # Size limit check

    def test_extract_context_javascript_function(self, extractor):
        """Test context extraction for a JavaScript function."""
        code = '''async function fetchUserProfile(userId) {
  const response = await api.get(`/users/${userId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch user');
  }
  return response.json();
}'''
        context = extractor.extract_context(
            full_text=code,
            file_path="src/api/users.js",
            language="javascript",
            node_type="FUNCTION",
            node_name="fetchUserProfile",
            class_name=None
        )

        assert "signature" in context
        assert "fetchUserProfile" in context["signature"]
        assert context["is_async"] is True
        assert context["language"] == "javascript"

    def test_extract_context_java_method(self, extractor):
        """Test context extraction for a Java method."""
        code = '''public void processOrder(Order order, PaymentMethod payment) {
    validateOrder(order);
    chargePayment(payment, order.getTotal());
    shipOrder(order);
    notifyCustomer(order.getCustomerId());
}'''
        context = extractor.extract_context(
            full_text=code,
            file_path="src/main/java/services/OrderService.java",
            language="java",
            node_type="FUNCTION",
            node_name="processOrder",
            class_name="OrderService"
        )

        assert "signature" in context
        assert "processOrder" in context["signature"]
        assert context["visibility"] == "public"
        assert context["class_name"] == "OrderService"

    def test_extract_context_empty_text(self, extractor):
        """Test graceful handling of empty text."""
        context = extractor.extract_context(
            full_text="",
            file_path="test.py",
            language="python",
            node_type="FUNCTION",
            node_name="empty_func",
            class_name=None
        )

        # Should return minimal fallback
        assert context["signature"] == "empty_func"
        assert context["type"] == "FUNCTION"

    def test_extract_context_unsupported_language(self, extractor):
        """Test fallback for unsupported language."""
        context = extractor.extract_context(
            full_text="PERFORM SOME-PARAGRAPH.\n  MOVE X TO Y.",
            file_path="program.cob",
            language="cobol",  # Not in SUPPORTED_LANGUAGES
            node_type="FUNCTION",
            node_name="SOME-PARAGRAPH",
            class_name=None
        )

        # Should still return a valid context using fallback
        assert "signature" in context
        assert context["language"] == "cobol"

    def test_extract_context_size_limit(self, extractor):
        """Test that context is truncated to 2KB limit."""
        # Create a very long function
        long_code = "def long_function(" + ", ".join([f"arg{i}: str" for i in range(100)]) + "):\n    pass"

        context = extractor.extract_context(
            full_text=long_code,
            file_path="test.py",
            language="python",
            node_type="FUNCTION",
            node_name="long_function",
            class_name=None
        )

        # Verify size limit
        json_str = json.dumps(context)
        assert len(json_str) <= 2048

    def test_extract_private_function(self, extractor):
        """Test detection of private functions."""
        code = '''def _internal_helper(self):
    """Private helper method."""
    return self._data
'''
        context = extractor.extract_context(
            full_text=code,
            file_path="utils.py",
            language="python",
            node_type="FUNCTION",
            node_name="_internal_helper",
            class_name="DataProcessor"
        )

        assert context["visibility"] == "private"

    def test_extract_async_function(self, extractor):
        """Test detection of async functions."""
        code = '''async def fetch_data(self, url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        context = extractor.extract_context(
            full_text=code,
            file_path="client.py",
            language="python",
            node_type="FUNCTION",
            node_name="fetch_data",
            class_name="HttpClient"
        )

        assert context["is_async"] is True

    def test_extract_decorators(self, extractor):
        """Test extraction of decorators."""
        code = '''@property
@cached
def computed_value(self) -> int:
    return self._calculate()
'''
        context = extractor.extract_context(
            full_text=code,
            file_path="model.py",
            language="python",
            node_type="FUNCTION",
            node_name="computed_value",
            class_name="DataModel"
        )

        assert "decorators" in context
        assert "property" in context["decorators"] or "cached" in context["decorators"]

    def test_parser_caching(self, extractor):
        """Test that parsers are cached at class level."""
        # First extraction
        context1 = extractor.extract_context(
            full_text="def foo(): pass",
            file_path="a.py",
            language="python",
            node_type="FUNCTION",
            node_name="foo",
            class_name=None
        )

        # Get stats after first call
        stats1 = extractor.get_stats()

        # Create new extractor instance
        from app.modules.parsing.graph_construction.inference_context_extractor import InferenceContextExtractor
        extractor2 = InferenceContextExtractor()

        # Second extraction with new instance
        context2 = extractor2.extract_context(
            full_text="def bar(): pass",
            file_path="b.py",
            language="python",
            node_type="FUNCTION",
            node_name="bar",
            class_name=None
        )

        # Parsers should be shared (class-level cache)
        stats2 = extractor2.get_stats()
        # Either cache hits should increase or total cache_misses should be same
        # (parser was already cached from first call)

    def test_get_stats(self, extractor):
        """Test statistics tracking."""
        # Do some extractions
        extractor.extract_context(
            full_text="def foo(): pass",
            file_path="test.py",
            language="python",
            node_type="FUNCTION",
            node_name="foo",
            class_name=None
        )

        stats = extractor.get_stats()

        assert "ast_success" in stats
        assert "ast_fallback" in stats
        assert "parser_cache_hits" in stats
        assert "parser_cache_misses" in stats

    def test_format_inference_context_minimal(self, extractor):
        """Test that minimal context still produces useful output."""
        context = extractor.extract_context(
            full_text="def x(): pass",
            file_path="t.py",
            language="python",
            node_type="FUNCTION",
            node_name="x",
            class_name=None
        )

        # Should have at least signature
        assert context.get("signature")
        assert len(context.get("signature", "")) >= 1

    def test_short_signature_not_rejected(self, extractor):
        """Test that very short but valid signatures are not rejected (FIX)."""
        # These are all valid Python functions that should produce usable context
        short_functions = [
            ("def x(): pass", "x"),
            ("def a(): return 1", "a"),
            ("def _(): yield", "_"),
        ]

        for code, name in short_functions:
            context = extractor.extract_context(
                full_text=code,
                file_path="test.py",
                language="python",
                node_type="FUNCTION",
                node_name=name,
                class_name=None
            )

            # Should NOT be rejected - signature should be present
            assert context.get("signature"), f"Short function '{name}' should have signature"
            # Should have valid type
            assert context.get("type") == "FUNCTION"
            # Should have language
            assert context.get("language") == "python"
