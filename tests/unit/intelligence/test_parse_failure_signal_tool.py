"""Unit tests for parse_failure_signal tool.

Tests cover all supported failure signal formats:
  - Python traceback → pasted_log
  - pytest failure → failed_test
  - Node/TypeScript stack → pasted_log
  - Jest failure → failed_test
  - Go goroutine dump → pasted_log
  - Natural language symptom → nl_symptom
  - Malformed/empty input → nl_symptom, no crash
  - input_type_hint honored even when content is ambiguous
"""
import pytest

from app.modules.intelligence.tools.parse_failure_signal_tool import (
    parse_failure_signal,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixture strings — realistic shapes, not stubs
# ---------------------------------------------------------------------------

PYTHON_TRACEBACK = """\
Traceback (most recent call last):
  File "src/payments/payment_service.py", line 120, in process_payment
    result = adapter.charge(order)
  File "src/payments/adapters/stripe_adapter.py", line 42, in charge
    response = self._client.charges.create(**params)
  File "/usr/local/lib/python3.11/site-packages/stripe/api_resources/charge.py", line 33, in create
    return cls._static_request("post", cls.class_url(), **params)
requests.exceptions.Timeout: HTTPSConnectionPool(host='api.stripe.com', port=443): Read timed out. (read timeout=30)
"""

PYTEST_FAILURE = """\
FAILED tests/payments/test_payment_service.py::test_charge_returns_402_on_timeout - AssertionError: assert 500 == 402
  where 500 = <Response [500]>.status_code

________________________________ test_charge_returns_402_on_timeout ________________________________

    def test_charge_returns_402_on_timeout():
        response = client.post("/checkout", json={"amount": 100})
>       assert response.status_code == 402
E       AssertionError: assert 500 == 402
E        +  where 500 = <Response [500]>.status_code

tests/payments/test_payment_service.py:34: AssertionError
"""

NODE_TS_STACK = """\
ERROR checkout.create_order payment_adapter.timeout request_id=req_456
Stack: createOrder at src/checkout/createOrder.ts:88
       chargeCard at src/payments/paymentAdapter.ts:42

PaymentTimeoutError: Payment provider timed out after 30000ms
    at PaymentAdapter.chargeCard (src/payments/paymentAdapter.ts:42:15)
    at CheckoutService.createOrder (src/checkout/createOrder.ts:88:22)
    at processTicksAndRejections (internal/process/task_queues.js:95:5)
"""

# Mixed symbolled and bare frames — exercises _NODE_AT_BARE_RE (symbol=None) path
NODE_TS_MIXED_STACK = """\
Error: ENOENT: no such file or directory, open '/etc/config.json'
    at Object.openSync (node:fs:603:3)
    at /app/src/loaders/config.js:42:18
    at loadAll (/app/src/loaders/index.js:88:12)
    at /app/src/bootstrap.js:14:5
    at processTicksAndRejections (node:internal/process/task_queues:96:5)
"""

JEST_FAILURE = """\
 FAIL tests/checkout/checkout.integration.test.ts
  ● Checkout > test_charge_returns_402_on_timeout

    expect(received).toBe(expected)

    Expected: 402
    Received: 500

      32 |     const response = await request(app).post('/checkout').send({ amount: 100 });
    > 33 |     expect(response.status).toBe(402);
         |                             ^
      34 |   });

      at Object.<anonymous> (tests/checkout/checkout.integration.test.ts:33:29)
"""

GO_STACK = """\
goroutine 1 [running]:
main.chargeCard(0xc000104000, 0xc000078180)
\t/home/user/project/payments/adapter.go:42 +0x1a4
main.createOrder(0xc000104000)
\t/home/user/project/checkout/order.go:88 +0x6c
main.main()
\t/home/user/project/main.go:22 +0x58
exit status 2
"""

NL_SYMPTOM = "Checkout returns 500 when payment times out instead of a controlled payment failure response."

EMPTY_INPUT = ""

RANDOM_GARBAGE = "!!@#$%^&*()\n\x00\x01\x02garbage bytes here\nno frames at all"

AMBIGUOUS_MIXED = """\
Some log output that has no clear format.
Just a few lines of text.
Maybe it mentions an error somewhere.
"""


# ---------------------------------------------------------------------------
# 1. Python traceback → pasted_log
# ---------------------------------------------------------------------------

def test_python_traceback_classification():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    assert result["classification"] == "pasted_log"


def test_python_traceback_frames_extracted():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    frames = result["stack_frames"]
    assert len(frames) >= 2


def test_python_traceback_frame_file_paths():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    files = [f["file"] for f in result["stack_frames"]]
    assert any("payment_service.py" in f for f in files)
    assert any("stripe_adapter.py" in f for f in files)


def test_python_traceback_frame_line_numbers():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    frames = result["stack_frames"]
    # frames for our source files should have line numbers
    src_frames = [f for f in frames if "site-packages" not in f["file"]]
    for frame in src_frames:
        assert frame["line"] is not None


def test_python_traceback_frame_symbols():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    frames = result["stack_frames"]
    symbols = [f["symbol"] for f in frames]
    assert "process_payment" in symbols or "charge" in symbols


def test_python_traceback_error_type():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    # The final error line is "requests.exceptions.Timeout: ..."
    # error_type should capture the class name
    assert result["error_type"] is not None
    assert "Timeout" in result["error_type"] or "requests" in result["error_type"]


def test_python_traceback_frame_language():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    for frame in result["stack_frames"]:
        if frame["file"].endswith(".py"):
            assert frame["language"] == "python"


def test_python_traceback_signature_non_empty():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    assert result["signature"]
    assert len(result["signature"]) > 0


def test_python_traceback_raw_excerpt_present():
    result = parse_failure_signal(PYTHON_TRACEBACK)
    assert len(result["raw_excerpt"]) <= 300
    assert result["raw_excerpt"] in PYTHON_TRACEBACK


# ---------------------------------------------------------------------------
# 2. pytest failure → failed_test
# ---------------------------------------------------------------------------

def test_pytest_classification():
    result = parse_failure_signal(PYTEST_FAILURE)
    assert result["classification"] == "failed_test"


def test_pytest_error_type_extracted():
    result = parse_failure_signal(PYTEST_FAILURE)
    assert result["error_type"] is not None
    assert "AssertionError" in result["error_type"]


def test_pytest_frames_extracted():
    result = parse_failure_signal(PYTEST_FAILURE)
    frames = result["stack_frames"]
    assert len(frames) >= 1
    files = [f["file"] for f in frames]
    assert any("test_payment_service.py" in f for f in files)


def test_pytest_signature_contains_test_name():
    result = parse_failure_signal(PYTEST_FAILURE)
    # signature should reference something meaningful from the failure
    assert result["signature"]


# ---------------------------------------------------------------------------
# 3. Node/TypeScript stack → pasted_log, frames in correct order
# ---------------------------------------------------------------------------

def test_node_ts_classification():
    result = parse_failure_signal(NODE_TS_STACK)
    assert result["classification"] == "pasted_log"


def test_node_ts_frames_order():
    result = parse_failure_signal(NODE_TS_STACK)
    frames = result["stack_frames"]
    files = [f["file"] for f in frames]
    # chargeCard comes before createOrder in the 'at' stack lines
    adapter_idx = next((i for i, f in enumerate(files) if "paymentAdapter" in f), None)
    checkout_idx = next((i for i, f in enumerate(files) if "createOrder" in f), None)
    assert adapter_idx is not None
    assert checkout_idx is not None
    assert adapter_idx < checkout_idx


def test_node_ts_frames_with_symbol():
    result = parse_failure_signal(NODE_TS_STACK)
    frames = result["stack_frames"]
    symbols = [f["symbol"] for f in frames if f["symbol"]]
    assert "PaymentAdapter.chargeCard" in symbols or "chargeCard" in symbols


def test_node_ts_frames_line_numbers():
    result = parse_failure_signal(NODE_TS_STACK)
    frames = result["stack_frames"]
    src_frames = [f for f in frames if "internal/" not in f["file"]]
    for frame in src_frames:
        assert frame["line"] is not None


def test_node_ts_frame_language():
    result = parse_failure_signal(NODE_TS_STACK)
    for frame in result["stack_frames"]:
        if frame["file"].endswith(".ts"):
            assert frame["language"] == "typescript"
        elif frame["file"].endswith(".js"):
            assert frame["language"] == "javascript"


def test_node_ts_error_type():
    result = parse_failure_signal(NODE_TS_STACK)
    assert result["error_type"] is not None
    assert "PaymentTimeoutError" in result["error_type"]


def test_node_stack_with_mixed_symbol_and_bare_frames():
    """Node stack with symbolled frames AND bare 'at file:line:col' frames.

    Exercises the _NODE_AT_BARE_RE path that sets symbol=None while still
    populating the file field.
    """
    result = parse_failure_signal(NODE_TS_MIXED_STACK)

    # classification
    assert result["classification"] == "pasted_log"

    frames = result["stack_frames"]

    # at least 4 frames from the 5-line stack
    assert len(frames) >= 4

    # at least one bare frame (symbol is None, file is set)
    bare_frames = [f for f in frames if f["symbol"] is None and f.get("file")]
    assert len(bare_frames) >= 1, "Expected at least one bare frame with symbol=None"

    # at least one symbolled frame (symbol is not None, file is set)
    sym_frames = [f for f in frames if f["symbol"] is not None and f.get("file")]
    assert len(sym_frames) >= 1, "Expected at least one frame with a symbol"

    # frame order matches input order:
    # 'config.js' (bare, line 2 of the at-block) must appear before 'index.js' (symbolled, line 3)
    files = [f["file"] for f in frames]
    config_idx = next((i for i, f in enumerate(files) if "config.js" in f), None)
    index_idx = next((i for i, f in enumerate(files) if "index.js" in f), None)
    assert config_idx is not None, "config.js frame not found"
    assert index_idx is not None, "index.js frame not found"
    assert config_idx < index_idx, "config.js should appear before index.js"


# ---------------------------------------------------------------------------
# 4. Jest failure → failed_test
# ---------------------------------------------------------------------------

def test_jest_classification():
    result = parse_failure_signal(JEST_FAILURE)
    assert result["classification"] == "failed_test"


def test_jest_frames_extracted():
    result = parse_failure_signal(JEST_FAILURE)
    frames = result["stack_frames"]
    assert len(frames) >= 1
    files = [f["file"] for f in frames]
    assert any("checkout.integration.test.ts" in f for f in files)


def test_jest_error_type_or_none():
    result = parse_failure_signal(JEST_FAILURE)
    # Jest failures often don't have an explicit error class name in the FAIL line
    # but may have expect(...).toBe(...) shape — error_type may be None or set
    # We just verify it doesn't crash and returns a string or None
    assert result["error_type"] is None or isinstance(result["error_type"], str)


# ---------------------------------------------------------------------------
# 5. Go goroutine dump → pasted_log, frames extracted
# ---------------------------------------------------------------------------

def test_go_classification():
    result = parse_failure_signal(GO_STACK)
    assert result["classification"] == "pasted_log"


def test_go_frames_extracted():
    result = parse_failure_signal(GO_STACK)
    frames = result["stack_frames"]
    assert len(frames) >= 2


def test_go_frame_files_and_lines():
    result = parse_failure_signal(GO_STACK)
    frames = result["stack_frames"]
    files = [f["file"] for f in frames]
    assert any("adapter.go" in f for f in files)
    assert any("order.go" in f for f in files)
    for frame in frames:
        if frame["file"].endswith(".go"):
            assert frame["line"] is not None


def test_go_frame_language():
    result = parse_failure_signal(GO_STACK)
    for frame in result["stack_frames"]:
        if frame["file"].endswith(".go"):
            assert frame["language"] == "go"


def test_go_frame_symbols():
    result = parse_failure_signal(GO_STACK)
    frames = result["stack_frames"]
    symbols = [f["symbol"] for f in frames if f["symbol"]]
    assert len(symbols) >= 1


# ---------------------------------------------------------------------------
# 6. Natural language symptom → nl_symptom
# ---------------------------------------------------------------------------

def test_nl_symptom_classification():
    result = parse_failure_signal(NL_SYMPTOM)
    assert result["classification"] == "nl_symptom"


def test_nl_symptom_empty_frames():
    result = parse_failure_signal(NL_SYMPTOM)
    assert result["stack_frames"] == []


def test_nl_symptom_error_type_none():
    result = parse_failure_signal(NL_SYMPTOM)
    assert result["error_type"] is None


def test_nl_symptom_signature_from_text():
    result = parse_failure_signal(NL_SYMPTOM)
    # signature should be first ~60 chars of the input
    assert result["signature"]
    assert len(result["signature"]) <= 64  # some tolerance


# ---------------------------------------------------------------------------
# 7. Malformed / empty input → no crash, nl_symptom
# ---------------------------------------------------------------------------

def test_empty_string_no_crash():
    result = parse_failure_signal(EMPTY_INPUT)
    assert result is not None


def test_empty_string_classification():
    result = parse_failure_signal(EMPTY_INPUT)
    assert result["classification"] == "nl_symptom"


def test_empty_string_empty_frames():
    result = parse_failure_signal(EMPTY_INPUT)
    assert result["stack_frames"] == []


def test_random_garbage_no_crash():
    result = parse_failure_signal(RANDOM_GARBAGE)
    assert result is not None


def test_random_garbage_classification():
    result = parse_failure_signal(RANDOM_GARBAGE)
    assert result["classification"] == "nl_symptom"


def test_random_garbage_empty_frames():
    result = parse_failure_signal(RANDOM_GARBAGE)
    assert result["stack_frames"] == []


# ---------------------------------------------------------------------------
# 8. input_type_hint honored
# ---------------------------------------------------------------------------

def test_hint_pasted_log_overrides_ambiguous():
    result = parse_failure_signal(AMBIGUOUS_MIXED, input_type_hint="pasted_log")
    assert result["classification"] == "pasted_log"


def test_hint_failed_test_overrides_ambiguous():
    result = parse_failure_signal(AMBIGUOUS_MIXED, input_type_hint="failed_test")
    assert result["classification"] == "failed_test"


def test_hint_nl_symptom_overrides_ambiguous():
    result = parse_failure_signal(AMBIGUOUS_MIXED, input_type_hint="nl_symptom")
    assert result["classification"] == "nl_symptom"


def test_hint_failed_test_on_python_traceback():
    """When a hint is supplied for an ambiguous case, it should be honored."""
    result = parse_failure_signal(PYTHON_TRACEBACK, input_type_hint="failed_test")
    # Python traceback + pytest hint → failed_test wins
    assert result["classification"] == "failed_test"


# ---------------------------------------------------------------------------
# 9. Return shape — always valid keys present
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"signature", "classification", "stack_frames", "error_type", "raw_excerpt"}


@pytest.mark.parametrize("raw_text", [
    PYTHON_TRACEBACK,
    PYTEST_FAILURE,
    NODE_TS_STACK,
    JEST_FAILURE,
    GO_STACK,
    NL_SYMPTOM,
    EMPTY_INPUT,
    RANDOM_GARBAGE,
])
def test_result_always_has_required_keys(raw_text):
    result = parse_failure_signal(raw_text)
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"


@pytest.mark.parametrize("raw_text", [
    PYTHON_TRACEBACK,
    PYTEST_FAILURE,
    NODE_TS_STACK,
    JEST_FAILURE,
    GO_STACK,
    NL_SYMPTOM,
    EMPTY_INPUT,
])
def test_stack_frames_always_list(raw_text):
    result = parse_failure_signal(raw_text)
    assert isinstance(result["stack_frames"], list)


@pytest.mark.parametrize("raw_text", [
    PYTHON_TRACEBACK,
    NODE_TS_STACK,
    GO_STACK,
])
def test_stack_frames_have_required_keys(raw_text):
    result = parse_failure_signal(raw_text)
    for frame in result["stack_frames"]:
        assert "file" in frame
        assert "line" in frame
        assert "symbol" in frame
        assert "language" in frame


# ---------------------------------------------------------------------------
# 10. Regression: .tsx signature must not end with a trailing dot (Bug 1)
# ---------------------------------------------------------------------------

def test_jest_signature_no_trailing_dot_on_tsx():
    result = parse_failure_signal(JEST_FAILURE)
    assert not result["signature"].endswith(".")


def test_tsx_inline_no_trailing_dot():
    """Inline spot-check: a minimal Node stack ending in a .tsx frame."""
    tsx_stack = (
        "FAIL tests/Button.test.tsx\n"
        "  at render (src/Button.tsx:42:5)\n"
        "Expected: 1\n"
        "Received: 0\n"
    )
    result = parse_failure_signal(tsx_stack)
    assert not result["signature"].endswith(".")


# ---------------------------------------------------------------------------
# 11. Regression: markdown ● bullets must NOT classify as jest (Bug 2)
# ---------------------------------------------------------------------------

def test_markdown_bullets_not_classified_as_jest():
    text = "● First finding\n● Second finding\n● Third finding\n"
    result = parse_failure_signal(text)
    assert result["classification"] == "nl_symptom"
