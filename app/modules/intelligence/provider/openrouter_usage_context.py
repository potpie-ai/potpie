"""
Context for collecting OpenRouter usage during a Celery agent task.
The worker publishes this in the stream 'end' event so the API can log it (uvicorn).
"""
from contextvars import ContextVar
from typing import List, Dict, Any, Optional

_openrouter_usage_list: ContextVar[List[Dict[str, Any]]] = ContextVar(
    "openrouter_usage_list"
)

# Rough credits per 1K tokens for estimation when API doesn't return cost (OpenRouter-style).
# Update from https://openrouter.ai/docs/models if you want better estimates.
_ESTIMATE_PER_1K_INPUT = 0.0001
_ESTIMATE_PER_1K_OUTPUT = 0.0003


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in credits when API doesn't provide it."""
    return (prompt_tokens / 1000.0 * _ESTIMATE_PER_1K_INPUT) + (
        completion_tokens / 1000.0 * _ESTIMATE_PER_1K_OUTPUT
    )


def push_usage(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost: Optional[float] = None,
) -> None:
    """Append one OpenRouter usage record to the current context (used in Celery task)."""
    try:
        lst = _openrouter_usage_list.get()
    except LookupError:
        lst = []
        _openrouter_usage_list.set(lst)
    lst.append({
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    })


def get_and_clear_usages() -> List[Dict[str, Any]]:
    """Return and clear the list of usage records for the current context (call from agent task)."""
    try:
        lst = _openrouter_usage_list.get()
        out = list(lst)
        lst.clear()
        return out
    except LookupError:
        return []


def init_usage_context() -> None:
    """Initialize an empty list for the current context (call at start of agent task)."""
    _openrouter_usage_list.set([])


def push_usage_from_run(run_usage: Any, model_name: str = "openrouter") -> None:
    """Append usage from a pydantic_ai RunUsage (e.g. run.usage() after stream completes).
    RunUsage has no cost; we pass None (caller can show estimate in logs via estimate_cost_for_log).
    """
    if run_usage is None:
        return
    prompt_tokens = getattr(run_usage, "input_tokens", 0) or 0
    completion_tokens = getattr(run_usage, "output_tokens", 0) or 0
    total_tokens = getattr(run_usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    push_usage(model_name, prompt_tokens, completion_tokens, total_tokens, cost=None)


def estimate_cost_for_log(prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in credits for use in log messages when API didn't return cost."""
    return _estimate_cost(prompt_tokens, completion_tokens)
