from __future__ import annotations

from typing import Any

from domain.ports.daemon.operations import OperationError

_STATUS_MAP = {
    "invalid_input": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "unavailable": 503,
    "degraded": 503,
    "internal_error": 500,
}


def error_envelope(error: OperationError) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "detail": error.detail,
            "recommended_next_action": error.recommended_next_action,
        }
    }


def status_for_error(code: str) -> int:
    return _STATUS_MAP.get(code, 500)
