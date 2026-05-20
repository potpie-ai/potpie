"""
Route high-level debug operations to the VS Code extension LocalServer via tunnel.

Socket.IO responses look like ``{success, result?, error?}``.
Direct HTTP to the LocalServer (when the backend resolves a local tunnel URL)
uses the same JSON shape.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from app.modules.intelligence.tools.local_search_tools.tunnel_utils import (
    SOCKET_TUNNEL_PREFIX,
    _execute_via_socket_full_response,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

DEBUG_ENDPOINTS: Dict[str, str] = {
    "debug_start": "/api/debug/start",
    "debug_stop": "/api/debug/stop",
    "debug_set_breakpoints": "/api/debug/breakpoints",
    "debug_snapshot": "/api/debug/snapshot",
    "debug_step_into": "/api/debug/step-into",
    "debug_step_out": "/api/debug/step-out",
    "debug_step_over": "/api/debug/step-over",
    "debug_continue": "/api/debug/continue",
    "debug_select_frame": "/api/debug/select-frame",
    "debug_list_sessions": "/api/debug/sessions",
    "debug_list_launch_configs": "/api/debug/launch-configs",
    "debug_list_adapters": "/api/debug/adapters",
}

_DEBUG_OPERATION_TIMEOUT_SECS: Dict[str, float] = {
    "debug_start": 120.0,
    "debug_snapshot": 120.0,
    "debug_step_into": 30.0,
    "debug_step_out": 30.0,
    "debug_step_over": 30.0,
    "debug_stop": 15.0,
    "debug_list_sessions": 15.0,
    "debug_continue": 10.0,
    "debug_set_breakpoints": 30.0,
    "debug_select_frame": 30.0,
    "debug_list_launch_configs": 5.0,
    "debug_list_adapters": 5.0,
}

_SNAPSHOT_LIKE_OPS = frozenset(
    {
        "debug_snapshot",
        "debug_step_into",
        "debug_step_out",
        "debug_step_over",
        "debug_select_frame",
    }
)


def _short_path(path: Optional[str]) -> str:
    if not path:
        return "?"
    base = os.path.basename(path.rstrip("/"))
    return base or path


def _normalize_extension_payload(body: Any) -> Dict[str, Any]:
    """Accept either bare ``{success,...}`` or legacy nested shapes."""
    if not isinstance(body, dict):
        return {"success": False, "error": "Invalid JSON response from extension"}
    if "success" in body:
        return body
    inner = body.get("body") if isinstance(body.get("body"), dict) else None
    if inner and "success" in inner:
        return inner
    return {"success": False, "error": str(body.get("error", "Unknown extension response"))}


def format_debug_result(operation: str, result: Any) -> str:
    """Turn extension ``result`` JSON into concise text for the LLM."""
    if result is None:
        return "✅ Debug operation completed (no result payload)."
    if not isinstance(result, dict):
        return str(result)

    if operation == "debug_set_breakpoints":
        return _format_breakpoints_result(result)

    if operation == "debug_list_sessions":
        return _format_sessions_result(result)

    if operation == "debug_list_launch_configs":
        return _format_launch_configs_result(result)

    if operation == "debug_list_adapters":
        return _format_adapters_result(result)

    if operation in _SNAPSHOT_LIKE_OPS:
        return _format_snapshot_result(result)

    lines = []
    if "session_id" in result:
        lines.append(f"session_id: {result['session_id']}")
    if "program" in result:
        lines.append(f"program: {result['program']}")
    if "language" in result:
        lines.append(f"language: {result['language']}")
    if "status" in result:
        lines.append(f"status: {result['status']}")
    if "stopped" in result:
        lines.append(f"stopped: {result['stopped']}")
    if not lines:
        text = "\n".join(f"{k}: {v}" for k, v in sorted(result.items()))
        return text[:8000]
    return "\n".join(lines)


def _format_snapshot_result(result: Dict[str, Any]) -> str:
    paused = result.get("paused_at") or {}
    fn = paused.get("function") or "?"
    line = paused.get("line")
    file_short = _short_path(paused.get("file"))
    head = (
        f"Paused at {file_short}:{line} in {fn}()"
        if line is not None
        else f"Paused at {file_short} in {fn}()"
    )

    out = [head, ""]

    stack = result.get("call_stack") or []
    if stack:
        out.append("Call Stack:")
        for i, frame in enumerate(stack[:25]):
            fp = _short_path(frame.get("file"))
            nm = frame.get("function") or "?"
            ln = frame.get("line")
            fid = frame.get("frame_id", "")
            suffix = f":{ln}" if ln is not None else ""
            out.append(f"  #{i}  {nm}()  {fp}{suffix}  (frame_id={fid})")
        if len(stack) > 25:
            out.append(f"  ... {len(stack) - 25} more frame(s)")
        out.append("")

    locals_ = result.get("locals") or {}
    if locals_:
        out.append("Local Variables:")
        for k, v in list(locals_.items())[:80]:
            vs = str(v)
            if len(vs) > 500:
                vs = vs[:497] + "..."
            out.append(f"  {k} = {vs}")
        if len(locals_) > 80:
            out.append(f"  ... {len(locals_) - 80} more variable(s)")
        out.append("")

    exprs = result.get("expression_results") or []
    if exprs:
        out.append("Expression Results:")
        for item in exprs:
            if not isinstance(item, dict):
                continue
            ex = item.get("expression", "?")
            if item.get("error"):
                out.append(f"  {ex} → error: {item['error']}")
            else:
                out.append(f"  {ex} = {item.get('result')}")
        out.append("")

    if result.get("session_id"):
        out.append(f"session_id: {result['session_id']}")
    if result.get("status"):
        out.append(f"status: {result['status']}")

    return "\n".join(out).strip()


def _format_breakpoints_result(result: Dict[str, Any]) -> str:
    file_path = result.get("file") or "?"
    short = _short_path(file_path)
    bps = result.get("breakpoints") or []
    lines = [f"Breakpoints in {short}:"]
    for bp in bps:
        if not isinstance(bp, dict):
            continue
        line_no = bp.get("line")
        verified = bp.get("verified", False)
        actual = bp.get("actual_line")
        msg = bp.get("message") or ""
        if verified:
            if actual is not None and actual != line_no:
                lines.append(
                    f"  Line {line_no}: verified (adjusted to line {actual})"
                    + (f" — {msg}" if msg else "")
                )
            else:
                lines.append(f"  Line {line_no}: verified" + (f" — {msg}" if msg else ""))
        else:
            lines.append(f"  Line {line_no}: not verified" + (f" — {msg}" if msg else ""))
    if len(lines) == 1:
        lines.append("  (no breakpoint details returned)")
    return "\n".join(lines)


def _format_sessions_result(result: Dict[str, Any]) -> str:
    sessions = result.get("sessions") or []
    if not sessions:
        return "Active debug sessions:\n  (none)"
    lines = [
        "Active debug sessions:",
        f"{'ID':<12} {'Program':<24} {'Lang':<10} Status",
        "-" * 60,
    ]
    for s in sessions[:30]:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("session_id", ""))[:12]
        prog = _short_path(s.get("program"))[:24]
        lang = str(s.get("language", ""))[:10]
        st = str(s.get("status", ""))
        lines.append(f"{sid:<12} {prog:<24} {lang:<10} {st}")
    if len(sessions) > 30:
        lines.append(f"... and {len(sessions) - 30} more")
    return "\n".join(lines)


def _format_launch_configs_result(result: Dict[str, Any]) -> str:
    configs = result.get("configs") or []
    if not configs:
        return "No launch configurations found. Check .vscode/launch.json in the workspace."
    lines = ["Available launch configurations:"]
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        name = cfg.get("name") or "Unnamed"
        typ = cfg.get("type") or "?"
        req = cfg.get("request") or "launch"
        prog = cfg.get("program") or cfg.get("module") or ""
        lines.append(f"  - **{name}** ({typ} / {req})" + (f" → `{prog}`" if prog else ""))
    return "\n".join(lines)


def _format_adapters_result(result: Dict[str, Any]) -> str:
    adapters = result.get("adapters") or []
    if not adapters:
        return "No debug adapters detected."
    lines = ["Available debug adapters:"]
    for a in adapters:
        if not isinstance(a, dict):
            continue
        lang = a.get("language") or "?"
        available = a.get("available", False)
        ext_id = a.get("extension_id") or ""
        status = "available" if available else "not installed"
        lines.append(f"  - **{lang}**: {status}" + (f" ({ext_id})" if ext_id else ""))
    return "\n".join(lines)


def _try_http_local_debug(
    endpoint: str,
    request_data: Dict[str, Any],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    force_tunnel = os.getenv("FORCE_TUNNEL", "").lower() in ["true", "1", "yes"]
    base_url = os.getenv("BASE_URL", "").lower()
    environment = os.getenv("ENVIRONMENT", "").lower()
    is_backend_local = (
        "localhost" in base_url
        or "127.0.0.1" in base_url
        or environment in ["local", "dev", "development"]
        or not base_url
    )
    if not is_backend_local or force_tunnel:
        return None

    from app.modules.tunnel.tunnel_service import _get_local_tunnel_server_url

    direct_base = _get_local_tunnel_server_url()
    if not direct_base:
        local_port_env = os.getenv("LOCAL_SERVER_PORT")
        direct_port = int(local_port_env) if local_port_env else 3001
        direct_base = f"http://localhost:{direct_port}"

    url = f"{direct_base.rstrip('/')}{endpoint}"
    try:
        with httpx.Client(timeout=2.0) as health_client:
            health = health_client.get(f"{direct_base.rstrip('/')}/health")
            if health.status_code != 200:
                return None
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            logger.warning(
                "[debug_tunnel] HTTP %s %s — %s",
                resp.status_code,
                url,
                resp.text[:300],
            )
            return None
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[debug_tunnel] direct HTTP unavailable: %s", exc)
        return None


def route_debug_command(
    operation: str,
    data: Dict[str, Any],
    user_id: Optional[str],
    conversation_id: Optional[str],
) -> str:
    """Execute a debug RPC via tunnel (socket preferred; optional direct HTTP when backend is local)."""
    endpoint = DEBUG_ENDPOINTS.get(operation)
    if not endpoint:
        return f"❌ Unknown debug operation: {operation}"

    if not user_id:
        return (
            "❌ Debug tools require an authenticated user context and VS Code extension tunnel.\n\n"
            "Ensure you are running in local mode with the Potpie extension connected."
        )

    timeout = _DEBUG_OPERATION_TIMEOUT_SECS.get(operation, 60.0)
    request_data = {**data, "conversation_id": conversation_id}

    from app.modules.intelligence.tools.code_changes_manager import (
        _get_tunnel_url,
        _get_repository,
        _get_branch,
    )
    from app.modules.tunnel.tunnel_service import get_tunnel_service

    context_tunnel_url = _get_tunnel_url()
    repository = _get_repository()
    branch = _get_branch()
    tunnel_service = get_tunnel_service()
    tunnel_url = tunnel_service.get_tunnel_url(
        user_id,
        conversation_id,
        tunnel_url=context_tunnel_url,
        repository=repository,
        branch=branch,
    )

    if not tunnel_url:
        return (
            "❌ No VS Code extension tunnel available for this workspace.\n\n"
            "**Fix:** Connect the Potpie extension (workspace registered), ensure repository context "
            "is present for this conversation, then retry."
        )

    raw: Optional[Dict[str, Any]] = None

    http_body = _try_http_local_debug(endpoint, request_data, timeout)
    if isinstance(http_body, dict):
        raw = _normalize_extension_payload(http_body)

    if raw is None and tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
        logger.info("[debug_tunnel] routing %s via Socket.IO → %s", operation, endpoint)
        sock = _execute_via_socket_full_response(
            user_id=user_id,
            conversation_id=conversation_id,
            endpoint=endpoint,
            payload=request_data,
            tunnel_url=tunnel_url,
            repository=repository,
            branch=branch,
            timeout=timeout,
        )
        raw = _normalize_extension_payload(sock) if sock else None

    if raw is None and tunnel_url and not tunnel_url.startswith(SOCKET_TUNNEL_PREFIX):
        base = tunnel_url.rstrip("/")
        url = f"{base}{endpoint}"
        try:
            logger.info("[debug_tunnel] routing %s via HTTP tunnel → %s", operation, url)
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    url,
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
            if resp.status_code == 200:
                raw = _normalize_extension_payload(resp.json())
            else:
                logger.warning(
                    "[debug_tunnel] HTTP tunnel %s failed: %s %s",
                    url,
                    resp.status_code,
                    resp.text[:300],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[debug_tunnel] HTTP tunnel error %s: %s", url, exc)

    if raw is None:
        return (
            f"❌ Debug request failed or timed out ({operation}).\n\n"
            "Check that the VS Code extension is connected and implements "
            f"`POST {endpoint}`."
        )

    if raw.get("success") and "result" in raw:
        try:
            return format_debug_result(operation, raw["result"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("format_debug_result failed: %s", exc)
            return f"✅ Success but formatting failed: {raw['result']!r}"

    err = raw.get("error") or raw.get("message") or "Unknown error"
    return f"❌ Debug operation failed ({operation}): {err}"
