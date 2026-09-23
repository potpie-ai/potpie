"""Process-isolated Windows Ladybug operations for the local daemon."""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import pickle
import subprocess
import sys
from typing import Any

from potpie.runtime.clients import (
    ClientOutcome,
    EngineOperation,
    TypedEngineOperationHandler,
)
from potpie.runtime.coordinator import OperationCoordinator
from potpie.runtime.local_engine import build_local_resource_manager
from potpie.runtime.composition import build_local_runtime
from potpie.runtime.server import AuthenticatedDaemonCaller
from potpie_context_engine import Failure, Success

logger = logging.getLogger(__name__)

_WORKER_FLAG = "--worker"
_DEFAULT_TIMEOUT_S = 120.0


class WindowsLadybugOperationHandler:
    """Keep Windows daemon semantic-search failures outside the daemon process."""

    def __init__(self, *, fallback: Any, backend_profile: str, backend: Any = None):
        self._fallback = fallback
        self._backend_profile = backend_profile.strip().lower()
        self._backend = backend

    async def handle(self, request, *, authentication: object) -> ClientOutcome:
        if sys.platform != "win32" or request.operation is not EngineOperation.SEARCH:
            return await self._fallback.handle(
                request,
                authentication=authentication,
            )
        logger.warning(
            "dispatching Windows daemon search to isolated worker "
            f"(request_id={request.request_id}, backend={self._backend_profile})",
        )
        try:
            # Doctor/status may have opened the parent's Ladybug handle. Win64
            # Ladybug is not safe to open from two processes at once, so close
            # the lazy shared provider before the child takes ownership. It
            # will reopen on the next parent-side operation if needed.
            await asyncio.to_thread(_close_backend_connections, self._backend)
        except Exception as exc:
            logger.exception(
                "could not release parent Ladybug connections before worker "
                f"(request_id={request.request_id})",
            )
            return _worker_failure(request, f"backend_close_{type(exc).__name__}")
        return await _run_search_worker(request)


async def _run_search_worker(request: Any) -> ClientOutcome:
    environment = os.environ.copy()
    # The worker executes the same local engine path as the passing in-process
    # CLI, without creating another detached daemon or touching its records.
    environment["CONTEXT_ENGINE_HOST_MODE"] = "in_process"
    environment["POTPIE_DAEMON_WORKER"] = "1"
    try:
        timeout_s = float(
            environment.get("POTPIE_DAEMON_WORKER_TIMEOUT_S", _DEFAULT_TIMEOUT_S)
        )
    except ValueError:
        timeout_s = _DEFAULT_TIMEOUT_S

    try:
        completed = await asyncio.to_thread(
            _run_worker_process,
            request,
            environment,
            timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_text = _decode_worker_output(exc.stderr)
        logger.error(
            "windows ladybug search worker timed out "
            f"(request_id={request.request_id})\n{stderr_text}".rstrip(),
        )
        return _worker_failure(request, "timeout")
    except Exception as exc:
        logger.exception(
            "windows ladybug search worker could not start "
            f"(request_id={request.request_id})",
        )
        return _worker_failure(request, type(exc).__name__)

    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    if stderr:
        stderr_text = _decode_worker_output(stderr)
        logger.info(
            "windows ladybug search worker stderr "
            f"(request_id={request.request_id}):\n{stderr_text}",
            extra={
                "request_id": request.request_id,
                "stderr": stderr_text,
            },
        )
    if completed.returncode != 0:
        logger.error(
            "windows ladybug search worker exited unsuccessfully "
            f"(request_id={request.request_id}, exit_code={completed.returncode})",
            extra={
                "request_id": request.request_id,
                "exit_code": completed.returncode,
            },
        )
        return _worker_failure(request, f"exit_{completed.returncode}")
    try:
        result = pickle.loads(stdout)  # noqa: S301 - private same-version worker
    except Exception:
        logger.exception(
            "windows ladybug search worker returned invalid output "
            f"(request_id={request.request_id})",
        )
        return _worker_failure(request, "invalid_output")
    if not isinstance(result, (Success, Failure)):
        logger.error(
            "windows ladybug search worker returned an invalid outcome "
            f"(request_id={request.request_id}, result_type={type(result).__name__})",
            extra={
                "request_id": request.request_id,
                "result_type": type(result).__name__,
            },
        )
        return _worker_failure(request, "invalid_outcome")
    return result


def _run_worker_process(
    request: Any, environment: dict[str, str], timeout_s: float
) -> subprocess.CompletedProcess[bytes]:
    """Run the native-risk child outside the daemon's Windows async pipe loop."""

    kwargs: dict[str, Any] = {
        "input": pickle.dumps(request),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": environment,
        "timeout": timeout_s,
        "check": False,
    }
    if sys.platform == "win32":
        # The daemon itself is windowless; do not create a console for its
        # short-lived native search child either.
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.run(  # noqa: S603 - launches this trusted interpreter
        [sys.executable, "-m", "potpie.daemon.windows_worker", _WORKER_FLAG],
        **kwargs,
    )


def _close_backend_connections(backend: Any) -> None:
    """Release parent-side native handles before a Windows search child starts."""

    if backend is None:
        return
    provider = getattr(backend, "graph_provider", None)
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def _decode_worker_output(value: bytes | str | None) -> str:
    if value is None:
        return "(no worker stderr)"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-4000:]
    return str(value)[-4000:]


def _worker_failure(request: Any, reason: str) -> Failure:
    from potpie.runtime.protocol import DaemonInternalError

    return Failure(
        DaemonInternalError(
            code="daemon_worker_failed",
            message="the isolated Windows Ladybug search worker failed",
            details={"request_id": request.request_id, "reason": reason},
            recommended_next_action="inspect the daemon log and retry the search",
        )
    )


def _run_worker() -> None:
    from potpie_context_engine.bootstrap.logging_setup import configure_logging

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        logger.warning("could not enable worker fault diagnostics", exc_info=True)
    configure_logging()
    try:
        from potpie_context_engine.adapters.outbound.graph.ladybug_windows_bootstrap import (
            ensure_ladybug_openssl,
        )

        openssl = ensure_ladybug_openssl()
        if not openssl.get("ok") and not openssl.get("skipped"):
            raise RuntimeError(
                "Ladybug OpenSSL bootstrap failed in isolated worker: "
                f"{openssl.get('error', 'unknown error')}"
            )
        request = pickle.loads(  # noqa: S301 - private same-version worker
            sys.stdin.buffer.read()
        )
        composition = build_local_runtime()
        handler = TypedEngineOperationHandler(
            build_local_resource_manager(composition.engine),
            coordinator=OperationCoordinator(),
            context_free_handler=composition.graph_metadata,
        )
        result = asyncio.run(
            handler.handle(request, authentication=AuthenticatedDaemonCaller())
        )
        sys.stdout.buffer.write(pickle.dumps(result))
        sys.stdout.buffer.flush()
    except BaseException:
        logger.critical("isolated Windows Ladybug worker failed", exc_info=True)
        raise


if __name__ == "__main__" and _WORKER_FLAG in sys.argv[1:]:
    _run_worker()
