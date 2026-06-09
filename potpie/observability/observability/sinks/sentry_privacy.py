from __future__ import annotations

import os
import posixpath
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

from ..redaction import sanitize_log_text

SENTRY_ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "service",
        "environment",
        "release",
        "cli_version",
        "python_version",
        "os",
        "arch",
        "command",
        "subcommand",
        "output_mode",
        "exit_code",
        "error.code",
        "error.kind",
        "is_expected",
    }
)
SENTRY_TELEMETRY_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {"anonymous_install_id", "invocation_id", "daemon_session_id"}
)
_DROP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "headers",
        "header",
        "authorization",
        "cookie",
        "cookies",
        "request",
        "response",
        "body",
        "data",
        "prompt",
        "episode",
        "episode_body",
        "terminal_output",
        "stdout",
        "stderr",
        "source",
        "source_code",
        "file_contents",
        "repo",
        "repo_name",
        "repository",
        "git_remote",
        "git_url",
        "env",
        "environment_variables",
        "vars",
        "value",
    }
)
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,}|(?:token|secret|password|api[_-]?key)=\S+|bearer\s+\S+)"
)
_ABS_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.-])(?:/[A-Za-z0-9._ -]+){2,}"
)
_WINDOWS_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\){1,}[^\\/:*?\"<>|\r\n]*"
)
_GIT_REMOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:git@[\w.-]+:[\w.-]+/[\w.-]+(?:\.git)?|"
    r"https?://[\w.-]+/[\w.-]+/[\w.-]+(?:\.git)?)"
    r"(?=[\s'\"),;]|$)"
)
_REPO_SLUG_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![:/\w.-])[\w.-]+/[\w.-]+(?:\.git)?(?![\w.-])"
)
_SENSITIVE_LOG_MESSAGE_MARKERS: Final[tuple[str, ...]] = (
    "prompt",
    "episode",
    "source code",
    "file contents",
    "terminal output",
    "confidential",
    "secret",
    "token",
    "password",
    "authorization",
)
_MAX_LOG_MESSAGE_CHARS: Final[int] = 240


@dataclass(slots=True)
class SentryRateLimiter:
    max_events_per_minute: int
    clock: Callable[[], float] = time.monotonic
    _buckets: dict[tuple[str, str, str], tuple[int, float]] = field(
        default_factory=dict
    )

    def allow(self, event: dict[str, Any]) -> bool:
        if self.max_events_per_minute <= 0:
            return False
        bucket = _rate_bucket(event)
        now = self.clock()
        count, started_at = self._buckets.get(bucket, (0, now))
        if now - started_at >= 60:
            count = 0
            started_at = now
        if count >= self.max_events_per_minute:
            self._buckets[bucket] = (count, started_at)
            return False
        self._buckets[bucket] = (count + 1, started_at)
        return True


def split_sentry_fields(
    fields: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    tags = {
        key: str(value)
        for key, value in fields.items()
        if key in SENTRY_ALLOWED_TAGS and value is not None
    }
    telemetry = {
        key: str(value)
        for key, value in fields.items()
        if key in SENTRY_TELEMETRY_CONTEXT_KEYS and value is not None
    }
    contexts = {"telemetry": telemetry} if telemetry else {}
    return tags, contexts


def scrub_sentry_event(
    event: dict[str, Any],
    hint: dict[str, Any],
    *,
    rate_limiter: SentryRateLimiter | None = None,
) -> dict[str, Any] | None:
    tags = _dict_value(event.get("tags"))
    if str(tags.get("is_expected", "")).lower() == "true":
        return None
    scrubbed = _scrub_value(event)
    if not isinstance(scrubbed, dict):
        return None
    filtered = dict(scrubbed)
    filtered["tags"] = _filter_tags(_dict_value(scrubbed.get("tags")))
    filtered["contexts"] = _filter_contexts(_dict_value(scrubbed.get("contexts")))
    filtered.pop("logentry", None)
    filtered.pop("request", None)
    filtered.pop("server_name", None)
    filtered.pop("user", None)
    _scrub_exception(filtered)
    if rate_limiter is not None and not rate_limiter.allow(filtered):
        return None
    return filtered


def scrub_sentry_breadcrumb(
    breadcrumb: dict[str, Any],
    hint: dict[str, Any],
) -> dict[str, Any] | None:
    message = str(breadcrumb.get("message", ""))
    category = str(breadcrumb.get("category", ""))
    if category == "cli" or "--" in message or "token" in message.lower():
        return None
    scrubbed = _scrub_value(breadcrumb)
    return scrubbed if isinstance(scrubbed, dict) else None


def scrub_sentry_log_message(message: str) -> str | None:
    lowered = message.lower()
    if any(marker in lowered for marker in _SENSITIVE_LOG_MESSAGE_MARKERS):
        return None
    scrubbed = _scrub_string(message).strip()
    if not scrubbed:
        return None
    return scrubbed[:_MAX_LOG_MESSAGE_CHARS]


def require_staging_smoke_environment() -> None:
    if os.getenv("RUN_SENTRY_STAGING_SMOKE") != "1":
        try:
            import pytest

            pytest.skip("set RUN_SENTRY_STAGING_SMOKE=1 to run staging smoke")
        except ModuleNotFoundError as exc:
            raise RuntimeError("set RUN_SENTRY_STAGING_SMOKE=1") from exc
    missing = [
        name
        for name in ("SENTRY_DSN", "SENTRY_ENVIRONMENT", "SENTRY_RELEASE")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"missing staging smoke env vars: {', '.join(missing)}")
    if os.getenv("SENTRY_ENVIRONMENT") != "staging":
        raise RuntimeError("staging smoke requires SENTRY_ENVIRONMENT=staging")


def _rate_bucket(event: dict[str, Any]) -> tuple[str, str, str]:
    tags = _dict_value(event.get("tags"))
    contexts = _dict_value(event.get("contexts"))
    telemetry = _dict_value(contexts.get("telemetry"))
    install_id = str(
        telemetry.get("anonymous_install_id")
        or tags.get("anonymous_install_id")
        or "unknown-install"
    )
    release = str(tags.get("release") or event.get("release") or "unknown-release")
    error_code = str(tags.get("error.code") or "unknown_error")
    return install_id, release, error_code


def _filter_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in tags.items()
        if key in SENTRY_ALLOWED_TAGS and value is not None
    }


def _filter_contexts(contexts: dict[str, Any]) -> dict[str, dict[str, str]]:
    telemetry = _dict_value(contexts.get("telemetry"))
    filtered = {
        key: str(value)
        for key, value in telemetry.items()
        if key in SENTRY_TELEMETRY_CONTEXT_KEYS and value is not None
    }
    return {"telemetry": filtered} if filtered else {}


def _scrub_exception(event: dict[str, Any]) -> None:
    exception = _dict_value(event.get("exception"))
    values = exception.get("values")
    if not isinstance(values, list):
        return
    for value in values:
        if not isinstance(value, dict):
            continue
        value.pop("value", None)
        stacktrace = _dict_value(value.get("stacktrace"))
        frames = stacktrace.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if isinstance(frame, dict):
                frame.pop("abs_path", None)
                frame.pop("context_line", None)
                frame.pop("post_context", None)
                frame.pop("pre_context", None)
                frame.pop("vars", None)
                filename = frame.get("filename")
                if isinstance(filename, str):
                    frame["filename"] = posixpath.basename(filename)


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_value(item) for item in value]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _DROP_KEYS:
                continue
            if key_text == "filename" and isinstance(child, str):
                clean[key_text] = posixpath.basename(child.replace("\\", "/"))
                continue
            clean[key_text] = _scrub_value(child)
        return clean
    return value


def _scrub_string(value: str) -> str:
    without_remote = _GIT_REMOTE_RE.sub("[redacted-git-remote]", value)
    without_repo = _REPO_SLUG_RE.sub("[redacted-repo]", without_remote)
    sanitized = sanitize_log_text(without_repo)
    without_email = _EMAIL_RE.sub("[redacted-email]", sanitized)
    without_token = _TOKEN_RE.sub("[redacted-secret]", without_email)
    without_unix_path = _ABS_PATH_RE.sub("[redacted-path]", without_token)
    without_path = _WINDOWS_PATH_RE.sub("[redacted-path]", without_unix_path)
    if "://" not in without_path:
        return without_path
    parts = urlsplit(without_path)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
