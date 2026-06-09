# ruff: noqa: S101
from __future__ import annotations

from observability.sinks.sentry_privacy import (
    SentryRateLimiter,
    scrub_sentry_breadcrumb,
    scrub_sentry_event,
)


def test_sentry_event_is_dropped_when_expected() -> None:
    event = {"tags": {"is_expected": "true", "error.code": "validation_error"}}

    assert scrub_sentry_event(event, {}) is None


def test_sentry_event_scrubs_forbidden_values_and_keeps_allowlist() -> None:
    event = {
        "message": "failed for user deeptendu@example.com token=ghp_123456789012345678901234567890123456",
        "tags": {
            "command": "query",
            "repo_name": "secret-repo",
            "error.code": "unexpected_cli_error",
        },
        "contexts": {
            "telemetry": {
                "anonymous_install_id": "install-1",
                "invocation_id": "invoke-1",
                "daemon_session_id": "daemon-1",
                "repo": "secret-repo",
            }
        },
        "server_name": "Frieren.local",
        "request": {
            "url": "https://example.test/path?token=secret",
            "headers": {"Authorization": "Bearer secret"},
            "data": {"prompt": "index this private source"},
        },
        "exception": {
            "values": [
                {
                    "value": "raw path /Users/dsantra/private/repo token=secret",
                    "stacktrace": {
                        "frames": [
                            {
                                "abs_path": "/Users/dsantra/private/repo/app.py",
                                "context_line": "print('private source')",
                                "filename": "/Users/dsantra/private/repo/app.py",
                                "post_context": ["print('after')"],
                                "pre_context": ["print('before')"],
                                "vars": {"password": "secret"},
                            }
                        ]
                    },
                }
            ]
        },
    }

    scrubbed = scrub_sentry_event(event, {})

    assert scrubbed is not None
    assert scrubbed["tags"] == {
        "command": "query",
        "error.code": "unexpected_cli_error",
    }
    exception = scrubbed["exception"]["values"][0]
    assert "value" not in exception
    assert "vars" not in exception["stacktrace"]["frames"][0]
    assert exception["stacktrace"]["frames"][0]["filename"] == "app.py"
    assert "server_name" not in scrubbed
    assert scrubbed["contexts"] == {
        "telemetry": {
            "anonymous_install_id": "install-1",
            "invocation_id": "invoke-1",
            "daemon_session_id": "daemon-1",
        }
    }
    blob = repr(scrubbed)
    assert "deeptendu@example.com" not in blob
    assert "ghp_" not in blob
    assert "secret-repo" not in blob
    assert "Authorization" not in blob
    assert "index this private source" not in blob
    assert "/Users/dsantra/private" not in blob
    assert "private source" not in blob
    assert "print('before')" not in blob
    assert "vars" not in blob


def test_sentry_event_scrubs_repo_slugs_and_git_remotes() -> None:
    event = {
        "message": (
            "failed in repo acme/secret-repo from "
            "git@github.com:acme/secret-repo.git and "
            "https://github.com/acme/secret-repo.git"
        ),
        "tags": {"error.code": "unexpected_cli_error"},
    }

    scrubbed = scrub_sentry_event(event, {})

    assert scrubbed is not None
    blob = repr(scrubbed)
    assert "acme/secret-repo" not in blob
    assert "git@github.com" not in blob
    assert "secret-repo.git" not in blob


def test_sentry_event_scrubs_windows_paths() -> None:
    event = {
        "message": r"failed reading C:\Users\deeptendu\repo\secret.py",
        "tags": {"error.code": "unexpected_cli_error"},
    }

    scrubbed = scrub_sentry_event(event, {})

    assert scrubbed is not None
    assert r"C:\Users\deeptendu" not in repr(scrubbed)


def test_sentry_breadcrumb_drops_command_argument_surfaces() -> None:
    breadcrumb = {"category": "cli", "message": "potpie query --token secret"}

    assert scrub_sentry_breadcrumb(breadcrumb, {}) is None


def test_sentry_breadcrumb_scrubs_common_secret_shapes() -> None:
    breadcrumb = {
        "category": "log",
        "message": (
            "Authorization: Basic abcdefghijklmnop "
            "Bearer qwertyuiopasdfghjklzxcvbnm "
            "sk-123456789012345678901234 "
            "AKIAABCDEFGHIJKLMNOP "
            "xoxb-1234567890-secret"
        ),
        "data": {
            "url": "https://example.test/path?token=secret",
            "headers": {"Authorization": "Bearer secret"},
        },
    }

    scrubbed = scrub_sentry_breadcrumb(breadcrumb, {})

    assert scrubbed is not None
    blob = repr(scrubbed)
    assert "Basic abcdefghijklmnop" not in blob
    assert "Bearer qwertyuiopasdfghjklzxcvbnm" not in blob
    assert "sk-123456789012345678901234" not in blob
    assert "AKIAABCDEFGHIJKLMNOP" not in blob
    assert "xoxb-1234567890-secret" not in blob
    assert "headers" not in blob


def test_sentry_rate_limiter_caps_by_install_release_and_code() -> None:
    limiter = SentryRateLimiter(max_events_per_minute=1)
    event = {
        "tags": {
            "anonymous_install_id": "install-1",
            "release": "potpie-cli@1.0.0",
            "error.code": "unexpected_cli_error",
        }
    }

    assert limiter.allow(event) is True
    assert limiter.allow(event) is False
