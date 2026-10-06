from __future__ import annotations

import ast
import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

DISTRIBUTION_DEFAULTS_OUT = Path("potpie/runtime/_distribution_defaults.py")
BUILD_INFO_OUT = Path("potpie/runtime/_build_info.py")
DISTRIBUTION_DEFAULT_INPUT_NAMES_BY_FIELD = {
    "environment": "POTPIE_ENVIRONMENT",
    "sentry_dsn": "POTPIE_SENTRY_DSN",
    "posthog_api_key": "POTPIE_POSTHOG_API_KEY",
    "posthog_host": "POTPIE_POSTHOG_HOST",
    "linear_client_id": "LINEAR_CLIENT_ID",
    "github_client_id": "POTPIE_GITHUB_CLIENT_ID",
}
BUILD_INFO_INPUT_NAMES_BY_FIELD = {
    "GIT_SHA": ("POTPIE_BUILD_GIT_SHA", "GITHUB_SHA"),
    "DIRTY": ("POTPIE_BUILD_DIRTY",),
    "BUILD_TIME": ("POTPIE_BUILD_TIME",),
}

HEADER = """\
# Auto-generated at wheel build time - do not edit manually.
# Runtime environment variables override these packaged public defaults.
"""

DEFAULT_DISTRIBUTION_ENVIRONMENT = "prod_oss"
DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"
REQUIRED_DISTRIBUTION_DEFAULTS = (
    "environment",
    "sentry_dsn",
    "posthog_api_key",
    "posthog_host",
    "linear_client_id",
    "github_client_id",
)
_DOTENV_SEARCH_START: Final[Path] = Path(__file__).resolve().parent
# The project root this helper ships in (``scripts/`` sits directly under it),
# in a checkout and in an unpacked sdist alike.
_SOURCE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_GIT_TIMEOUT_S: Final[float] = 10.0
_TRUE_FLAGS: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_FLAGS: Final = frozenset({"0", "false", "no", "off"})


def distribution_default_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return {
        "environment": _env_or_default(
            "POTPIE_ENVIRONMENT", DEFAULT_DISTRIBUTION_ENVIRONMENT, source
        ),
        "sentry_dsn": _env("POTPIE_SENTRY_DSN", source),
        "posthog_api_key": _env("POTPIE_POSTHOG_API_KEY", source),
        "posthog_host": _env_or_default(
            "POTPIE_POSTHOG_HOST", DEFAULT_POSTHOG_HOST, source
        ),
        "linear_client_id": _env("LINEAR_CLIENT_ID", source),
        "github_client_id": _env("POTPIE_GITHUB_CLIENT_ID", source),
    }


def build_info_values(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, str]:
    """Build identity for ``potpie/runtime/_build_info.py``.

    ``GIT_SHA`` comes from the build environment, else from the git checkout
    being built. ``DIRTY`` is ``"true"``/``"false"`` when known and ``""``
    otherwise: it describes the tree the rev was built from, so an explicit
    ``POTPIE_BUILD_DIRTY`` wins, and a checkout only answers for the rev it
    has checked out. Like the ``.env`` lookup, the checkout is consulted for
    an explicit ``environ`` only when ``source_root`` is given too.
    """
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    checkout_rev, checkout_dirty = (
        checkout_identity(source_root or _SOURCE_ROOT)
        if environ is None or source_root is not None
        else ("", "")
    )
    explicit_rev = _env("POTPIE_BUILD_GIT_SHA", source) or _env("GITHUB_SHA", source)
    rev = explicit_rev or checkout_rev
    dirty = _flag_value(_env("POTPIE_BUILD_DIRTY", source))
    if not dirty and rev and rev == checkout_rev:
        dirty = checkout_dirty
    return {
        "GIT_SHA": rev,
        "DIRTY": dirty,
        "BUILD_TIME": _env("POTPIE_BUILD_TIME", source) or _utc_now(),
    }


def checkout_identity(root: Path) -> tuple[str, str]:
    """``(rev, dirty)`` of the git checkout rooted at ``root``, else ``("", "")``.

    Only a checkout whose top level *is* ``root`` counts: an sdist unpacked
    inside some other repository must not stamp that repository's HEAD.

    ``dirty`` means modified *tracked* files (the ``git describe --dirty``
    definition), not untracked ones. Installers that build from a clone drop
    their own untracked markers into it (uv writes ``.ok``), and a clean rev
    must still stamp as clean there.
    """
    identity = _git(root, "rev-parse", "--show-toplevel", "HEAD")
    lines = identity.splitlines() if identity else []
    if len(lines) != 2:
        return "", ""
    toplevel, rev = (line.strip() for line in lines)
    try:
        same_root = Path(toplevel).resolve() == root.resolve()
    except OSError:
        same_root = False
    if not same_root or not rev:
        return "", ""
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return rev, ""
    return rev, "true" if status else "false"


def should_validate_distribution_defaults(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> bool:
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    return _env("POTPIE_VALIDATE_DISTRIBUTION_DEFAULTS", source).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate_distribution_defaults(values: Mapping[str, str]) -> None:
    missing = [
        name for name in REQUIRED_DISTRIBUTION_DEFAULTS if not _clean(values.get(name))
    ]
    if missing:
        raise RuntimeError(
            "Missing required distribution defaults: " + ", ".join(missing)
        )


def write_python_mapping(path: Path, name: str, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER
        + f"{name} = {{\n"
        + "".join(f"    {key!r}: {value!r},\n" for key, value in values.items())
        + "}\n",
        encoding="utf-8",
    )


def write_python_constants(path: Path, values: Mapping[str, str]) -> None:
    path.write_text(
        HEADER + "".join(f"{name} = {value!r}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def prefer_existing_distribution_default_values(
    path: Path,
    values: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    """Keep generated sdist distribution defaults unless their env input is set."""
    existing = _read_python_mapping(path, "DISTRIBUTION_DEFAULTS")
    if not existing:
        return dict(values)
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)
    merged = dict(values)
    for name in values:
        input_name = DISTRIBUTION_DEFAULT_INPUT_NAMES_BY_FIELD[name]
        if name in existing and not _env(input_name, source):
            merged[name] = existing[name]
    return merged


def prefer_existing_build_info_values(
    path: Path,
    values: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> dict[str, str]:
    """Keep generated sdist build metadata unless its field input is set."""
    existing = _read_python_constants(path)
    if not existing:
        return dict(values)
    source = _merged_build_environ(environ, dotenv_start=dotenv_start)

    def has_input(name: str) -> bool:
        return any(
            _env(input_name, source)
            for input_name in BUILD_INFO_INPUT_NAMES_BY_FIELD[name]
        )

    merged = dict(values)
    for name in values:
        if name != "DIRTY" and name in existing and not has_input(name):
            merged[name] = existing[name]
    # The dirty flag describes the tree its rev was built from, so it is kept
    # only together with a kept rev; a stamp that predates the flag says "".
    if (
        "DIRTY" in values
        and not has_input("DIRTY")
        and "GIT_SHA" in existing
        and not has_input("GIT_SHA")
    ):
        merged["DIRTY"] = existing.get("DIRTY", "")
    return merged


def _env(name: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return _clean(source.get(name))


def _env_or_default(
    name: str,
    default: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    return _env(name, environ) or default


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _flag_value(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in _TRUE_FLAGS:
        return "true"
    if lowered in _FALSE_FLAGS:
        return "false"
    return ""


def _git(root: Path, *args: str) -> str | None:
    """``git -C root *args`` stdout, or ``None`` when git is absent or refuses.

    Output goes to a temp file rather than a pipe: on Windows a git grandchild
    that outlives the timeout keeps an inherited pipe open, and the reader
    join would never return.
    """
    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with tempfile.TemporaryFile() as out:
            completed = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(root), *args],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_S,
                **kwargs,
            )
            if completed.returncode != 0:
                return None
            out.seek(0)
            return out.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _utc_now() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _merged_build_environ(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_start: Path | None = None,
) -> Mapping[str, str]:
    source = os.environ if environ is None else environ
    if dotenv_start is None and environ is not None:
        return source

    dotenv = _read_nearest_dotenv(dotenv_start or _DOTENV_SEARCH_START)
    if not dotenv:
        return source

    merged = dict(dotenv)
    merged.update(source)
    return merged


def _read_nearest_dotenv(start: Path) -> dict[str, str]:
    path = _find_nearest_dotenv(start)
    if path is None:
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def _find_nearest_dotenv(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for ancestor in (cur, *cur.parents):
        candidate = ancestor / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _read_python_mapping(path: Path, mapping_name: str) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != mapping_name:
            continue
        try:
            raw = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): _clean(value) for key, value in raw.items()}
    return {}


def _read_python_constants(path: Path) -> dict[str, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    values: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            values[target.id] = node.value.value
    return values
