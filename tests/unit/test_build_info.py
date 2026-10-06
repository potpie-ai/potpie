"""``potpie.build_info``, ``potpie --version``, and the build stamp feeding them."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import builtins
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import potpie.runtime
from potpie import build_info
from potpie.cli import main as cli_main

FIXTURE_REV = "a" * 40
OTHER_REV = "b" * 40

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_build_config_values() -> ModuleType:
    path = _SCRIPTS_DIR / "build_config_values.py"
    spec = importlib.util.spec_from_file_location("_test_build_info_values", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_config_values = _load_build_config_values()


# Held here: tests replace the module attribute, and the cache belongs to
# the real function.
_cached_build_stamp = build_info.build_stamp


@pytest.fixture(autouse=True)
def _fresh_stamp():
    _cached_build_stamp.cache_clear()
    yield
    _cached_build_stamp.cache_clear()


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("a", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "init")
    return root


# --- what the runtime reports ------------------------------------------------


def test_describe_reports_the_distribution_and_the_engine_separately() -> None:
    info = build_info.describe()

    assert info["name"] == "potpie"
    assert isinstance(info["version"], str) and info["version"]
    assert set(info["build"]) == {"rev", "dirty", "built_at"}
    assert info["engine"]["name"] == "potpie-context-engine"
    assert isinstance(info["engine"]["version"], str) and info["engine"]["version"]


def test_human_line_shows_the_short_rev_and_the_dirty_mark() -> None:
    base = {"name": "potpie", "version": "2.0.1"}
    clean = {**base, "build": {"rev": FIXTURE_REV, "dirty": False}}
    dirty = {**base, "build": {"rev": FIXTURE_REV, "dirty": True}}
    unknown = {**base, "build": {"rev": None, "dirty": None}}

    assert build_info.human_line(clean) == "potpie 2.0.1 (aaaaaaaaaa)"
    assert build_info.human_line(dirty) == "potpie 2.0.1 (aaaaaaaaaa, dirty)"
    assert build_info.human_line(unknown) == "potpie 2.0.1 (build rev unknown)"


def test_short_rev_tolerates_anything() -> None:
    assert build_info.short_rev(None) is None
    assert build_info.short_rev("") is None
    assert build_info.short_rev(12) is None
    assert build_info.short_rev("abcdef0123456789") == "abcdef0123"


def test_build_stamp_is_a_dict_even_when_nothing_is_known() -> None:
    assert isinstance(build_info.build_stamp(), dict)


def test_build_is_stale_only_when_both_revs_are_known_and_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": FIXTURE_REV})
    assert build_info.build_is_stale({"rev": FIXTURE_REV}) is False
    assert build_info.build_is_stale({"rev": OTHER_REV}) is True
    assert build_info.build_is_stale({"rev": None}) is None

    monkeypatch.setattr(build_info, "build_stamp", lambda: {})
    assert build_info.build_is_stale({"rev": OTHER_REV}) is None


def test_build_stamp_reads_the_module_the_build_hook_generates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated = tmp_path / "_build_info.py"
    build_config_values.write_python_constants(
        generated,
        {"GIT_SHA": FIXTURE_REV, "DIRTY": "true", "BUILD_TIME": "2026-06-28T00:00:00Z"},
    )
    module = ModuleType("potpie.runtime._build_info")
    exec(  # noqa: S102 - executes the fixture module this test just generated
        compile(generated.read_text(encoding="utf-8"), str(generated), "exec"),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, "potpie.runtime._build_info", module)
    monkeypatch.setattr(potpie.runtime, "_build_info", module, raising=False)

    stamp = build_info.build_stamp()

    assert stamp["rev"] == FIXTURE_REV
    assert stamp["dirty"] is True
    assert stamp["built_at"] == "2026-06-28T00:00:00Z"
    assert stamp["version"] == build_info.distribution_version()


def test_unknown_generated_fields_read_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("potpie.runtime._build_info")
    module.GIT_SHA = ""  # type: ignore[attr-defined]
    module.DIRTY = ""  # type: ignore[attr-defined]
    module.BUILD_TIME = "2026-06-28T00:00:00Z"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "potpie.runtime._build_info", module)
    monkeypatch.setattr(potpie.runtime, "_build_info", module, raising=False)

    stamp = build_info.build_stamp()

    assert stamp["rev"] is None
    assert stamp["dirty"] is None


def test_an_editable_checkout_reads_its_stamp_from_git(
    monkeypatch: pytest.MonkeyPatch, checkout: Path
) -> None:
    """No generated module (an editable install): ask the checkout instead."""
    monkeypatch.setitem(sys.modules, "potpie.runtime._build_info", None)
    monkeypatch.delattr(potpie.runtime, "_build_info", raising=False)
    monkeypatch.setattr(build_info, "_CHECKOUT_ROOT", checkout)

    stamp = build_info.build_stamp()

    assert stamp["rev"] == _git(checkout, "rev-parse", "HEAD")
    assert stamp["dirty"] is False
    assert stamp["built_at"] is None


def test_a_runtime_that_cannot_import_is_not_read_as_a_checkout(
    monkeypatch: pytest.MonkeyPatch, checkout: Path
) -> None:
    real_import = builtins.__import__

    def import_without_a_dependency(name: str, *args: object, **kwargs: object):
        if name == "potpie.runtime._build_info":
            raise ModuleNotFoundError("No module named 'aiohttp'", name="aiohttp")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "potpie.runtime._build_info", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_a_dependency)
    monkeypatch.setattr(build_info, "_CHECKOUT_ROOT", checkout)

    assert build_info.build_stamp() == {}


def test_checkout_stamp_counts_tracked_changes_only(checkout: Path) -> None:
    # Untracked files are not dirt: uv drops an untracked `.ok` marker into
    # every checkout it builds from, and a clean rev must read as clean there.
    (checkout / ".ok").write_text("", encoding="utf-8")
    assert build_info._checkout_stamp(checkout)["dirty"] is False

    (checkout / "tracked.txt").write_text("b", encoding="utf-8")
    assert build_info._checkout_stamp(checkout)["dirty"] is True


def test_checkout_stamp_ignores_an_enclosing_repository(
    checkout: Path, tmp_path: Path
) -> None:
    nested = checkout / "site-packages"
    nested.mkdir()

    assert build_info._checkout_stamp(nested) == {}
    assert build_info._checkout_stamp(tmp_path / "missing") == {}


# --- `potpie --version` --------------------------------------------------------


def test_json_version_reports_the_distribution_build_and_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        build_info,
        "build_stamp",
        lambda: {
            "rev": FIXTURE_REV,
            "dirty": False,
            "built_at": "2026-06-28T00:00:00Z",
        },
    )

    cli_main.run_cli(["--json", "--version"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "potpie"
    assert payload["version"]
    assert payload["build"] == {
        "rev": FIXTURE_REV,
        "dirty": False,
        "built_at": "2026-06-28T00:00:00Z",
    }
    assert payload["engine"]["name"] == "potpie-context-engine"
    assert payload["engine"]["version"]
    assert payload["python"] == platform.python_version()
    assert payload["executable"] == sys.executable


def test_human_version_leads_with_potpie_and_its_short_rev(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        build_info, "build_stamp", lambda: {"rev": FIXTURE_REV, "dirty": True}
    )

    cli_main.run_cli(["--version"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("potpie ")
    assert lines[0].endswith("(aaaaaaaaaa, dirty)")
    assert lines[1].startswith("potpie-context-engine ")
    assert lines[2] == f"python {platform.python_version()} ({sys.executable})"


# --- what the build hook stamps ----------------------------------------------


def test_checkout_identity_stamps_rev_and_tracked_dirt(checkout: Path) -> None:
    head = _git(checkout, "rev-parse", "HEAD")

    assert build_config_values.checkout_identity(checkout) == (head, "false")

    (checkout / ".ok").write_text("", encoding="utf-8")
    assert build_config_values.checkout_identity(checkout) == (head, "false")

    (checkout / "tracked.txt").write_text("b", encoding="utf-8")
    assert build_config_values.checkout_identity(checkout) == (head, "true")


def test_checkout_identity_outside_a_checkout_is_empty(
    checkout: Path, tmp_path: Path
) -> None:
    """An sdist unpacked inside another repository must not take its HEAD."""
    unpacked = checkout / "dist" / "potpie-2.0.1"
    unpacked.mkdir(parents=True)

    assert build_config_values.checkout_identity(unpacked) == ("", "")
    assert build_config_values.checkout_identity(tmp_path / "missing") == ("", "")


def test_build_info_values_fall_back_to_the_checkout(checkout: Path) -> None:
    head = _git(checkout, "rev-parse", "HEAD")
    environ = {"POTPIE_BUILD_TIME": "2026-06-28T00:00:00Z"}

    values = build_config_values.build_info_values(environ, source_root=checkout)

    assert values == {
        "GIT_SHA": head,
        "DIRTY": "false",
        "BUILD_TIME": "2026-06-28T00:00:00Z",
    }


def test_build_info_dirty_describes_only_the_checked_out_rev(checkout: Path) -> None:
    head = _git(checkout, "rev-parse", "HEAD")
    (checkout / "tracked.txt").write_text("b", encoding="utf-8")

    same = build_config_values.build_info_values(
        {"GITHUB_SHA": head}, source_root=checkout
    )
    other = build_config_values.build_info_values(
        {"GITHUB_SHA": FIXTURE_REV}, source_root=checkout
    )
    explicit = build_config_values.build_info_values(
        {"GITHUB_SHA": FIXTURE_REV, "POTPIE_BUILD_DIRTY": "0"}, source_root=checkout
    )

    assert (same["GIT_SHA"], same["DIRTY"]) == (head, "true")
    assert (other["GIT_SHA"], other["DIRTY"]) == (FIXTURE_REV, "")
    assert (explicit["GIT_SHA"], explicit["DIRTY"]) == (FIXTURE_REV, "false")


def test_an_sdist_stamp_keeps_its_rev_and_dirty_flag_together(tmp_path: Path) -> None:
    stamp = tmp_path / "_build_info.py"
    build_config_values.write_python_constants(
        stamp,
        {"GIT_SHA": FIXTURE_REV, "DIRTY": "true", "BUILD_TIME": "2026-06-27T00:00:00Z"},
    )
    fresh = {
        "GIT_SHA": OTHER_REV,
        "DIRTY": "false",
        "BUILD_TIME": "2026-06-28T00:00:00Z",
    }

    kept = build_config_values.prefer_existing_build_info_values(
        stamp, fresh, environ={}
    )
    overridden = build_config_values.prefer_existing_build_info_values(
        stamp, {**fresh, "DIRTY": ""}, environ={"GITHUB_SHA": OTHER_REV}
    )

    assert kept == {
        "GIT_SHA": FIXTURE_REV,
        "DIRTY": "true",
        "BUILD_TIME": "2026-06-27T00:00:00Z",
    }
    # A rev from the build environment does not inherit the sdist's flag.
    assert (overridden["GIT_SHA"], overridden["DIRTY"]) == (OTHER_REV, "")


def test_an_sdist_stamp_without_a_dirty_flag_reads_as_unknown(tmp_path: Path) -> None:
    stamp = tmp_path / "_build_info.py"
    build_config_values.write_python_constants(
        stamp, {"GIT_SHA": FIXTURE_REV, "BUILD_TIME": "2026-06-27T00:00:00Z"}
    )

    values = build_config_values.prefer_existing_build_info_values(
        stamp,
        {"GIT_SHA": OTHER_REV, "DIRTY": "false", "BUILD_TIME": "2026-06-28T00:00:00Z"},
        environ={},
    )

    assert (values["GIT_SHA"], values["DIRTY"]) == (FIXTURE_REV, "")
