from __future__ import annotations

from types import SimpleNamespace

from potpie_context_engine.adapters.outbound.install.local_installer import LocalInstaller


def test_installer_rejects_missing_executable(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert LocalInstaller().is_installed() is False


def test_installer_requires_workflow_commands(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/tmp/potpie")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="│ doctor help\n│ status help\n",
            stderr="",
        ),
    )
    assert LocalInstaller().is_installed() is False


def test_installer_accepts_usable_cli(monkeypatch) -> None:
    from potpie_context_engine.adapters.outbound.install.cli_probe import (
        REQUIRED_COMMAND_GROUPS,
    )

    monkeypatch.setattr("shutil.which", lambda _name: "/tmp/potpie")
    def help_for(argv, **_kwargs):
        group = tuple(argv[1:-1])
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                f"│ {command} help" for command in REQUIRED_COMMAND_GROUPS[group]
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "subprocess.run",
        help_for,
    )
    assert LocalInstaller().is_installed() is True
