"""Live setup wizard behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from adapters.inbound.cli.ui import setup_ux
from adapters.inbound.cli.ui import setup_wizard_ui
from adapters.inbound.cli.ui.setup_wizard_ui import SetupWizardUI
from domain.lifecycle import (
    DONE,
    NOT_IMPLEMENTED,
    PlannedSetupStep,
    SetupPlan,
    SetupPreview,
    SetupReport,
    StepResult,
)

pytestmark = pytest.mark.unit


def test_embedding_model_step_uses_onboarding_label() -> None:
    assert setup_ux.STEP_LABELS["embeddings.model"] == (
        "Setting up embedding model…",
        "Embedding model ready",
    )


class _FakeSetup:
    def __init__(self) -> None:
        self.observer = None

    def set_observer(self, observer) -> None:
        self.observer = observer

    def preview(self, plan: SetupPlan) -> SetupPreview:
        return SetupPreview(
            plan=plan,
            steps=(
                PlannedSetupStep(
                    "config",
                    hard=True,
                    owner="config",
                    action="write config",
                ),
                PlannedSetupStep(
                    "embeddings.model",
                    hard=False,
                    owner="embeddings",
                    action="prepare model",
                ),
                PlannedSetupStep(
                    "auth",
                    hard=False,
                    owner="auth",
                    action="init local auth",
                ),
            ),
        )

    def run(self, plan: SetupPlan) -> SetupReport:
        assert self.observer is not None
        self.observer.step_started(step="config", hard=True)
        self.observer.step_completed(
            result=StepResult("config", DONE, "config ready", hard=True),
            duration_ms=1,
        )
        self.observer.step_started(step="embeddings.model", hard=False)
        self.observer.step_completed(
            result=StepResult(
                "embeddings.model",
                DONE,
                "all-MiniLM-L6-v2 ready",
                hard=False,
            ),
            duration_ms=2,
        )
        self.observer.step_started(step="auth", hard=False)
        self.observer.step_completed(
            result=StepResult(
                "auth",
                NOT_IMPLEMENTED,
                "local auth/identity provisioning not implemented",
                hard=False,
            ),
            duration_ms=3,
        )
        return SetupReport(
            plan=plan,
            steps=(
                StepResult("config", DONE, "config ready", hard=True),
                StepResult(
                    "embeddings.model",
                    DONE,
                    "all-MiniLM-L6-v2 ready",
                    hard=False,
                ),
                StepResult(
                    "auth",
                    NOT_IMPLEMENTED,
                    "local auth/identity provisioning not implemented",
                    hard=False,
                ),
            ),
        )


class _FailingObserver:
    def step_started(self, *, step: str, hard: bool) -> None:
        del step, hard
        raise RuntimeError("observer failed")

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        del result, duration_ms
        raise RuntimeError("observer failed")


class _RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def step_started(self, *, step: str, hard: bool) -> None:
        del hard
        self.calls.append(("started", step))

    def step_completed(self, *, result: StepResult, duration_ms: int) -> None:
        del duration_ms
        self.calls.append(("completed", result.step))


def test_composite_setup_observer_isolates_callback_failures() -> None:
    recorder = _RecordingObserver()
    observer = setup_ux._CompositeSetupObserver(_FailingObserver(), recorder)

    observer.step_started(step="config", hard=True)
    observer.step_completed(
        result=StepResult("config", DONE, "config ready", hard=True),
        duration_ms=1,
    )

    assert recorder.calls == [("started", "config"), ("completed", "config")]


def test_live_setup_marks_embedding_step_running_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    original_start = SetupWizardUI.start_step
    original_complete = SetupWizardUI.complete_step

    def start(self, step_id: str) -> None:
        events.append(("start", step_id))
        original_start(self, step_id)

    def complete(self, step_id: str, **kwargs) -> None:
        events.append(("complete", step_id))
        original_complete(self, step_id, **kwargs)

    monkeypatch.setattr(SetupWizardUI, "start_step", start)
    monkeypatch.setattr(SetupWizardUI, "complete_step", complete)
    monkeypatch.setattr(setup_ux, "capture_wizard_event", lambda *_a, **_k: None)

    setup_ux.run_setup_live(
        _FakeSetup(),
        SetupPlan(repo=None, defer_default_pot=True, defer_skills=True),
        repo=Path("."),
        agent="claude",
        scan=False,
        use_rich=False,
    )

    assert events == [
        ("start", "config"),
        ("complete", "config"),
        ("start", "embeddings.model"),
        ("complete", "embeddings.model"),
    ]


def test_auth_step_is_hidden_from_live_setup() -> None:
    setup = _FakeSetup()

    report = setup_ux.run_setup_live(
        setup,
        SetupPlan(repo=None, defer_default_pot=True, defer_skills=True),
        repo=Path("."),
        agent="claude",
        scan=False,
        use_rich=False,
    )

    assert any(step.step == "auth" for step in report.steps)
    assert "auth" not in {
        step.step_id
        for step in setup.observer._observers[0]._wizard.steps  # noqa: SLF001
    }


def test_auth_not_implemented_is_hidden_from_replay(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "capture_wizard_event", lambda *_a, **_k: None)

    report = SetupReport(
        plan=SetupPlan(repo=None, defer_default_pot=True, defer_skills=True),
        steps=(
            StepResult("config", DONE, "config ready", hard=True),
            StepResult(
                "auth",
                NOT_IMPLEMENTED,
                "local auth/identity provisioning not implemented",
                hard=False,
            ),
        ),
    )

    setup_ux.render_setup_report(
        report,
        repo=Path("."),
        agent="claude",
        scan=False,
        use_rich=False,
    )

    captured = capsys.readouterr()
    assert "auth" not in captured.out.lower()
    assert "not implemented" not in captured.out.lower()


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, data: str) -> int:
        return len(data)

    def flush(self) -> None:
        return None


def test_live_ui_enabled_for_normal_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_FORCE_UI", raising=False)
    monkeypatch.delenv("POTPIE_NO_LIVE", raising=False)
    monkeypatch.delenv("POTPIE_PLAIN", raising=False)
    monkeypatch.setenv("CODEX_CI", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, "stderr", _FakeStream(tty=True))
    monkeypatch.setattr(sys, "stdout", _FakeStream(tty=True))

    assert setup_wizard_ui.rich_ui_enabled(as_json=False)
    assert setup_wizard_ui.live_ui_enabled(as_json=False)
    assert setup_ux.rich_enabled(as_json=False)

    monkeypatch.setenv("POTPIE_NO_LIVE", "1")

    assert setup_wizard_ui.rich_ui_enabled(as_json=False)
    assert not setup_wizard_ui.live_ui_enabled(as_json=False)
    assert not setup_ux.rich_enabled(as_json=False)


def test_setup_live_can_be_forced_when_streams_are_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POTPIE_PLAIN", raising=False)
    monkeypatch.delenv("POTPIE_NO_LIVE", raising=False)
    monkeypatch.setenv("POTPIE_FORCE_UI", "1")
    monkeypatch.setattr(sys, "stderr", _FakeStream(tty=False))
    monkeypatch.setattr(sys, "stdout", _FakeStream(tty=False))

    assert setup_wizard_ui.rich_ui_enabled(as_json=False)
    assert setup_wizard_ui.live_ui_enabled(as_json=False)


def test_setup_console_uses_stdout_when_it_is_only_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stderr", _FakeStream(tty=False))
    monkeypatch.setattr(sys, "stdout", _FakeStream(tty=True))

    assert not setup_wizard_ui._setup_console_uses_stderr()


def test_live_ui_can_be_disabled_on_normal_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "CI",
        "CODEX_CI",
        "POTPIE_FORCE_UI",
        "POTPIE_NO_LIVE",
        "POTPIE_PLAIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, "stderr", _FakeStream(tty=True))
    monkeypatch.setattr(sys, "stdout", _FakeStream(tty=True))

    assert setup_wizard_ui.rich_ui_enabled(as_json=False)
    assert setup_wizard_ui.live_ui_enabled(as_json=False)
    assert setup_ux.rich_enabled(as_json=False)

    monkeypatch.setenv("POTPIE_NO_LIVE", "1")

    assert not setup_wizard_ui.live_ui_enabled(as_json=False)
    assert not setup_ux.rich_enabled(as_json=False)


def test_interactive_onboarding_does_not_require_live_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "CI",
        "CODEX_CI",
        "POTPIE_FORCE_UI",
        "POTPIE_NO_LIVE",
        "POTPIE_PLAIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POTPIE_NO_LIVE", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, "stderr", _FakeStream(tty=True))
    monkeypatch.setattr(sys, "stdout", _FakeStream(tty=True))
    monkeypatch.setattr(sys, "stdin", _FakeStream(tty=True))

    assert not setup_ux.rich_enabled(as_json=False)
    assert setup_ux.interactive_onboarding_enabled(as_json=False)


def test_plain_setup_streams_embedding_step_before_completion(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_ux, "capture_wizard_event", lambda *_a, **_k: None)

    setup_ux.run_setup_plain(
        _FakeSetup(),
        SetupPlan(repo=None, defer_default_pot=True, defer_skills=True),
        repo=Path("."),
        agent="claude",
        scan=False,
    )

    err = capsys.readouterr().err
    start = err.index("› Setting up embedding model…")
    done = err.index("✓ Embedding model ready all-MiniLM-L6-v2 ready")
    assert start < done
