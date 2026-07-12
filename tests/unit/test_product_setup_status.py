"""Root setup sequencing and flat product status scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from potpie.auth.services import AccountIdentity
from potpie.setup import ProductSetupService, ProductStatusService, SetupPlan
from potpie.skills import SkillInfo, SkillOperationResult, SkillStatus
from potpie_context_engine.contracts import (
    EngineStatusReport,
    PotListResult,
    ProvisionPlan,
    ProvisionReport,
    ProvisionStep,
    SourceListResult,
)
from potpie_context_engine.domain.ports.services.pot_management import (
    PotInfo,
    SourceInfo,
)

pytestmark = pytest.mark.unit


class _Provision:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def inspect(self, request):
        return ProvisionPlan(
            backend="embedded",
            data_dir="/tmp/potpie",
            steps=(ProvisionStep("storage.prepare", True, "needed"),),
        )

    async def apply(self, request):
        if self.fail:
            raise RuntimeError("engine provision failed")
        return ProvisionReport(
            ok=True,
            backend="embedded",
            steps=(ProvisionStep("backend.provision", True, "done"),),
        )


class _Pots:
    def __init__(self, active: PotInfo | None = None) -> None:
        self.active = active

    async def info(self, request):
        return self.active

    async def create(self, request):
        self.active = PotInfo(
            pot_id="pot-1", name=request.name, created_at="now", active=True
        )
        return self.active

    async def list(self, request):
        items = (self.active,) if self.active else ()
        return PotListResult(items=items, count=len(items))


class _Sources:
    def __init__(self, items: tuple[SourceInfo, ...] = ()) -> None:
        self.items = items

    async def list(self, request):
        return SourceListResult(items=self.items, count=len(self.items))

    async def add(self, request):
        source = SourceInfo(
            source_id="source-1",
            kind=request.kind,
            location=request.location,
            name=request.name or request.location,
            status="ready",
        )
        self.items = (*self.items, source)
        return source


class _Context:
    def __init__(self, report: EngineStatusReport) -> None:
        self.report = report
        self.calls = 0

    async def status(self, request):
        self.calls += 1
        return self.report


class _Daemon:
    def __init__(self, *, up: bool, backend: str = "embedded") -> None:
        self.up = up
        self.backend = backend
        self.started = 0

    def status(self):
        return {"up": self.up, "backend": self.backend, "mode": "detached"}

    def start(self, *, backend: str):
        self.up = True
        self.backend = backend
        self.started += 1
        return {"pid": 123}

    def stop(self):
        self.up = False


class _Skills:
    def __init__(self, *, missing: bool = False, changed: bool = True) -> None:
        self.missing = missing
        self.changed = changed

    def install(self, *, agent: str, scope: str):
        return SkillOperationResult(
            agent=agent,
            operation="install",
            changed=("potpie-cli",) if self.changed else (),
        )

    def status(self, *, agent: str, scope: str):
        missing = (SkillInfo("potpie-cli", "Potpie CLI", "2"),) if self.missing else ()
        installed = (
            () if self.missing else (SkillInfo("potpie-cli", "Potpie CLI", "2"),)
        )
        return SkillStatus(agent=agent, installed=installed, missing=missing)


@dataclass
class _Config:
    values: dict[str, str]

    def set(self, key: str, value: str) -> None:
        self.values[key] = value


def _runtime(
    tmp_path: Path,
    *,
    mode: str = "in-process",
    daemon_up: bool = True,
    provision_fail: bool = False,
    active: PotInfo | None = None,
    sources: tuple[SourceInfo, ...] = (),
    missing_skills: bool = False,
    backend_ready: bool = True,
):
    engine_status = EngineStatusReport(
        schema_version="1",
        pot_id=active.pot_id if active else None,
        pot_name=active.name if active else None,
        backend="embedded",
        backend_ready=backend_ready,
        storage_ready=backend_ready,
        ingestion_ready=backend_ready,
        source_count=len(sources),
        degraded_reasons=() if backend_ready else ("backend degraded",),
    )
    return SimpleNamespace(
        settings=SimpleNamespace(
            data_dir=tmp_path, runtime_mode=mode, backend="embedded"
        ),
        engine=SimpleNamespace(
            provision=_Provision(fail=provision_fail),
            pots=_Pots(active),
            sources=_Sources(sources),
            context=_Context(engine_status),
        ),
        config=_Config({}),
        installer=SimpleNamespace(is_installed=lambda: True),
        auth=SimpleNamespace(
            whoami=lambda: AccountIdentity("anonymous", False, "none")
        ),
        daemon=_Daemon(up=daemon_up),
        skills=_Skills(missing=missing_skills),
    )


def test_fresh_setup_sequences_product_then_engine(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    setup = ProductSetupService(runtime)

    report = setup.run(
        SetupPlan(
            host_mode="in-process",
            repo=str(tmp_path),
            pot="project",
            agent="codex",
        )
    )

    assert report.ok
    assert [step.step for step in report.steps] == [
        "config",
        "installer",
        "auth",
        "daemon",
        "backend.provision",
        "pot.default",
        "source",
        "skills",
    ]
    assert runtime.config.values == {
        "runtime_mode": "in-process",
        "backend": "embedded",
    }
    assert runtime.engine.pots.active.name == "project"
    assert runtime.engine.sources.items[0].kind == "repo"


def test_already_configured_setup_is_idempotent(tmp_path: Path) -> None:
    active = PotInfo(pot_id="pot-1", name="project", created_at="now", active=True)
    location = str(tmp_path.resolve())
    source = SourceInfo(
        source_id="source-1",
        kind="repo",
        location=location,
        name=location,
        status="ready",
    )
    runtime = _runtime(tmp_path, active=active, sources=(source,))
    runtime.skills.changed = False

    report = ProductSetupService(runtime).run(
        SetupPlan(repo=str(tmp_path), pot="project", agent="codex")
    )

    assert report.ok
    states = {step.step: step.state for step in report.steps}
    assert states["pot.default"] == "skipped"
    assert states["source"] == "skipped"
    assert states["skills"] == "skipped"


def test_setup_starts_daemon_before_engine_provision(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, mode="daemon", daemon_up=False)

    report = ProductSetupService(runtime).run(
        SetupPlan(host_mode="daemon", repo=None, defer_skills=True)
    )

    assert report.ok
    assert runtime.daemon.started == 1
    assert [
        step.state for step in report.steps if step.step == "backend.provision"
    ] == ["done"]


def test_failed_engine_provision_is_a_hard_setup_failure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, provision_fail=True)

    report = ProductSetupService(runtime).run(SetupPlan(repo=None, defer_skills=True))

    assert not report.ok
    failed = next(step for step in report.steps if step.step == "backend.provision")
    assert failed.state == "failed"
    assert "engine provision failed" in (failed.detail or "")


def test_status_does_not_call_engine_when_daemon_is_unavailable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, mode="daemon", daemon_up=False)

    result = ProductStatusService(runtime).get()

    assert not result.ready
    assert result.daemon_state == "unavailable"
    assert result.recommended_next_action["command"] == "potpie daemon start"
    assert runtime.engine.context.calls == 0


def test_flat_status_enriches_degraded_engine_and_missing_skills(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    active = PotInfo(pot_id="pot-1", name="project", created_at="now", active=True)
    runtime = _runtime(
        tmp_path,
        active=active,
        missing_skills=True,
        backend_ready=False,
    )

    result = ProductStatusService(runtime).get()
    data = result.to_dict()

    assert set(data) == {
        "schema_version",
        "ready",
        "runtime_mode",
        "daemon_state",
        "pot_id",
        "pot_name",
        "backend",
        "backend_ready",
        "storage_ready",
        "ingestion_ready",
        "source_count",
        "last_ingestion_at",
        "skills_state",
        "setup_state",
        "issues",
        "recommended_next_action",
    }
    assert data["skills_state"] == "missing"
    assert data["recommended_next_action"]["command"] == "potpie graph backend doctor"
