"""Root composition of engine facts and product readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from potpie_context_engine.contracts import EngineStatusRequest

from potpie.install.status import collect_cli_install_status
from potpie.runtime.sync_view import await_engine
from potpie.setup.contracts import ProductStatusResult


@dataclass(slots=True)
class ProductStatusService:
    runtime: Any

    def get(
        self, *, pot_id: str | None = None, harness: str = "claude"
    ) -> ProductStatusResult:
        daemon = self.runtime.daemon.status()
        daemon_up = bool(daemon.get("up"))
        daemon_state = "up" if daemon_up else "unavailable"
        issues: list[str] = []

        engine_report = None
        if self.runtime.settings.runtime_mode == "daemon" and not daemon_up:
            issues.append("The Potpie daemon is not reachable.")
        else:
            try:
                engine_report = await_engine(
                    self.runtime.engine.context.status(
                        EngineStatusRequest(pot_id=pot_id)
                    )
                )
            except Exception as exc:  # noqa: BLE001 - status must remain diagnostic.
                issues.append(str(exc))

        try:
            skills = self.runtime.skills.status(agent=harness, scope="global")
            if skills.missing:
                skills_state = "missing"
                issues.append(
                    "Missing Potpie skills: "
                    + ", ".join(skill.id for skill in skills.missing)
                )
            elif skills.outdated:
                skills_state = "outdated"
                issues.append(
                    "Outdated Potpie skills: "
                    + ", ".join(skill.id for skill in skills.outdated)
                )
            else:
                skills_state = "ready"
        except Exception as exc:  # noqa: BLE001
            skills_state = "unknown"
            issues.append(f"Skill status unavailable: {exc}")

        setup_state = (
            "configured"
            if (self.runtime.settings.data_dir / "config.json").exists()
            else "not_configured"
        )
        if setup_state != "configured":
            issues.append("Product setup has not completed.")

        if engine_report is not None:
            issues.extend(engine_report.degraded_reasons)
            backend = engine_report.backend
            backend_ready = engine_report.backend_ready
            storage_ready = engine_report.storage_ready
            ingestion_ready = engine_report.ingestion_ready
            pot_name = engine_report.pot_name
            resolved_pot_id = engine_report.pot_id
            source_count = engine_report.source_count
            last_ingestion_at = engine_report.last_ingestion_at
        else:
            backend = self.runtime.settings.backend
            backend_ready = False
            storage_ready = False
            ingestion_ready = False
            pot_name = None
            resolved_pot_id = pot_id
            source_count = 0
            last_ingestion_at = None

        recommended = self._recommended_action(
            daemon_up=daemon_up,
            setup_state=setup_state,
            skills_state=skills_state,
            backend_ready=backend_ready,
        )
        ready = bool(
            daemon_up
            and engine_report is not None
            and backend_ready
            and storage_ready
            and ingestion_ready
            and setup_state == "configured"
            and skills_state == "ready"
        )
        return ProductStatusResult(
            schema_version="1",
            ready=ready,
            runtime_mode=self.runtime.settings.runtime_mode,
            daemon_state=daemon_state,
            pot_id=resolved_pot_id,
            pot_name=pot_name,
            backend=backend,
            backend_ready=backend_ready,
            storage_ready=storage_ready,
            ingestion_ready=ingestion_ready,
            source_count=source_count,
            last_ingestion_at=last_ingestion_at,
            skills_state=skills_state,
            setup_state=setup_state,
            issues=tuple(dict.fromkeys(issue for issue in issues if issue)),
            recommended_next_action=recommended,
        )

    def doctor(self, *, harness: str = "claude") -> dict[str, Any]:
        status = self.get(harness=harness)
        return {
            **status.to_dict(),
            "cli_install": collect_cli_install_status(),
            "daemon": dict(self.runtime.daemon.status()),
        }

    def _recommended_action(
        self,
        *,
        daemon_up: bool,
        setup_state: str,
        skills_state: str,
        backend_ready: bool,
    ) -> dict[str, str] | None:
        if self.runtime.settings.runtime_mode == "daemon" and not daemon_up:
            return {
                "command": "potpie daemon start",
                "reason": "The configured runtime mode is daemon.",
            }
        if setup_state != "configured":
            return {"command": "potpie setup", "reason": "Setup is incomplete."}
        if not backend_ready:
            return {
                "command": "potpie graph backend doctor",
                "reason": "The active engine backend is degraded.",
            }
        if skills_state in {"missing", "outdated"}:
            return {
                "command": "potpie skills update --all",
                "reason": "Installed Potpie skills are incomplete or outdated.",
            }
        return None


__all__ = ["ProductStatusService"]
