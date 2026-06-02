"""Potpie repo-local skill lifecycle service."""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters.inbound.cli.skill_catalog import (
    CANONICAL_SKILLS_DIR,
    LOCK_PATH,
    SCHEMA_VERSION,
    SkillCatalogError,
    canonical_skills_dir,
    copy_bundled_skill_atomic,
    discover_installed_skills,
    discover_bundled_skills,
    hash_path_dir,
    relative_to_root,
    skill_dir,
    validate_skill_id,
)
from adapters.inbound.cli.skill_lock import (
    read_lock,
    remove_lock_entry,
    upsert_lock_entry,
    write_lock,
)
from adapters.inbound.cli.skill_targets import (
    InvalidAgentError,
    llm_guidance,
    normalize_agent,
    recommended_skill_ids,
)


@dataclass
class SkillManagerError(Exception):
    code: str
    message: str
    recommended_command: str | None = None
    detail: Any = None

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.recommended_command:
            error["recommendedCommand"] = self.recommended_command
        if self.detail is not None:
            error["detail"] = self.detail
        return {"ok": False, "schemaVersion": SCHEMA_VERSION, "error": error}


class SkillManager:
    """Service object used by CLI commands and context-status advisory."""

    def __init__(self, root: str | Path = ".", *, agent: str = "default") -> None:
        self.root = Path(root).expanduser().resolve()
        try:
            self.agent = normalize_agent(agent)
        except InvalidAgentError as exc:
            raise SkillManagerError(
                "INVALID_AGENT",
                str(exc),
                detail={"validAgents": ["default", "codex", "claude", "cursor"]},
            ) from exc

    def status(self) -> dict[str, Any]:
        available_entries, catalog_diagnostics = discover_bundled_skills()
        available = [entry.to_catalog_dict() for entry in available_entries]
        catalog = {entry.id: entry for entry in available_entries}
        installed_rows, installed_diagnostics = discover_installed_skills(self.root)
        installed_by_id = {row["id"]: row for row in installed_rows}
        lock, lock_diagnostic = read_lock(self.root)
        lock_skills = lock.get("skills", {})

        installed: list[dict[str, Any]] = []
        outdated: list[dict[str, Any]] = []
        locally_modified: list[dict[str, Any]] = []
        for row in sorted(installed_rows, key=lambda item: str(item["id"])):
            skill_id = str(row["id"])
            catalog_entry = catalog.get(skill_id)
            lock_entry = lock_skills.get(skill_id)
            lock_payload = {"present": isinstance(lock_entry, dict)}
            if isinstance(lock_entry, dict):
                lock_payload.update(
                    {
                        "sourceType": lock_entry.get("sourceType"),
                        "templateHash": lock_entry.get("templateHash"),
                        "installedHash": lock_entry.get("installedHash"),
                    }
                )
            output_row = {**row, "status": "installed", "lock": lock_payload}
            if catalog_entry:
                output_row["templateHash"] = catalog_entry.template_hash
            installed.append(output_row)

            installed_hash = str(row["installedHash"])
            template_hash = catalog_entry.template_hash if catalog_entry else None
            if isinstance(lock_entry, dict) and installed_hash != lock_entry.get(
                "installedHash"
            ):
                locally_modified.append(
                    {
                        "id": skill_id,
                        "installedHash": installed_hash,
                        "lockedHash": lock_entry.get("installedHash"),
                    }
                )
            if catalog_entry and isinstance(lock_entry, dict):
                if catalog_entry.template_hash != lock_entry.get("templateHash"):
                    outdated.append(
                        {
                            "id": skill_id,
                            "installedHash": installed_hash,
                            "templateHash": catalog_entry.template_hash,
                            "lockedTemplateHash": lock_entry.get("templateHash"),
                        }
                    )
            elif catalog_entry and template_hash != installed_hash:
                outdated.append(
                    {
                        "id": skill_id,
                        "installedHash": installed_hash,
                        "templateHash": template_hash,
                        "lockedTemplateHash": None,
                    }
                )

        recommended = [
            skill_id
            for skill_id in recommended_skill_ids(self.agent)
            if skill_id in catalog
        ]
        missing = [
            skill_id for skill_id in recommended if skill_id not in installed_by_id
        ]
        diagnostics = [*catalog_diagnostics, *installed_diagnostics]
        if lock_diagnostic:
            diagnostics.append(lock_diagnostic)
        next_actions = self._next_actions(missing, outdated, locally_modified)
        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "canonicalDir": CANONICAL_SKILLS_DIR.as_posix(),
            "lockPath": LOCK_PATH.as_posix(),
            "available": available,
            "installed": installed,
            "recommended": recommended,
            "missing": missing,
            "outdated": outdated,
            "locallyModified": locally_modified,
            "diagnostics": diagnostics,
            "next_actions": next_actions,
            "nextActions": next_actions,
            "llmGuidance": llm_guidance(self.agent),
        }

    def list_skills(self, *, mode: str) -> dict[str, Any]:
        status = self.status()
        if mode == "available":
            skills = status["available"]
        elif mode == "installed":
            skills = status["installed"]
        elif mode == "recommended":
            available = {row["id"]: row for row in status["available"]}
            skills = [
                available[skill_id]
                for skill_id in status["recommended"]
                if skill_id in available
            ]
        else:
            raise SkillManagerError("INVALID_ARGUMENTS", f"Unknown list mode {mode!r}")
        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "mode": mode,
            "skills": skills,
        }

    def install(
        self, skill_id: str | None = None, *, yes: bool = False, force: bool = False
    ) -> dict[str, Any]:
        status = self.status()
        catalog = {row["id"]: row for row in status["available"]}
        targets = [skill_id] if skill_id else list(status["recommended"])
        if not targets:
            targets = list(catalog)
        results = []
        lock, lock_error = read_lock(self.root)
        if lock_error:
            raise SkillManagerError(
                "INVALID_LOCKFILE",
                lock_error["message"],
                recommended_command="Fix or remove .agents/skills-lock.json, then retry.",
            )
        for target_id in targets:
            safe_id = self._known_skill_or_error(str(target_id), catalog)
            results.append(
                self._install_one(safe_id, catalog[safe_id], lock, yes=yes, force=force)
            )
        write_lock(self.root, lock)
        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "installed": results,
        }

    def update(
        self, skill_id: str | None = None, *, all_: bool = False, yes: bool = False
    ) -> dict[str, Any]:
        if skill_id and all_:
            raise SkillManagerError(
                "INVALID_ARGUMENTS", "Pass either <skill-id> or --all, not both."
            )
        status = self.status()
        catalog = {row["id"]: row for row in status["available"]}
        installed_ids = {row["id"] for row in status["installed"]}
        modified_ids = {row["id"] for row in status["locallyModified"]}
        if skill_id:
            safe_id = validate_skill_id(skill_id)
            if safe_id not in installed_ids:
                raise SkillManagerError("NOT_INSTALLED", f"{safe_id} is not installed.")
            if safe_id not in catalog:
                raise SkillManagerError("NOT_IN_CATALOG", f"{safe_id} is not bundled.")
            targets = [safe_id]
        elif all_:
            targets = sorted(sid for sid in installed_ids if sid in catalog)
        else:
            targets = [row["id"] for row in status["outdated"]]

        if not targets:
            return {
                "ok": True,
                "schemaVersion": SCHEMA_VERSION,
                "root": str(self.root),
                "agent": self.agent,
                "updated": [],
                "skipped": [],
            }
        if not yes:
            raise SkillManagerError(
                "NEEDS_YES",
                "Refusing to update skills without --yes.",
                recommended_command=(
                    f"potpie skills update --path {self._quoted_root()} --yes"
                ),
            )

        lock, lock_error = read_lock(self.root)
        if lock_error:
            raise SkillManagerError("INVALID_LOCKFILE", lock_error["message"])
        updated: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for target_id in targets:
            if target_id in modified_ids:
                item = {
                    "id": target_id,
                    "status": "skipped",
                    "reason": "locally_modified",
                    "recommendedCommand": (
                        f"potpie skills install {target_id} "
                        f"--path {self._quoted_root()} --yes --force"
                    ),
                }
                skipped.append(item)
                if skill_id:
                    raise SkillManagerError(
                        "LOCALLY_MODIFIED_REFUSED",
                        f"{target_id} has local modifications.",
                        recommended_command=item["recommendedCommand"],
                    )
                continue
            updated.append(
                self._install_one(
                    target_id, catalog[target_id], lock, yes=True, force=False
                )
            )
        write_lock(self.root, lock)
        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "updated": updated,
            "skipped": skipped,
        }

    def remove(self, skill_id: str, *, yes: bool = False) -> dict[str, Any]:
        safe_id = validate_skill_id(skill_id)
        target = skill_dir(self.root, safe_id)
        if not target.exists():
            raise SkillManagerError("NOT_INSTALLED", f"{safe_id} is not installed.")
        lock, lock_error = read_lock(self.root)
        if lock_error:
            raise SkillManagerError("INVALID_LOCKFILE", lock_error["message"])
        if safe_id not in lock.get("skills", {}):
            raise SkillManagerError(
                "UNOWNED_SKILL_REFUSED",
                f"{safe_id} is not owned by Potpie lockfile.",
            )
        if not yes:
            raise SkillManagerError(
                "NEEDS_YES",
                "Refusing to remove a skill without --yes.",
                recommended_command=(
                    f"potpie skills remove {safe_id} "
                    f"--path {self._quoted_root()} --yes"
                ),
            )
        shutil.rmtree(target)
        remove_lock_entry(lock, safe_id)
        write_lock(self.root, lock)
        return {
            "ok": True,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "removed": [{"id": safe_id, "status": "removed"}],
        }

    def doctor(self) -> dict[str, Any]:
        status = self.status()
        lock, lock_error = read_lock(self.root)
        installed_ids = {row["id"] for row in status["installed"]}
        catalog_ids = {row["id"] for row in status["available"]}
        lock_skills = lock.get("skills", {})
        diagnostics = list(status["diagnostics"])
        for skill_id in sorted(installed_ids - catalog_ids):
            diagnostics.append(
                {
                    "code": "UNKNOWN_INSTALLED_SKILL",
                    "skillId": skill_id,
                    "message": f"{skill_id} is installed but not in the bundled catalog.",
                }
            )
        for skill_id in sorted(set(lock_skills) - installed_ids):
            diagnostics.append(
                {
                    "code": "LOCK_ENTRY_WITHOUT_INSTALL",
                    "skillId": skill_id,
                    "message": f"{skill_id} has a lock entry but no installed directory.",
                }
            )
        for skill_id in sorted(installed_ids - set(lock_skills)):
            diagnostics.append(
                {
                    "code": "INSTALLED_WITHOUT_LOCK",
                    "skillId": skill_id,
                    "message": f"{skill_id} is installed but not owned by the lockfile.",
                }
            )
        if lock_error:
            diagnostics.append(lock_error)
        ok = not diagnostics
        return {
            "ok": ok,
            "schemaVersion": SCHEMA_VERSION,
            "root": str(self.root),
            "agent": self.agent,
            "canonicalDir": CANONICAL_SKILLS_DIR.as_posix(),
            "lockPath": LOCK_PATH.as_posix(),
            "checks": {
                "catalogReadable": bool(status["available"]),
                "canonicalDirExists": canonical_skills_dir(self.root).exists(),
                "lockfileValid": lock_error is None,
            },
            "diagnostics": diagnostics,
            "recommendedCommand": None
            if ok
            else f"potpie skills status --path {self._quoted_root()}",
        }

    def context_status_advisory(self) -> dict[str, Any]:
        status = self.status()
        return {
            "agent": status["agent"],
            "recommended": status["recommended"],
            "installed": [
                {
                    "id": row["id"],
                    "path": row["installedPath"],
                    "installedHash": row["installedHash"],
                }
                for row in status["installed"]
            ],
            "missing": status["missing"],
            "outdated": status["outdated"],
            "locallyModified": status["locallyModified"],
            "nextActions": status["nextActions"],
            "doNotEditInstalledSkillsDirectly": True,
        }

    def _install_one(
        self,
        skill_id: str,
        catalog_entry: dict[str, Any],
        lock: dict[str, Any],
        *,
        yes: bool,
        force: bool,
    ) -> dict[str, Any]:
        target = skill_dir(self.root, skill_id)
        existing_hash = hash_path_dir(target) if target.exists() else None
        lock_entry = lock.get("skills", {}).get(skill_id)
        if existing_hash == catalog_entry["templateHash"]:
            upsert_lock_entry(
                lock,
                skill_id=skill_id,
                source=catalog_entry["templatePath"].removesuffix("/SKILL.md"),
                skill_path=relative_to_root(target / "SKILL.md", self.root),
                template_hash=catalog_entry["templateHash"],
                installed_hash=existing_hash,
            )
            return {
                "id": skill_id,
                "status": "unchanged",
                "installedHash": existing_hash,
            }
        if (
            target.exists()
            and isinstance(lock_entry, dict)
            and existing_hash != lock_entry.get("installedHash")
            and not force
        ):
            raise SkillManagerError(
                "LOCALLY_MODIFIED_REFUSED",
                f"{skill_id} has local modifications.",
                recommended_command=(
                    f"potpie skills install {skill_id} "
                    f"--path {self._quoted_root()} --yes --force"
                ),
            )
        if target.exists() and not yes:
            raise SkillManagerError(
                "NEEDS_YES",
                "Refusing to overwrite existing skill without --yes.",
                recommended_command=(
                    f"potpie skills install {skill_id} "
                    f"--path {self._quoted_root()} --yes"
                ),
            )
        try:
            installed_hash = copy_bundled_skill_atomic(skill_id, self.root)
        except SkillCatalogError as exc:
            raise SkillManagerError("UNKNOWN_SKILL", str(exc)) from exc
        upsert_lock_entry(
            lock,
            skill_id=skill_id,
            source=catalog_entry["templatePath"].removesuffix("/SKILL.md"),
            skill_path=relative_to_root(target / "SKILL.md", self.root),
            template_hash=catalog_entry["templateHash"],
            installed_hash=installed_hash,
        )
        return {
            "id": skill_id,
            "status": "installed" if existing_hash is None else "updated",
            "installedPath": relative_to_root(target / "SKILL.md", self.root),
            "installedHash": installed_hash,
        }

    def _known_skill_or_error(
        self, skill_id: str, catalog: dict[str, dict[str, Any]]
    ) -> str:
        try:
            safe_id = validate_skill_id(skill_id)
        except SkillCatalogError as exc:
            raise SkillManagerError("INVALID_SKILL_ID", str(exc)) from exc
        if safe_id not in catalog:
            raise SkillManagerError(
                "UNKNOWN_SKILL",
                f"{safe_id} is not in the bundled Potpie skill catalog.",
                recommended_command="potpie skills list --available",
            )
        return safe_id

    def _next_actions(
        self,
        missing: list[str],
        outdated: list[dict[str, Any]],
        locally_modified: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        actions: list[dict[str, str]] = []
        for skill_id in missing:
            command = (
                f"potpie skills install {skill_id} "
                f"--path {self._quoted_root()} --yes"
            )
            actions.append(
                {
                    "action": "install",
                    "skillId": skill_id,
                    "reason": "missing",
                    "command": command,
                    "recommendedCommand": command,
                }
            )
        for row in outdated:
            skill_id = str(row["id"])
            command = (
                f"potpie skills update {skill_id} "
                f"--path {self._quoted_root()} --yes"
            )
            actions.append(
                {
                    "action": "update",
                    "skillId": skill_id,
                    "reason": "outdated",
                    "command": command,
                    "recommendedCommand": command,
                }
            )
        for row in locally_modified:
            skill_id = str(row["id"])
            command = (
                f"potpie skills install {skill_id} "
                f"--path {self._quoted_root()} --yes --force"
            )
            actions.append(
                {
                    "action": "force-install",
                    "skillId": skill_id,
                    "reason": "locally_modified",
                    "command": command,
                    "recommendedCommand": command,
                }
            )
        return actions

    def _quoted_root(self) -> str:
        return shlex.quote(str(self.root))
