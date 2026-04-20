from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import app.core.models  # noqa: F401 — register all models before Message/attachment mappers
from sqlalchemy.orm import Session

from app.modules.users.user_model import User
from seed.loaders import agents_loader, conversations_loader, projects_loader, user_loader
from seed.paths import data_dir
from seed.schemas import (
    ConversationsFile,
    CustomAgentsFile,
    Manifest,
    ProjectsFile,
    UserSeed,
)
from seed.state import SeedState


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_profile_manifest(profile: str) -> Manifest:
    base = data_dir(profile)
    mf = base / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"Missing manifest: {mf}")
    return Manifest.model_validate(_read_json(mf))


def require_existing_user_uid(db: Session, target_email: str) -> str:
    """Return ``users.uid`` for ``target_email``; seed only applies to existing accounts."""
    existing = db.query(User).filter(User.email == target_email).first()
    if not existing:
        raise ValueError(
            f"No user with email {target_email!r}. Create the account first (sign up); "
            "seed does not create users."
        )
    return existing.uid


def run_apply(
    profile: str,
    email: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply seed data for profile and target email. Returns summary dict."""
    from seed.env import load_seed_env

    load_seed_env()
    from app.core.database import SessionLocal

    base = data_dir(profile)
    if not base.is_dir():
        raise FileNotFoundError(f"Unknown profile directory: {base}")

    manifest = load_profile_manifest(profile)
    user_path = base / "user.json"
    if not user_path.exists():
        raise FileNotFoundError(f"Missing {user_path}")

    user_data = UserSeed.model_validate(_read_json(user_path))
    target_email = email.strip().lower()
    if user_data.email is not None and user_data.email.strip().lower() != target_email:
        raise ValueError(
            f"CLI --email {target_email!r} conflicts with user.json email {user_data.email!r}. "
            "Remove email from user.json to always use --email, or align them."
        )

    summary: dict[str, Any] = {
        "profile": profile,
        "email": target_email,
        "dry_run": dry_run,
    }

    db = SessionLocal()
    try:
        resolved_uid = require_existing_user_uid(db, target_email)
    finally:
        db.close()

    summary["uid"] = resolved_uid

    if dry_run:
        summary["would_apply"] = True
        return summary

    if force:
        from seed.wipe import run_wipe

        run_wipe(
            email=target_email,
            uid=None,
            profile=profile,
            dry_run=False,
            yes=True,
            keep_user=True,
        )
    else:
        prior = SeedState.load(profile)
        if prior and prior.get("email") == target_email and prior.get("uid"):
            raise SystemExit(
                f"Seed state exists for profile {profile!r} ({prior.get('uid')}). "
                "Use --force to wipe and re-apply."
            )

    db = SessionLocal()
    try:
        resolved_uid = require_existing_user_uid(db, target_email)
        user_loader.apply_user(
            db, user_data, target_email=target_email, resolved_uid=resolved_uid
        )

        state: dict[str, Any] = {
            "profile": profile,
            "email": target_email,
            "uid": resolved_uid,
            "project_ids": [],
            "custom_agent_ids": [],
            "conversation_ids": [],
        }

        for step in manifest.steps:
            if step == "user":
                continue
            if step == "projects":
                p = base / "projects.json"
                if not p.exists():
                    continue
                pf = ProjectsFile.model_validate(_read_json(p))
                state["project_ids"] = projects_loader.apply_projects(
                    db, pf, resolved_uid
                )
            elif step == "custom_agents":
                p = base / "custom_agents.json"
                if not p.exists():
                    continue
                af = CustomAgentsFile.model_validate(_read_json(p))
                state["custom_agent_ids"] = agents_loader.apply_agents(
                    db, af, resolved_uid
                )
            elif step == "conversations":
                p = base / "conversations.json"
                if not p.exists():
                    continue
                cf = ConversationsFile.model_validate(_read_json(p))
                state["conversation_ids"] = conversations_loader.apply_conversations(
                    db, cf, resolved_uid
                )
            else:
                raise ValueError(f"Unknown manifest step: {step!r}")

        SeedState(state).save(profile)
        summary["uid"] = resolved_uid
        summary["state"] = state
        return summary
    finally:
        db.close()


def validate_profile(profile: str) -> list[str]:
    """Validate JSON files for profile; return list of error strings (empty if ok)."""
    errors: list[str] = []
    try:
        manifest = load_profile_manifest(profile)
    except Exception as e:
        return [str(e)]

    base = data_dir(profile)
    try:
        UserSeed.model_validate(_read_json(base / "user.json"))
    except Exception as e:
        errors.append(f"user.json: {e}")

    optional = {
        "projects": ProjectsFile,
        "custom_agents": CustomAgentsFile,
        "conversations": ConversationsFile,
    }
    name_map = {
        "projects": "projects.json",
        "custom_agents": "custom_agents.json",
        "conversations": "conversations.json",
    }
    for step in manifest.steps:
        if step == "user" or step not in optional:
            continue
        path = base / name_map[step]
        if not path.exists():
            continue
        try:
            optional[step].model_validate(_read_json(path))
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    return errors
