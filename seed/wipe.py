from __future__ import annotations

import app.core.models  # noqa: F401 — register all models before ORM use
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.conversations.message.message_model import Message
from app.modules.intelligence.agents.custom_agents.custom_agent_model import (
    CustomAgent,
    CustomAgentShare,
)
from app.modules.integrations.integration_model import Integration
from app.modules.media.media_model import MessageAttachment
from app.modules.projects.projects_model import Project
from app.modules.search.search_models import SearchIndex
from app.modules.tasks.task_model import Task
from app.modules.users.user_model import User
from app.modules.auth.auth_provider_model import UserAuthProvider
from app.modules.users.user_preferences_model import UserPreferences
from seed.state import SeedState


def _uid_from_email(db: Session, email: str) -> str | None:
    u = db.query(User).filter(User.email == email).first()
    return u.uid if u else None


def _delete_workflow_tables(db: Session, uid: str) -> None:
    """Best-effort delete rows in potpie-workflows tables for this user."""
    engine = db.get_bind()
    from sqlalchemy import inspect as sql_inspect

    insp = sql_inspect(engine)

    def has_table(name: str) -> bool:
        return insp.has_table(name)

    if not has_table("workflows"):
        return

    # Delete executions for workflows owned by user (child tables cascade)
    if has_table("workflow_executions"):
        db.execute(
            text(
                """
                DELETE FROM workflow_executions
                WHERE wf_id IN (SELECT id FROM workflows WHERE created_by = :uid)
                """
            ),
            {"uid": uid},
        )

    if has_table("workflow_graphs"):
        db.execute(
            text(
                """
                DELETE FROM workflow_graphs
                WHERE workflow_id IN (SELECT id FROM workflows WHERE created_by = :uid)
                """
            ),
            {"uid": uid},
        )

    db.execute(text("DELETE FROM workflows WHERE created_by = :uid"), {"uid": uid})

    if has_table("trigger_hashes"):
        db.execute(text("DELETE FROM trigger_hashes WHERE user_id = :uid"), {"uid": uid})


def run_wipe(
    *,
    email: str | None = None,
    uid: str | None = None,
    profile: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    keep_user: bool = False,
) -> dict[str, str | bool]:
    """Delete seed data for the user identified by email or uid.

    When ``keep_user`` is True, the ``users`` row and related login rows
    (``user_preferences``, ``user_auth_providers``) are kept so the account
    can be re-seeded after ``apply --force``.
    """
    from seed.env import load_seed_env

    load_seed_env()

    if not email and not uid:
        raise ValueError("Provide --email or --uid")

    db = SessionLocal()
    try:
        if uid:
            target_uid = uid
        else:
            assert email is not None
            target_uid = _uid_from_email(db, email)
            if not target_uid:
                if profile:
                    SeedState().remove_file(profile)
                return {"deleted": False, "reason": "user not found"}

        if not dry_run and not yes:
            raise SystemExit("Refusing to wipe without --yes (or use --dry-run)")

        if dry_run:
            return {"deleted": False, "dry_run": True, "uid": target_uid}

        # Messages and attachments
        conv_ids = [
            r[0]
            for r in db.query(Conversation.id)
            .filter(Conversation.user_id == target_uid)
            .all()
        ]
        if conv_ids:
            msg_ids = [
                r[0]
                for r in db.query(Message.id)
                .filter(Message.conversation_id.in_(conv_ids))
                .all()
            ]
            if msg_ids:
                db.query(MessageAttachment).filter(
                    MessageAttachment.message_id.in_(msg_ids)
                ).delete(synchronize_session=False)
            db.query(Message).filter(Message.conversation_id.in_(conv_ids)).delete(
                synchronize_session=False
            )
        db.query(Conversation).filter(Conversation.user_id == target_uid).delete(
            synchronize_session=False
        )

        db.query(CustomAgentShare).filter(
            CustomAgentShare.shared_with_user_id == target_uid
        ).delete(synchronize_session=False)
        agent_ids = [
            r[0]
            for r in db.query(CustomAgent.id)
            .filter(CustomAgent.user_id == target_uid)
            .all()
        ]
        if agent_ids:
            db.query(CustomAgentShare).filter(
                CustomAgentShare.agent_id.in_(agent_ids)
            ).delete(synchronize_session=False)
        db.query(CustomAgent).filter(CustomAgent.user_id == target_uid).delete(
            synchronize_session=False
        )

        proj_ids = [
            p[0]
            for p in db.query(Project.id).filter(Project.user_id == target_uid).all()
        ]
        if proj_ids:
            db.query(Task).filter(Task.project_id.in_(proj_ids)).delete(
                synchronize_session=False
            )
            db.query(SearchIndex).filter(SearchIndex.project_id.in_(proj_ids)).delete(
                synchronize_session=False
            )
        db.query(Project).filter(Project.user_id == target_uid).delete(
            synchronize_session=False
        )

        _delete_workflow_tables(db, target_uid)

        db.query(Integration).filter(Integration.created_by == target_uid).delete(
            synchronize_session=False
        )
        if not keep_user:
            db.query(UserPreferences).filter(
                UserPreferences.user_id == target_uid
            ).delete(synchronize_session=False)
            db.query(UserAuthProvider).filter(
                UserAuthProvider.user_id == target_uid
            ).delete(synchronize_session=False)
            db.query(User).filter(User.uid == target_uid).delete(
                synchronize_session=False
            )

        db.commit()

        if profile:
            SeedState().remove_file(profile)

        return {"deleted": True, "uid": target_uid}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
