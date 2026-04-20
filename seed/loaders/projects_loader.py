from __future__ import annotations

import base64

from sqlalchemy.orm import Session

from app.modules.projects.projects_model import Project
from seed.schemas import ProjectsFile


def apply_projects(db: Session, data: ProjectsFile, user_id: str) -> list[str]:
    ids: list[str] = []
    for p in data.projects:
        props = None
        if p.properties_base64:
            props = base64.b64decode(p.properties_base64)
        existing = db.query(Project).filter(Project.id == p.id).first()
        if existing:
            if existing.user_id != user_id:
                raise ValueError(
                    f"Project {p.id!r} belongs to another user ({existing.user_id})"
                )
            existing.repo_name = p.repo_name
            existing.branch_name = p.branch_name
            existing.status = p.status
            existing.commit_id = p.commit_id
            existing.repo_path = p.repo_path
            if props is not None:
                existing.properties = props
        else:
            proj = Project(
                id=p.id,
                repo_name=p.repo_name,
                branch_name=p.branch_name,
                user_id=user_id,
                repo_path=p.repo_path,
                commit_id=p.commit_id,
                status=p.status,
                properties=props,
            )
            db.add(proj)
        ids.append(p.id)
    db.commit()
    return ids
