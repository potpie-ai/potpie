from __future__ import annotations

from sqlalchemy.orm import Session

from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
from seed.schemas import CustomAgentsFile


def apply_agents(db: Session, data: CustomAgentsFile, user_id: str) -> list[str]:
    ids: list[str] = []
    for a in data.agents:
        existing = db.query(CustomAgent).filter(CustomAgent.id == a.id).first()
        tasks = a.tasks
        if isinstance(tasks, list):
            tasks = tasks or []
        row = {
            "id": a.id,
            "user_id": user_id,
            "role": a.role,
            "goal": a.goal,
            "backstory": a.backstory,
            "system_prompt": a.system_prompt,
            "tasks": tasks,
            "deployment_url": a.deployment_url,
            "deployment_status": a.deployment_status,
            "visibility": a.visibility,
        }
        if existing:
            for k, v in row.items():
                if k == "id":
                    continue
                setattr(existing, k, v)
        else:
            db.add(CustomAgent(**row))
        ids.append(a.id)
    db.commit()
    return ids
