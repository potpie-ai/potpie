from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.modules.users.user_model import User
from app.modules.projects.projects_model import Project
from datetime import datetime

# User CRUD operations
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.provider_username == username).first()

def create_user(db: Session, user: User):
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def update_user(db: Session, user_id: str, **kwargs):
    db.query(User).filter(User.uid == user_id).update(kwargs)
    db.commit()

def delete_user(db: Session, user_id: str):
    db.query(User).filter(User.uid == user_id).delete()
    db.commit()

# Project CRUD operations
def get_project_by_id(db: Session, project_id: int):
    return db.query(Project).filter(Project.id == project_id).first()

def get_projects_by_user_id(db: Session, user_id: str):
    return db.query(Project).filter(Project.user_id == user_id).all()

def create_project(db: Session, project: Project):
    project.created_at = datetime.utcnow()
    project.updated_at = datetime.utcnow()
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def update_project(db: Session, project_id: int, **kwargs):
    project = db.query(Project).filter(Project.id == project_id).first()

    if project is None:
        return None  # Project doesn't exist

    result = db.query(Project).filter(Project.id == project_id).update(kwargs)

    if result > 0:
        db.commit()
        return result

    return None

def delete_project(db: Session, project_id: int):
    db.query(Project).filter(Project.id == project_id).delete()
    db.commit()


def get_projects_by_repo_name(db: Session, repo_name: str, user_id: str, is_deleted: bool = False):
    try:
        projects = db.query(Project).filter(
            and_(
                Project.repo_name == repo_name,
                Project.user_id == user_id,
                Project.is_deleted == is_deleted
            )
        ).all()

        return projects
    except Exception as e:
        db.rollback()
        # Log the error
        print(f"Error fetching projects: {str(e)}")
        # You might want to raise a custom exception here instead of returning None
        return None


