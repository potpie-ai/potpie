import os
from app.core.database import SessionLocal
from app.modules.projects.projects_model import Project
from app.modules.users.user_model import User
from sqlalchemy.sql import func

class DummyDataSetup:
    def __init__(self):
        self.db = SessionLocal()

    def setup_dummy_user(self):
        try:
            # Check if the dummy user already exists
            user_exists = self.db.query(User).filter_by(uid=os.getenv("defaultUsername")).first()
            if not user_exists:
                # Create a dummy user
                user = User(
                    uid=os.getenv("defaultUsername"),
                    email=f"{os.getenv('defaultUsername')}@momentum.sh",
                    display_name="Dummy User",
                    email_verified=True,
                    created_at=func.now(),
                    last_login_at=func.now(),
                    provider_info={},
                    provider_username="self",
                )
                self.db.add(user)
                self.db.commit()
                print(f"Created dummy user with uid: {user.uid}")
            else:
                print("Dummy user already exists")
        finally:
            self.db.close()

    def setup_dummy_project(self):
        try:
            # Check if the dummy user exists
            dummy_user = self.db.query(User).filter_by(uid=os.getenv("defaultUsername")).first()
            if dummy_user:
                # Check if the dummy project already exists
                project_exists = self.db.query(Project).filter_by(directory="dummy_directory").first()
                if not project_exists:
                    # Create a dummy project
                    dummy_project = Project(
                        directory="dummy_directory",
                        is_default=True,
                        project_name="Dummy Project Created To Test AI Agent",
                        properties=b'{}',
                        repo_name="dummy_repo",
                        branch_name="main",
                        user_id=dummy_user.uid,
                        created_at=func.now(),
                        commit_id="dummy_commit_id",
                        is_deleted=False,
                        updated_at=func.now(),
                        status="created"
                    )
                    self.db.add(dummy_project)
                    self.db.commit()
                    print(f"Created dummy project with id: {dummy_project.id}")
                else:
                    print("Dummy project already exists")
            else:
                print("Dummy user not found, cannot create dummy project")
        finally:
            self.db.close()
