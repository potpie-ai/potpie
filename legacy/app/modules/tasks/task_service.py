from app.modules.tasks.task_model import (  # Update import to include Task model and TaskType
    Task,
    TaskType,
)


class TaskService:
    def __init__(self, db):
        self.db = db

    def create_task(self, task_type: TaskType, custom_status: str, project_id: int):
        new_task = Task(
            task_type=task_type, custom_status=custom_status, project_id=project_id
        )
        self.db.add(new_task)
        self.db.commit()
        self.db.refresh(new_task)
        return new_task

    def get_task(self, task_id: int):
        return self.db.query(Task).filter(Task.id == task_id).first()

    def update_task(self, task_id: int, custom_status: str = None, result: str = None):
        task = self.get_task(task_id)
        if task:
            if custom_status is not None:
                task.custom_status = custom_status
            if result is not None:
                task.result = result
            self.db.commit()
            self.db.refresh(task)
            return task
        return None

    def delete_task(self, task_id: int):
        task = self.get_task(task_id)
        if task:
            self.db.delete(task)
            self.db.commit()
            return True
        return False
