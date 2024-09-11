from app.core.celery_config import celery_app
from app.core.database import SessionLocal
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService


@celery_app.task(queue="parsing_queue")
def process_parsing(repo_details: dict, user_id: str, user_email: str, project_id: str):
    db = SessionLocal()
    try:
        parsing_service = ParsingService(db)
        parsing_service.parse_directory(
            ParsingRequest(**repo_details), user_id, user_email, project_id
        )
    finally:
        db.close()
