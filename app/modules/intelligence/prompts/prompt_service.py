import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from uuid6 import uuid7
from datetime import datetime, timezone

from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptResponse, PromptType, PromptUpdate
from app.modules.intelligence.prompts.prompt_model import Prompt, PromptStatusType

logger = logging.getLogger(__name__)

class PromptServiceError(Exception):
    """Base exception class for PromptService errors."""

class PromptNotFoundError(PromptServiceError):
    """Raised when a prompt is not found."""

class PromptCreationError(PromptServiceError):
    """Raised when there's an error creating a prompt."""

class PromptUpdateError(PromptServiceError):
    """Raised when there's an error updating a prompt."""

class PromptDeletionError(PromptServiceError):
    """Raised when there's an error deleting a prompt."""

class PromptFetchError(PromptServiceError):
    """Raised when there's an error fetching a prompt."""

class PromptListError(PromptServiceError):
    """Raised when there's an error listing prompts."""

class PromptService:
    def __init__(self, db: Session):
        self.db = db

    @classmethod
    def create(cls, db: Session):
        return cls(db)

    async def create_prompt(self, prompt: PromptCreate, user_id: str) -> PromptResponse:
        try:
            with self.db.begin():
                prompt_id = str(uuid7())
                now = datetime.now(timezone.utc)
                new_prompt = Prompt(
                    id=prompt_id,
                    text=prompt.text,
                    visibility=prompt.visibility,
                    type=PromptType.USER,
                    status=prompt.status or PromptStatusType.ACTIVE,
                    created_by=user_id,
                    created_at=now,
                    updated_at=now,
                    version=1  # Add version field
                )
                self.db.add(new_prompt)
        
            logger.info(f"Created new prompt with ID: {prompt_id}, user_id: {user_id}")
            return PromptResponse(
                id=new_prompt.id,
                text=new_prompt.text,
                type=new_prompt.type,
                visibility=new_prompt.visibility,
                version=new_prompt.version,
                status=new_prompt.status,
                created_by=new_prompt.created_by,
                created_at=new_prompt.created_at.isoformat(),
                updated_at=new_prompt.updated_at.isoformat()
            )
        except IntegrityError as e:
            logger.error(f"IntegrityError in create_prompt: {e}", exc_info=True)
            raise PromptServiceError("Failed to create prompt due to a database integrity error.") from e
        except Exception as e:
            logger.error(f"Unexpected error in create_prompt: {e}", exc_info=True)
            self.db.rollback()
            raise PromptServiceError("An unexpected error occurred while creating the prompt.") from e

    async def update_prompt(self, prompt_id: str, prompt: PromptUpdate, user_id: str) -> PromptResponse:
        try:
            with self.db.begin():
                db_prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
                if not db_prompt:
                    raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")
                
                for field, value in prompt.model_dump(exclude_unset=True).items():
                    setattr(db_prompt, field, value)
                
                db_prompt.updated_at = datetime.now(timezone.utc)
                db_prompt.version += 1  # Increment version on update
                # The transaction will be automatically committed if we reach this point
            
            logger.info(f"Updated prompt with ID: {prompt_id}, user_id: {user_id}")
            return PromptResponse(
                id=db_prompt.id,
                text=db_prompt.text,
                type=db_prompt.type,
                visibility=db_prompt.visibility,
                version=db_prompt.version,
                status=db_prompt.status,
                created_by=db_prompt.created_by,
                created_at=db_prompt.created_at.isoformat(),
                updated_at=db_prompt.updated_at.isoformat()
            )
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in update_prompt: {e}", exc_info=True)
            # The transaction will be automatically rolled back
            raise PromptServiceError(f"Failed to update prompt {prompt_id} due to a database error") from e
        except Exception as e:
            logger.error(f"Unexpected error in update_prompt: {e}", exc_info=True)
            # Ensure rollback in case of any other exception
            self.db.rollback()
            raise PromptServiceError(f"Failed to update prompt {prompt_id} due to an unexpected error") from e

    async def delete_prompt(self, prompt_id: str, user_id: str) -> None:
        try:
            deleted_prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).delete()
            if deleted_prompt == 0:
                raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")
            self.db.commit()
            logger.info(f"Deleted prompt with ID: {prompt_id}, user_id: {user_id}")
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in delete_prompt: {e}", exc_info=True)
            self.db.rollback()
            raise PromptDeletionError(f"Failed to delete prompt {prompt_id} due to a database error") from e
        except Exception as e:
            logger.error(f"Unexpected error in delete_prompt: {e}", exc_info=True)
            self.db.rollback()
            raise PromptDeletionError(f"Failed to delete prompt {prompt_id} due to an unexpected error") from e

    async def fetch_prompt(self, prompt_id: str, user_id: str) -> PromptResponse:
        try:
            prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
            if not prompt:
                raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")
            return PromptResponse(
                id=prompt.id,
                text=prompt.text,
                type=prompt.type,
                visibility=prompt.visibility,
                version=prompt.version,
                status=prompt.status,
                created_by=prompt.created_by,
                created_at=prompt.created_at.isoformat(),
                updated_at=prompt.updated_at.isoformat()
            )
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in fetch_prompt: {e}", exc_info=True)
            raise PromptFetchError(f"Failed to fetch prompt {prompt_id} due to a database error") from e
        except Exception as e:
            logger.error(f"Unexpected error in fetch_prompt: {e}", exc_info=True)
            raise PromptFetchError(f"Failed to fetch prompt {prompt_id} due to an unexpected error") from e

    async def list_prompts(self, query: Optional[str], skip: int, limit: int, user_id: str) -> List[PromptResponse]:
        try:
            prompts_query = self.db.query(Prompt)
            if query:
                prompts_query = prompts_query.filter(Prompt.text.ilike(f"%{query}%"))
            prompts = prompts_query.offset(skip).limit(limit).all()
            return [PromptResponse.model_validate({
                **prompt.__dict__,
                'created_at': prompt.created_at.isoformat(),
                'updated_at': prompt.updated_at.isoformat()
            }) for prompt in prompts]
        except SQLAlchemyError as e:
            logger.error(f"Database error in list_prompts: {e}", exc_info=True)
            raise PromptListError("Failed to list prompts due to a database error") from e
        except Exception as e:
            logger.error(f"Unexpected error in list_prompts: {e}", exc_info=True)
            raise PromptListError("Failed to list prompts due to an unexpected error") from e

    async def get_all_prompts(self, skip: int, limit: int, user_id: str) -> List[PromptResponse]:
        try:
            prompts = self.db.query(Prompt).offset(skip).limit(limit).all()
            return [PromptResponse.model_validate({
                **prompt.__dict__,
                'created_at': prompt.created_at.isoformat(),
                'updated_at': prompt.updated_at.isoformat()
            }) for prompt in prompts]
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_all_prompts: {e}", exc_info=True)
            raise PromptListError("Failed to get all prompts due to a database error") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_all_prompts: {e}", exc_info=True)
            raise PromptListError("Failed to get all prompts due to an unexpected error") from e