import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from uuid6 import uuid7

from app.modules.intelligence.prompts.prompt_model import (
    AgentPromptMapping,
    Prompt,
    PromptStatusType,
)
from app.modules.intelligence.prompts.prompt_schema import (
    AgentPromptMappingCreate,
    AgentPromptMappingResponse,
    PromptCreate,
    PromptListResponse,
    PromptResponse,
    PromptType,
    PromptUpdate,
)

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

    async def create_prompt(
        self, prompt: PromptCreate, user_id: Optional[str]
    ) -> PromptResponse:
        try:
            prompt_id = str(uuid7())
            now = datetime.now(timezone.utc)
            new_prompt = Prompt(
                id=prompt_id,
                text=prompt.text,
                type=prompt.type,
                status=prompt.status or PromptStatusType.ACTIVE,
                created_by=user_id,
                created_at=now,
                updated_at=now,
                version=1,
            )
            self.db.add(new_prompt)
            self.db.commit()
            self.db.refresh(new_prompt)

            logger.info(
                f"Created new prompt with ID: {prompt_id}, user_id: {user_id or 'System'}"
            )
            return PromptResponse.model_validate(new_prompt)
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"IntegrityError in create_prompt: {e}", exc_info=True)
            raise PromptCreationError(
                "Failed to create prompt due to a database integrity error."
            ) from e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error in create_prompt: {e}", exc_info=True)
            raise PromptCreationError(
                "An unexpected error occurred while creating the prompt."
            ) from e

    async def update_prompt(
        self, prompt_id: str, prompt: PromptUpdate, user_id: str
    ) -> PromptResponse:
        try:
            db_prompt = (
                self.db.query(Prompt)
                .filter(Prompt.id == prompt_id, Prompt.created_by == user_id)
                .first()
            )
            if not db_prompt:
                raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")

            for field, value in prompt.model_dump(exclude_unset=True).items():
                setattr(db_prompt, field, value)

            db_prompt.updated_at = datetime.now(timezone.utc)
            db_prompt.version += 1

            self.db.commit()
            self.db.refresh(db_prompt)

            logger.info(f"Updated prompt with ID: {prompt_id}, user_id: {user_id}")
            return PromptResponse.model_validate(db_prompt)
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error in update_prompt: {e}", exc_info=True)
            raise PromptUpdateError(
                f"Failed to update prompt {prompt_id} due to a database error"
            ) from e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error in update_prompt: {e}", exc_info=True)
            raise PromptUpdateError(
                f"Failed to update prompt {prompt_id} due to an unexpected error"
            ) from e

    async def delete_prompt(self, prompt_id: str, user_id: str) -> None:
        try:
            result = (
                self.db.query(Prompt)
                .filter(Prompt.id == prompt_id, Prompt.created_by == user_id)
                .delete()
            )
            if result == 0:
                raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")
            self.db.commit()
            logger.info(f"Deleted prompt with ID: {prompt_id}, user_id: {user_id}")
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error in delete_prompt: {e}", exc_info=True)
            raise PromptDeletionError(
                f"Failed to delete prompt {prompt_id} due to a database error"
            ) from e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error in delete_prompt: {e}", exc_info=True)
            raise PromptDeletionError(
                f"Failed to delete prompt {prompt_id} due to an unexpected error"
            ) from e

    async def fetch_prompt(self, prompt_id: str, user_id: str) -> PromptResponse:
        try:
            prompt = (
                self.db.query(Prompt)
                .filter(Prompt.id == prompt_id)
                .first()
            )
            if not prompt:
                raise PromptNotFoundError(f"Prompt with id {prompt_id} not found")
            return PromptResponse.model_validate(prompt)
        except PromptNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error in fetch_prompt: {e}", exc_info=True)
            raise PromptFetchError(
                f"Failed to fetch prompt {prompt_id} due to a database error"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in fetch_prompt: {e}", exc_info=True)
            raise PromptFetchError(
                f"Failed to fetch prompt {prompt_id} due to an unexpected error"
            ) from e

    async def list_prompts(
        self, query: Optional[str], skip: int, limit: int, user_id: str
    ) -> PromptListResponse:
        try:
            prompts_query = (
                self.db.query(Prompt)
                .filter(Prompt.created_by == user_id)
                .order_by(Prompt.text)
            )

            if query:
                prompts_query = prompts_query.filter(Prompt.text.ilike(f"%{query}%"))

            total = prompts_query.count()
            prompts = prompts_query.offset(skip).limit(limit).all()

            prompt_responses = [
                PromptResponse.model_validate(prompt) for prompt in prompts
            ]

            return PromptListResponse(prompts=prompt_responses, total=total)
        except SQLAlchemyError as e:
            logger.error(f"Database error in list_prompts: {e}", exc_info=True)
            raise PromptListError(
                "Failed to list prompts due to a database error"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in list_prompts: {e}", exc_info=True)
            raise PromptListError(
                "Failed to list prompts due to an unexpected error"
            ) from e

    async def map_agent_to_prompt(
        self, mapping: AgentPromptMappingCreate
    ) -> AgentPromptMappingResponse:
        try:
            existing_mapping = (
                self.db.query(AgentPromptMapping)
                .filter(
                    AgentPromptMapping.agent_id == mapping.agent_id,
                    AgentPromptMapping.prompt_stage == mapping.prompt_stage,
                )
                .first()
            )

            if existing_mapping:
                existing_mapping.prompt_id = mapping.prompt_id
                self.db.commit()
                self.db.refresh(existing_mapping)
                return AgentPromptMappingResponse.model_validate(existing_mapping)
            else:
                new_mapping = AgentPromptMapping(
                    id=str(uuid7()),
                    agent_id=mapping.agent_id,
                    prompt_id=mapping.prompt_id,
                    prompt_stage=mapping.prompt_stage,
                )
                self.db.add(new_mapping)
                self.db.commit()
                self.db.refresh(new_mapping)
                return AgentPromptMappingResponse.model_validate(new_mapping)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error in map_agent_to_prompt: {e}", exc_info=True)
            raise PromptServiceError("Failed to map agent to prompt", e) from e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error in map_agent_to_prompt: {e}", exc_info=True)
            raise PromptServiceError(
                "Failed to map agent to prompt due to an unexpected error"
            ) from e

    async def create_or_update_system_prompt(
        self, prompt: PromptCreate, agent_id: str, stage: int
    ) -> PromptResponse:
        try:
            existing_prompt = (
                self.db.query(Prompt)
                .join(AgentPromptMapping)
                .filter(
                    Prompt.type == prompt.type,
                    Prompt.created_by.is_(None),
                    AgentPromptMapping.agent_id == agent_id,
                    AgentPromptMapping.prompt_stage == stage,
                )
                .first()
            )

            if existing_prompt:
                # Check if the prompt needs to be updated
                update_needed = False
                update_reasons = []

                if existing_prompt.text != prompt.text:
                    update_needed = True
                    update_reasons.append("text changed")

                if str(existing_prompt.status) != str(prompt.status):
                    update_needed = True
                    update_reasons.append("status changed")
                    logger.info(f"Status changed from {existing_prompt.status} to {prompt.status}")

                if update_needed:
                    existing_prompt.text = prompt.text
                    existing_prompt.status = prompt.status
                    existing_prompt.updated_at = datetime.now(timezone.utc)
                    existing_prompt.version += 1
                    prompt_to_return = existing_prompt
                    logger.info(f"Existing prompt is updated. Reasons: {', '.join(update_reasons)}")
                else:
                    prompt_to_return = existing_prompt
                    logger.info("Existing prompt is kept as it is. No changes detected.")
            else:
                # Create new prompt
                new_prompt = Prompt(
                    id=str(uuid7()),
                    text=prompt.text,
                    type=prompt.type,
                    status=prompt.status or PromptStatusType.ACTIVE,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    version=1,
                )
                self.db.add(new_prompt)
                prompt_to_return = new_prompt
                print("Inserting a new prompt.")

            self.db.commit()
            self.db.refresh(prompt_to_return)
            return PromptResponse.model_validate(prompt_to_return)
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error in create_or_update_system_prompt: {e}", exc_info=True)
            raise PromptServiceError("Failed to create or update system prompt") from e

    async def get_prompts_by_agent_id_and_types(
        self, agent_id: str, prompt_types: List[PromptType]
    ) -> List[PromptResponse]:
        try:
            prompts = (
                self.db.query(Prompt)
                .join(AgentPromptMapping)
                .filter(
                    AgentPromptMapping.agent_id == agent_id,
                    Prompt.type.in_(prompt_types),
                )
                .all()
            )

            return [PromptResponse.model_validate(prompt) for prompt in prompts]
        except SQLAlchemyError as e:
            raise PromptServiceError(
                "Failed to get prompts by agent ID and types"
            ) from e
