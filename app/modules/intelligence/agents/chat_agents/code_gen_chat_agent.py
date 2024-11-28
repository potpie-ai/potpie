import json
import logging
import time
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List

from crewai import Agent, Task
from langchain.schema import HumanMessage, SystemMessage

from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.message.message_schema import NodeContext

from app.modules.intelligence.agents.agents.code_gen_agent import kickoff_code_generation_crew
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService

from app.modules.intelligence.prompts.prompt_service import PromptService

logger = logging.getLogger(__name__)


class CodeGenerationChatAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = llm
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None
        self.db = db


    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
        node_ids: List[NodeContext],
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()
        try:
            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            tool_results = []
            citations = []
            code_gen_start_time = time.time()
            
            # Call multi-agent code generation instead of RAG
            code_gen_result = await kickoff_code_generation_crew(
                query,
                project_id,
                validated_history[-5:],
                node_ids,
                self.db,
                self.llm,
                self.mini_llm,
                user_id,
            )
            
            code_gen_duration = time.time() - code_gen_start_time
            logger.info(
                f"Time elapsed since entering run: {time.time() - start_time:.2f}s, "
                f"Duration of Code Generation: {code_gen_duration:.2f}s"
            )

            result = code_gen_result.raw
            
            tool_results = [SystemMessage(content=result)]

            add_chunk_start_time = time.time()
            self.history_manager.add_message_chunk(
                conversation_id,
                tool_results[0].content,
                MessageType.AI_GENERATED,
                citations=citations,
            )
            add_chunk_duration = time.time() - add_chunk_start_time
            logger.info(
                f"Time elapsed since entering run: {time.time() - start_time:.2f}s, "
                f"Duration of adding message chunk: {add_chunk_duration:.2f}s"
            )

            # Timing for flushing message buffer
            flush_buffer_start_time = time.time()
            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )
            flush_buffer_duration = time.time() - flush_buffer_start_time
            logger.info(
                f"Time elapsed since entering run: {time.time() - start_time:.2f}s, "
                f"Duration of flushing message buffer: {flush_buffer_duration:.2f}s"
            )
            
            yield json.dumps({"citations": citations, "message": result})
            
        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}")
            error_message = f"An error occurred during code generation: {str(e)}"
            yield json.dumps({"error": error_message})

            
