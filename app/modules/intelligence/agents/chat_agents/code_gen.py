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
from app.modules.intelligence.agents.agentic_tools.code_gen_agent import kickoff_code_generation_crew

from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService

from app.modules.intelligence.prompts.prompt_service import PromptService

logger = logging.getLogger(__name__)


class CodeGenerationAgent:
    def __init__(self, mini_llm, llm, db: Session):
        self.mini_llm = mini_llm
        self.llm = llm
        self.history_manager = ChatHistoryService(db)
        self.prompt_service = PromptService(db)
        self.agents_service = AgentsService(db)
        self.chain = None
        self.db = db

# [Previous imports and class definitions remain the same until the agent creation]

    async def create_agents(self):
        code_generator = Agent(
            role="Code Generation Agent",
            goal="Generate precise, copy-paste ready code modifications that maintain project consistency",
            backstory=f"""
                You are an expert code generation agent specialized in creating production-ready, 
                immediately usable code modifications. Your primary responsibilities include:
                1. Analyzing existing codebase context and understanding dependencies
                2. Planning code changes that maintain exact project patterns and style
                3. Implementing changes with copy-paste ready output
                4. Following existing code conventions exactly as shown in the input files
                5. Never modifying string literals, escape characters, or formatting unless specifically requested

                Key principles:
                - Provide required new imports in a separate code block
                - Output only the specific functions/classes being modified
                - Never change existing string formats or escape characters
                - Maintain exact indentation and spacing patterns from original code
                - Include clear section markers for where code should be inserted/modified

                Maintain efficiency within {self.max_iter} iterations while ensuring production-ready output.
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_multiple_node_ids,
                self.get_node_neighbours,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )

        return code_generator

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        node_ids: List[NodeContext],
        code_results: List[Dict[str, Any]],
        code_generator,
    ):
        code_generation_task = Task(
            description=f"""
            Work within {self.max_iter} iterations to generate copy-paste ready code based on:
            - Query: {query}
            - Project ID: {project_id}
            - Target Node IDs: {[node.model_dump() for node in node_ids]}
            - Existing Code Context: {code_results}

            Follow this structured approach:

            1. Context Analysis:
               - Review existing code structure and patterns
               - Use the get node neighbours tool to get the code context for nodes when required.
               - Identify dependencies and relationships
               - Understand the scope of required changes
               - Review existing code style precisely
               - Note exact indentation patterns
               - Identify string literal formats
               - Review import organization patterns

            2. Implementation Planning:
                 Break down the required changes into discrete steps
               - Identify potential impacts on other code sections
               - Plan the most efficient implementation approach
               - Plan changes that maintain exact formatting
               - Never modify existing patterns or escape characters unless requested
               - Identify required new imports

            3. Code Generation Format:
               File: `path/to/file.py`
               Replace function/class: `name_of_function`
               - or -
               Insert after line: `X`

               Required imports (if any):
               ```python
               from new.module import NewClass
               ```

               Modified/New Code:
               ```language
               modified_function_name():
                   # Your code here
                   
               ```               

            Important Output Rules:
            1. Always separate new imports from main code changes
            2. Never modify existing string literals or escape characters
            3. Follow existing indentation exactly
            4. Include only the specific functions/classes being modified
            5. Provide clear location markers for each code block
            6. Maintain exact spacing patterns from original code
            7. For new files, provide complete file content with imports at top
            8. Format code blocks with language specification for syntax highlighting.

            
            Keep responses focused and formatted for easy copy-paste into an IDE.
            No conversation or unnecessary explanation - just clear, actionable code blocks.
            """,
            expected_output="Copy-paste ready code changes with clear location markers and required imports",
            agent=code_generator,
        )

        return code_generation_task

# [Rest of the implementation remains the same]
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

            # Format the result for the response
            # result = {
            #     "plan": code_gen_result.plan.dict(),
            #     "modifications": [mod.dict() for mod in code_gen_result.modifications],
            #     "validation_notes": code_gen_result.validation_notes
            # }
            result = code_gen_result.raw
            
            tool_results = [SystemMessage(content=result)]

            # Timing for adding message chunk
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

            
