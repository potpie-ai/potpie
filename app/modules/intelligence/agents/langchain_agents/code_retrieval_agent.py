import asyncio
import logging
from typing import AsyncGenerator, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from neo4j import GraphDatabase

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_name_tool import GetCodeFromNodeNameTool
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import GetCodeFromNodeIdTool
from app.modules.projects.projects_model import Project

logger = logging.getLogger(__name__)

class CodeRetrievalAgent:
    def __init__(self, openai_key: str, sql_db: Session):
        self.llm = ChatOpenAI(
            api_key=openai_key, temperature=0.7, model_kwargs={"stream": True}
        )
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.tools = [
            GetCodeFromNodeNameTool(sql_db),
            GetCodeFromNodeIdTool(sql_db)
        ]
        self.chain = None

    def get_repo_name(self, project_id: str) -> str:
        project = self.sql_db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found")
        return project.repo_name

    async def _create_chain(self) -> RunnableSequence:
        system_prompt = """
        You are an AI assistant specialized in retrieving and analyzing code from a knowledge graph. 
        Your task is to assist users in finding and understanding specific code snippets based on node names or IDs.
        Use the provided tools to fetch the relevant code and provide insightful explanations.

        Guidelines:
        1. Always use the appropriate tool to retrieve the code based on the user's query.
        2. If the code is successfully retrieved, analyze it and provide a brief explanation of its purpose and functionality.
        3. If the code cannot be found, inform the user and suggest possible reasons or alternative approaches.
        4. Be concise but informative in your responses.
        5. If asked about relationships or context, use your understanding of the code structure to provide relevant information.

        Remember, your primary goal is to help users understand the codebase through efficient retrieval and clear explanations.
        """

        human_prompt = """
        User Query: {input}

        Please retrieve the relevant code and provide an analysis based on the query. 
        If you need to use any tools to fetch the code, do so before responding.

        Your response should include:
        1. The retrieved code snippet (if available)
        2. A brief explanation of the code's purpose and functionality
        3. Any relevant context or relationships within the codebase
        4. Suggestions or next steps if the user needs more information

        Response:
        """

        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="tool_results"),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        )
        return prompt_template | self.llm

    async def _run_tools(self, query: str, repo_name: str) -> List[SystemMessage]:
        tool_results = []
        for tool in self.tools:
            try:
                tool_input = {"repo_name": repo_name}
                if "node_name" in tool.args_schema.__fields__:
                    tool_input["node_name"] = query
                elif "node_id" in tool.args_schema.__fields__:
                    tool_input["node_id"] = query

                logger.debug(f"Running tool {tool.name} with input: {tool_input}")

                tool_result = (
                    await tool.arun(**tool_input)
                    if hasattr(tool, "arun")
                    else await asyncio.to_thread(tool.run, **tool_input)
                )

                if tool_result:
                    tool_results.append(
                        SystemMessage(content=f"Tool {tool.name} result: {tool_result}")
                    )
            except Exception as e:
                logger.error(f"Error running tool {tool.name}: {str(e)}", exc_info=True)

        return tool_results

    async def run(
        self,
        query: str,
        project_id: str,
        user_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            if not self.chain:
                self.chain = await self._create_chain()

            repo_name = self.get_repo_name(project_id)

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            tool_results = await self._run_tools(query, repo_name)

            full_query = f"Query: {query}\nRepository: {repo_name}"
            inputs = {
                "history": validated_history,
                "tool_results": tool_results,
                "input": full_query,
            }

            logger.debug(f"Inputs to LLM: {inputs}")

            full_response = ""
            async for chunk in self.chain.astream(inputs):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += content
                self.history_manager.add_message_chunk(
                    conversation_id, content, MessageType.AI_GENERATED
                )
                yield content

            logger.debug(f"Full LLM response: {full_response}")

            self.history_manager.flush_message_buffer(
                conversation_id, MessageType.AI_GENERATED
            )

        except Exception as e:
            logger.error(f"Error during CodeRetrievalAgent run: {str(e)}", exc_info=True)
            yield f"An error occurred: {str(e)}"