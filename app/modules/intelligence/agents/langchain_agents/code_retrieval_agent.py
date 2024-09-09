import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.chat_history_service import ChatHistoryService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_name_tool import GetCodeFromNodeNameTool
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import GetCodeFromNodeIdTool

logger = logging.getLogger(__name__)

class CodeRetrievalAgent:
    def __init__(self, openai_key: str, sql_db: Session):
        self.llm = ChatOpenAI(api_key=openai_key, temperature=0.7, model_kwargs={"stream": True})
        self.sql_db = sql_db
        self.history_manager = ChatHistoryService(sql_db)
        self.tools = [
            GetCodeFromNodeIdTool(sql_db)
        ]
        self.chain = None

    async def _create_chain(self) -> RunnableSequence:
        system_prompt = """
        You are an AI assistant specialized in retrieving code from a knowledge graph. 
        Your task is to assist users in finding specific code snippets based on node names or IDs.
        Use the provided tools to fetch the relevant code.

        Guidelines:
        1. Always use the appropriate tool to retrieve the code based on the user's query.
        2. If the code cannot be found, inform the user.

        Remember, your primary goal is to return the code snippet.
        """

        human_prompt = """
        User Query: {input}

        Please retrieve the relevant code based on the query. 
        If you need to use any tools to fetch the code, do so before responding.

        Your response should include only the retrieved code snippet 
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

    def _extract_node_info(self, query: str) -> Dict[str, str]:
        query_lower = query.lower()
        
        node_id_keywords = ["node id", "nodeid", "id of node", "id of the node"]
        node_name_keywords = ["node name", "nodename", "name of node", "name of the node"]
        
        for keyword in node_id_keywords:
            if keyword in query_lower:
                _, value = query_lower.split(keyword, 1)
                return {"node_id": value.strip()}
        
        for keyword in node_name_keywords:
            if keyword in query_lower:
                _, value = query_lower.split(keyword, 1)
                return {"node_name": value.strip()}
        
        return {"node_name": query.strip()}

    async def _run_tools(self, query: str, repo_id: str) -> List[SystemMessage]:
        tool_results = []
        node_info = self._extract_node_info(query)
        
        for tool in self.tools:
            try:
                tool_input = {"repo_id": repo_id, **node_info}
                
                logger.debug(f"Running tool {tool.name} with input: {tool_input}")

                tool_result = await tool.arun(**tool_input)

                if tool_result:
                    tool_results.append(
                        SystemMessage(content=f"Tool {tool.name} result: {tool_result}")
                    )
                    break  # Stop after first successful tool execution
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

            history = self.history_manager.get_session_history(user_id, conversation_id)
            validated_history = [
                (
                    HumanMessage(content=str(msg))
                    if isinstance(msg, (str, int, float))
                    else msg
                )
                for msg in history
            ]

            tool_results = await self._run_tools(query, project_id)

            full_query = f"Query: {query}\nProject ID: {project_id}"
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