import os
import asyncio
from typing import AsyncGenerator, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
import logging
from sqlalchemy.orm import Session
from app.modules.conversations.message.message_model import MessageType
from app.modules.intelligence.memory.postgres_history_manager import PostgresChatHistoryManager  # Import MessageType

# Set up logging
logger = logging.getLogger(__name__)

class IntelligentAgent:
    def __init__(self, openai_key: str, tools: List, db: Session):
        os.environ['OPENAI_API_KEY'] = openai_key
        self.llm = ChatOpenAI(temperature=0.7)
        self.tools = tools
        self.history_manager = PostgresChatHistoryManager(db)
        self.chain = self._create_chain()

    def _create_chain(self) -> RunnableSequence:
        # Construct the prompt template
        prompt_template = ChatPromptTemplate(
            messages=[
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )

        # Create a sequence of operations by chaining them together
        return prompt_template | self.llm

    async def run(self, query: str, user_id: str, conversation_id: str) -> AsyncGenerator[str, None]:
        # Ensure the query is a string
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        # Load the chat history
        history = self.history_manager.get_session_history(user_id, conversation_id)

        # Prepare the input dictionary for the chain
        inputs = {
            "input": query,
            "history": history
        }

        # Invoke the chain with the prepared inputs
        result = await self.chain.ainvoke(inputs)

        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)

        # Handle tool invocation here
        tool_results = await self._run_tools(query)

        # Combine the results from the LLM and tools if needed
        combined_tool_results = "\n".join(tool_results)
        final_result = f"{result}\n\n{combined_tool_results}"

        # Save the new interaction (query and final result) to the database
        self.history_manager.add_message(conversation_id, query, MessageType.HUMAN, user_id)
        self.history_manager.add_message(conversation_id, final_result, MessageType.AI_GENERATED)

        # Ensure the final result is a string before yielding
        yield str(final_result)

    async def _run_tools(self, query: str) -> List[str]:
        """Run all tools asynchronously and gather their results."""
        tool_results = []
        for tool in self.tools:
            try:
                if hasattr(tool, 'run'):
                    tool_result = await asyncio.to_thread(tool.run, query)
                elif hasattr(tool, 'arun'):
                    tool_result = await tool.arun(query)
                else:
                    tool_result = ""
            except Exception as e:
                logger.error(f"Error running tool {tool.name}: {str(e)}")
                tool_result = f"Error running tool {tool.name}: {str(e)}"
            
            # Ensure tool_result is a string
            if not isinstance(tool_result, str):
                tool_result = str(tool_result)
                
            tool_results.append(tool_result)
        return tool_results
