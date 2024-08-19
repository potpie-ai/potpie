import os
from typing import AsyncGenerator, List
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.pydantic_v1 import BaseModel, Field

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

class IntelligentAgent:
    def __init__(self, openai_key: str, tools: List):
        os.environ['OPENAI_API_KEY'] = openai_key
        self.llm = ChatOpenAI(temperature=0.7)
        self.tools = tools
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
        history = get_session_history(user_id, conversation_id)

        # Prepare the input dictionary for the chain
        inputs = {
            "input": query,
            "history": history.messages
        }

        # Invoke the chain with the prepared inputs
        result = await self.chain.ainvoke(inputs)

        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)

        # If tools are required, handle tool invocation here
        tool_results = []
        for tool in self.tools:
            if hasattr(tool, 'run'):
                tool_result = tool.run(query)
            elif hasattr(tool, 'arun'):
                tool_result = await tool.arun(query)
            else:
                tool_result = ""
            
            # Ensure tool_result is a string
            if not isinstance(tool_result, str):
                tool_result = str(tool_result)
                
            tool_results.append(tool_result)

        # Combine the results from the LLM and tools if needed
        combined_tool_results = "\n".join(tool_results)
        final_result = f"{result}\n\n{combined_tool_results}"

        # Save the new interaction (query and final result) to memory
        history.add_message(HumanMessage(content=query))
        history.add_message(AIMessage(content=final_result))

        # Ensure the final result is a string before yielding
        yield str(final_result)