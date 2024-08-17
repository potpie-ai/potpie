import os
from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, Tool
from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage

class IntelligentAgent:
    def __init__(self, openai_key: str, tools: list):
        self.llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_key)
        # Explicitly set the output_key to avoid the warning
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
        self.tools = tools
        self.agent_executor = self._create_agent_executor()

    def _create_agent_executor(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an intelligent assistant. Use the provided tools to answer user queries."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        tools = [Tool(name=tool.name, func=tool.run, description=tool.description) for tool in self.tools]
        agent = create_openai_functions_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True
        )

    async def run(self, query: str) -> AsyncGenerator[str, None]:
        result = await self.agent_executor.ainvoke({"input": query})
        
        for key, value in result.items():
            if isinstance(value, str):
                yield value
