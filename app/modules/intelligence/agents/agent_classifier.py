from typing import List
from langchain.schema import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import MessageResponse
from app.modules.intelligence.prompts.classification_prompts import ClassificationResult

class AgentClassification(BaseModel):
    agent_id: str = Field(..., description="ID of the agent that should handle the query")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Reasoning behind the agent selection")

class AgentClassifier:
    def __init__(self, llm, available_agents):
        self.llm = llm
        self.available_agents = available_agents
        self.parser = PydanticOutputParser(pydantic_object=AgentClassification)
        
    def create_prompt(self) -> str:
        
        return """You are an expert agent router.
         User's query: {query}
         Conversation history: {history}
        
        Based on the user's query and conversation history,
        select the most appropriate agent from the following options:

        Available Agents:
        {agents_desc}

        Analyze the query and select the agent that best matches the user's needs.
        Consider:
        1. The specific task or question type
        2. Required expertise
        3. Context from conversation history
        4. Any explicit agent requests

        {format_instructions}
        """

    async def classify(self, messages: List[MessageResponse]) -> AgentClassification:
        """Classify the conversation and determine which agent should handle it"""
        
        if not messages:
            return AgentClassification(
                agent_id=self.available_agents[0].id,  # Default to first agent
                confidence=0.0,
                reasoning="No messages to classify"
            )
            
        # Format agent descriptions
        agents_desc = "\n".join([
            f"{i+1}. {agent.id}: {agent.description}"
            for i, agent in enumerate(self.available_agents)
        ])
        
        # Get the last message and up to 10 messages of history
        last_message = messages[-1].content if messages else ""
        history = [msg.content for msg in messages[-10:]] if len(messages) > 1 else []
        
        inputs = {
            "query": last_message,
            "history": history,
            "agents_desc": agents_desc,
            "format_instructions": self.parser.get_format_instructions()
        }
        
        # Rest of the classification logic...
        
        prompt = ChatPromptTemplate.from_template(self.create_prompt())
        
        chain = prompt | self.llm | self.parser
        
        result = await chain.ainvoke(
            inputs
        )
        
        return result 