from typing import AsyncGenerator, Dict
from app.modules.intelligence.agents.chat_agent import (
    ChatAgentResponse,
    ChatContext,
    ChatAgent,
    AgentWithInfo,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AutoRouterAgent(ChatAgent):
    """AutoRouterAgent routes the request into one of n system agents based on query"""

    def __init__(
        self,
        llm_provider: ProviderService,
        agents: Dict[str, AgentWithInfo],
    ):
        self.llm_provider = llm_provider
        self.agents = agents
        self.agent_descriptions = "\n".join(
            [
                f"agent_id: {agents[id].id}\n description: {agents[id].description}\n\n"
                for id in agents
            ]
        )

    async def _run_classification(
        self, ctx: ChatContext, agent_descriptions: str
    ) -> ChatAgent:
        # classify the query into agent needed or not
        prompt = classification_prompt.format(
            agent_id=ctx.curr_agent_id,
            agent_descriptions=agent_descriptions,
            query=ctx.query,
            history=" ,".join(message for message in ctx.history),
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert agent classifier that helps route queries to the most appropriate agent. Agents have full access to the users code repository",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            classification: ClassificationResponse = (
                await self.llm_provider.call_llm_with_structured_output(
                    messages=messages,
                    output_schema=ClassificationResponse,  # type: ignore
                )
            )

            agent_id = classification.agent_id
            confidence = float(classification.confidence_score)
            selected_agent_id = (
                agent_id
                if confidence >= 0.5 and self.agents[agent_id]
                else ctx.curr_agent_id
            )
            logger.info(f"Classification successful: using {selected_agent_id} agent")
        except (ValueError, TypeError, KeyError, Exception) as e:
            logger.error("Classification error, falling back to current agent: %e", e)
            selected_agent_id = ctx.curr_agent_id

        return self.agents[selected_agent_id].agent

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        async for chunk in agent.run_stream(ctx):
            yield chunk


class ClassificationResponse(BaseModel):
    agent_id: str = Field(description="agent_id of the best matching agent")
    confidence_score: float = Field(
        description="confidence score of the best matching agent, should be the maximum confidence score and be a valid floating point number between 0 and 1"
    )


classification_prompt = """
    You are part of the ai agentic system that has deep understanding of the users codebase/repository. You are being used to route the query to appropriate specialized agent
    Given the user query and the current agent ID, select the most appropriate agent by comparing the query’s requirements with each agent’s specialties.

    User Query: {query}
    Current Agent ID: {agent_id}
    Chat history:
    {history}
    --- end of Chat history ----

    Available agents and their specialties:
    {agent_descriptions}

    Follow the instructions below to determine the best matching agent and provide a confidence score:

    Analysis Instructions (DO NOT include these instructions in the final answer):
    1. **Semantic Analysis:**
    - Identify the key topics, technical terms, and the user’s intent from the query.
    - Compare these elements to each agent’s detailed specialty description.
    - Focus on specific skills, tools, frameworks, and domain expertise mentioned.

    2. **Contextual Weighting:**
    - If the query strongly aligns with the current agent’s known capabilities, add +0.15 confidence for direct core expertise and +0.1 for related domain knowledge.
    - If the query introduces new topics outside the current agent’s domain, do not apply the current agent bias. Instead, evaluate all agents equally based on their described expertise.
    - Refer to the chat history to get context about current query and it's possible answer but classify only based on the current query, \
      it's only the current query that is being routed to the appropriate agent along with the chat history

    3. **Multi-Agent Evaluation:**
    - Consider all agents’ described specialties thoroughly, not just the current agent.
    - For overlapping capabilities, favor the agent with more specialized expertise or more relevant tools/methodologies.
    - If no agent clearly surpasses a 0.5 confidence threshold, select the agent with the highest confidence score, even if it is below 0.5.

    4. **Confidence Scoring Guidelines:**
    - 0.9-1.0: Ideal match with the agent’s core, primary expertise.
    - 0.7-0.9: Strong match with the agent’s known capabilities.
    - 0.5-0.7: Partial or related match, not a direct specialty.
    - Below 0.5: Weak match; consider if another agent is more suitable, but still choose the best available option.

    IMPORTANT:
    - Classify based on the current query, history data is already present. You are choosing agent to process current query only
    - Select general purpose agent only if the agent doesn't need to go through the repository to answer query
    - Don't choose general purpose agent if the agent has to access user repository since general purpose agent doesn't have tools that have access to the codebase
    - Use general purpose agent for queries like greetings, simple web lookups or follow-up questions etc that don't require repository access

"""
