from typing import AsyncGenerator, Dict
from app.modules.intelligence.agents_copy.chat_agent import (
    ChatAgentResponse,
    ChatContext,
    ChatAgent,
    AgentWithInfo,
)
from .llm_chat import LLM
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)

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
            [f"- {agents[id].id}: {agents[id].description}" for id in agents]
        )

    async def _run_classification(
        self, ctx: ChatContext, agent_descriptions: str
    ) -> ChatAgent:
        # classify the query into agent needed or not
        prompt = classification_prompt.format(
            agent_id=ctx.curr_agent_id,
            agent_descriptions=agent_descriptions,
            query=ctx.query,
        )
        classifier = LLM(
            self.llm_provider,
            prompt_template=prompt,
        )
        classification = await classifier.run(ctx)

        try:
            agent_id, confidence = classification.response.strip("`").split("|")
            confidence = float(confidence)
            selected_agent_id = (
                agent_id
                if confidence >= 0.5 and self.agents[agent_id]
                else ctx.curr_agent_id
            )
        except (ValueError, TypeError):
            logger.error("Classification format error, falling back to current agent")
            selected_agent_id = ctx.curr_agent_id

        return self.agents[selected_agent_id].agent

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        agent = await self._run_classification(ctx, self.agent_descriptions)
        return await agent.run_stream(ctx)


classification_prompt = """
    Given the user query and the current agent ID, select the most appropriate agent by comparing the query’s requirements with each agent’s specialties.

    Query: {query}
    Current Agent ID: {agent_id}

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

    3. **Multi-Agent Evaluation:**
    - Consider all agents’ described specialties thoroughly, not just the current agent.
    - For overlapping capabilities, favor the agent with more specialized expertise or more relevant tools/methodologies.
    - If no agent clearly surpasses a 0.5 confidence threshold, select the agent with the highest confidence score, even if it is below 0.5.

    4. **Confidence Scoring Guidelines:**
    - 0.9-1.0: Ideal match with the agent’s core, primary expertise.
    - 0.7-0.9: Strong match with the agent’s known capabilities.
    - 0.5-0.7: Partial or related match, not a direct specialty.
    - Below 0.5: Weak match; consider if another agent is more suitable, but still choose the best available option.

    Final Output Requirements:
    - Return ONLY the chosen agent_id and the confidence score in the format:
    `agent_id|confidence`

    Examples:
    - Direct expertise match: `debugging_agent|0.95`
    - Related capability (current agent): `current_agent_id|0.75`
    - Need different expertise: `ml_training_agent|0.85`
    - Overlapping domains, choose more specialized: `choose_higher_expertise_agent|0.80`
"""
