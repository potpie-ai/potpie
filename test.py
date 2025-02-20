""" Delete this file, this is temporary testing """

from app.modules.intelligence.agents_copy.chat_agents.crewai_rag_agent import (
    CrewAIRagAgent,
    AgentConfig,
    TaskConfig,
    ChatContext,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from app.modules.intelligence.agents_copy.chat_agents.system_agents.blast_radius_agent import (
    BlastRadiusAgent,
)
from app.modules.intelligence.agents_copy.chat_agents.llm_chat import (
    LLM,
)
from app.modules.intelligence.agents_copy.chat_agents.auto_router_agent import (
    AutoRouterAgent,
    AgentWithInfo,
)
from app.modules.intelligence.agents_copy.chat_agents.adaptive_agent import (
    AdaptiveAgent,
    AgentType,
    PromptService,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_file_structure import (
    get_code_file_structure_tool,
)
from app.modules.intelligence.tools.code_query_tools.get_code_graph_from_node_id_tool import (
    get_code_graph_from_node_id_tool,
)
from app.modules.intelligence.agents_copy.chat_agents.supervisor_agent import (
    SupervisorAgent,
)
from app.core.database import get_db
from dotenv import load_dotenv
from app.modules.utils.logger import setup_logger
import asyncio


logger = setup_logger(__name__)


async def go():
    print("starting test")
    load_dotenv()
    session = next(get_db())
    user_id = "0tyA7AEi9KbHYRyLu33qV9BjPgg1"

    llm_provider = ProviderService(session, user_id)
    tools_provider = ToolService(session, user_id)
    prompt_provider = PromptService(session)
    project_id = "0194d052-19b2-7b01-82a0-299da1538355"

    agent = CrewAIRagAgent(
        llm_provider=llm_provider,
        config=AgentConfig(
            role="Code Reviewer Agent",
            goal="Review the code for user and answer queries",
            backstory="You are a helpful code review agent that has access to entire codebase. User want's your help reviewing code",
            tasks=[
                TaskConfig(
                    description="review the code",
                    expected_output="code review in markdown format",
                )
            ],
        ),
        tools=[
            get_code_file_structure_tool(session),
            get_code_from_probable_node_name_tool(session, user_id),
            get_code_graph_from_node_id_tool(session),
        ],
    )
    blast_radius_agent = BlastRadiusAgent(
        llm_provider,
        tools_provider,
    )

    simple_llm_agent = LLM(
        llm_provider,
        "You are the Code Review Agent, review the code given in the query. Answer the following query: {query}",
    )

    adaptive_agent = AdaptiveAgent(
        llm_provider, prompt_provider, blast_radius_agent, AgentType.CODE_CHANGES
    )

    auto_router_agent = AutoRouterAgent(
        llm_provider,
        agents=[
            AgentWithInfo(
                id="code_changes_agent",
                name="Code Changes Agent",
                description="An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                agent=adaptive_agent,
            ),
            AgentWithInfo(
                id="code_review_agent",
                name="Code Review Agent",
                description="An agent specialized in code reviews, use this for reviewing codes",
                agent=simple_llm_agent,
            ),
        ],
        curr_agent_id="code_changes_agent",
    )

    agent = SupervisorAgent(
        llm_provider, tools_provider, prompt_provider, "code_changes_agent"
    )

    res = await agent.run(
        ctx=ChatContext(
            query="""Can you generate some code to show examples of using prompt service in the repo? Also respond with list of tools you used and the data returned by them, if they errored out mention that""",
            project_id=project_id,
            curr_agent_id="code_changes_agent",
            history=[],
        )
    )
    print(res.response, res.citations)

    # res = simple_llm_agent.run_stream("who built the taj mahal?", history=[])
    # async for chunk in res:
    #     print(chunk.response, chunk.citations)
    #     print("==============")


asyncio.run(go())
