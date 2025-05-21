import re
from typing import AsyncGenerator, List, Optional, Any, Dict, Union
import asyncio
import uuid
import json

from app.modules.intelligence.agents.chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallResponse,
    ToolCallEventType,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
    AgentProvider,
)
from app.modules.utils.logger import setup_logger
from pydantic import BaseModel, Field

# Agno imports
from agno.agent import Agent
from agno.models.litellm import LiteLLM

import json
from langchain_core.tools import BaseTool, StructuredTool
from agno.tools import tool


class AgnoToolWrapper:
    """
    A class-based wrapper for Agno AI tools that allows proper serialization.
    This is an alternative approach if the dict-based approach still has issues.
    """

    def __init__(self, lc_tool: Union[BaseTool, StructuredTool]):
        self.lc_tool = lc_tool
        self.name = re.sub(" ", "", lc_tool.name)
        self.description = lc_tool.description
        self.parameters = self._extract_parameters()
        self.return_schema = {
            "result": {
                "type": "string",
                "description": "The result of the tool execution",
            }
        }

    def _extract_parameters(self) -> Dict[str, Any]:
        parameter_schema = {}

        # Similar parameter extraction as in the function above
        # Copy the parameter extraction logic from the function version

        # If no parameters found, use default input
        if not parameter_schema:
            parameter_schema["input"] = {
                "type": "string",
                "description": "Input for the tool",
                "required": True,
            }

        return parameter_schema

    def function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # For structured tools, we need to pass params as kwargs
            if isinstance(self.lc_tool, StructuredTool):
                result = self.lc_tool.run(**params)
            # For basic tools, we need to pass a single string as input
            else:
                # If there's an 'input' parameter, use that
                if "input" in params:
                    result = self.lc_tool.run(params["input"])
                # Otherwise, convert all params to a string
                else:
                    input_str = str(params)
                    result = self.lc_tool.run(input_str)

            return {
                "result": str(result)
            }  # Ensure result is string for JSON serialization
        except Exception as e:
            return {"error": str(e)}


logger = setup_logger(__name__)


class TeamAgentConfig(BaseModel):
    """Configuration for the Team Agent"""

    team_instructions: List[str] = Field(
        default=["Respond in a clear, concise manner", "Use markdown for formatting"],
        description="Instructions for the team",
    )
    show_tool_calls: bool = Field(
        default=True, description="Whether to show tool calls in the response"
    )
    planner_role: str = Field(
        default="Plan the approach to solve the problem",
        description="Role description for the planner agent",
    )
    executor_role: str = Field(
        default="Execute the plan and provide the final solution",
        description="Role description for the executor agent",
    )
    planner_instructions: List[str] = Field(
        default=["Break down complex problems", "Consider all relevant factors"],
        description="Instructions for the planner agent",
    )
    executor_instructions: List[str] = Field(
        default=["Follow the plan precisely", "Be thorough in execution"],
        description="Instructions for the executor agent",
    )
    coordinator_instructions: List[str] = Field(
        default=[
            "First ask the Planner to create a structured plan",
            "Then have the Executor implement the plan using available tools",
            "Combine their outputs into a comprehensive response",
        ],
        description="Instructions for coordinating between planner and executor",
    )


class TeamChatAgent(ChatAgent):
    """
    A ChatAgent implementation that uses Agno AI's Team coordination with a
    Planner agent and an Executor agent, coordinated by a leader agent.
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        tools: List[Any],
        config: Optional[TeamAgentConfig] = None,
    ):
        """Initialize the TeamChatAgent with provider, tools, and optional configuration"""
        self.llm_provider = llm_provider
        self.tools = [AgnoToolWrapper(tool) for tool in tools]
        self.config = config or TeamAgentConfig()

        # Create the team when initialized
        self._create_team()

    def _create_team(self):
        """Create the Agno AI Team with Planner and Executor agents"""
        llm = self._get_llm()

        # Create Planner agent
        self.planner_agent = Agent(
            name="Planner",
            role=self.config.planner_role,
            model=llm,
            tools=[],  # Planner doesn't use tools directly
            instructions=self.config.planner_instructions,
            show_tool_calls=self.config.show_tool_calls,
            markdown=True,
            reasoning=True,
        )

        # Create Executor agent
        self.executor_agent = Agent(
            name="Executor",
            role=self.config.executor_role,
            model=llm,
            tools=self.tools,  # Executor uses all available tools
            instructions=self.config.executor_instructions,
            show_tool_calls=self.config.show_tool_calls,
            markdown=True,
            reasoning=True,
        )

        # Create the coordinator/leader agent
        self.team_agent = Agent(
            name="TeamCoordinator",
            team=[self.planner_agent, self.executor_agent],
            model=llm,
            description="You coordinate a planner and executor to solve problems efficiently",
            instructions=self.config.coordinator_instructions,
            show_tool_calls=self.config.show_tool_calls,
            markdown=True,
            reasoning=True,
        )

    def _get_llm(self):
        """Get the LiteLLM model from the provider"""
        # Build parameters using the config object
        config = self.llm_provider.chat_config
        params = self.llm_provider._build_llm_params(config)
        routing_provider = config.model.split("/")[0]

        # Get extra parameters and headers for API calls
        extra_params, headers = self.llm_provider.get_extra_params_and_headers(
            routing_provider
        )
        params.update(extra_params)

        # Create the LiteLLM model with appropriate parameters
        return self.llm_provider.get_agno_model()

    def _format_context(self, ctx: ChatContext) -> str:
        """Format the context information for the Agno Team"""
        # Format context information to include with the query
        formatted_context = f"""
CONTEXT:
Project ID: {ctx.project_id}
Project Name: {ctx.project_name}
Agent ID: {ctx.curr_agent_id}

Chat History:
{chr(10).join(ctx.history)}

Additional Context:
{ctx.additional_context}

NODE IDs: {', '.join(ctx.node_ids) if ctx.node_ids else 'None'}

QUERY:
{ctx.query}
"""
        return formatted_context

    def _extract_citations(self, response: str) -> List[str]:
        """Extract citations from the response"""
        citations = []
        citation_marker = "###Citations###"

        if citation_marker in response:
            # Extract the citations part
            citation_section = response.split(citation_marker)[-1].strip()
            # Split by commas and strip whitespace
            citations = [cite.strip() for cite in citation_section.split(",")]
            # Remove empty citations
            citations = [cite for cite in citations if cite]

        return citations

    def _process_tool_calls(self, team_response: str) -> List[ToolCallResponse]:
        """Process and extract tool calls from the team response"""
        tool_calls = []

        # Currently, Agno doesn't provide direct access to tool calls in a structured format
        # We would need to extract this information from the response if needed
        # For now, returning an empty list
        return tool_calls

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run synchronously in a blocking manner, return entire response at once"""
        try:
            formatted_context = self._format_context(ctx)
            # Use the get_response method from Agent class
            response = self.team_agent.run(formatted_context)

            # Extract citations and tool calls
            citations = self._extract_citations(str(response.content))

            return ChatAgentResponse(
                response=str(response.content), citations=citations, tool_calls=[]
            )
        except Exception as e:
            logger.error(f"Error in TeamChatAgent.run: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                response=f"Error running team chat agent: {str(e)}",
                citations=[],
                tool_calls=[],
            )

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, yield response piece by piece"""
        try:
            formatted_context = self._format_context(ctx)

            # Use Agno's streaming capability via the Agent class
            response_generator = self.team_agent.run(formatted_context, stream=True)

            accumulated_response = ""
            for chunk in response_generator:
                accumulated_response += str(chunk.content)

                # Extract citations from accumulated response (may be incomplete)
                citations = self._extract_citations(accumulated_response)

                # Yield current chunk as ChatAgentResponse
                yield ChatAgentResponse(
                    response=str(
                        chunk.content
                    ),  # Just the new chunk, not the accumulated response
                    citations=citations,
                    tool_calls=[],  # Tool calls are not incrementally available
                )

            # Final yield with complete response and all citations
            final_citations = self._extract_citations(accumulated_response)
            tool_calls = self._process_tool_calls(accumulated_response)

            yield ChatAgentResponse(
                response="",  # Empty response to indicate completion
                citations=final_citations,
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Error in TeamChatAgent.run_stream: {str(e)}", exc_info=True)
            yield ChatAgentResponse(
                response=f"Error in streaming team response: {str(e)}",
                citations=[],
                tool_calls=[],
            )
