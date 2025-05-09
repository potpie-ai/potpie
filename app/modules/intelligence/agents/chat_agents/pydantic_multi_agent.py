import re
from typing import List, AsyncGenerator, Dict, Any, Optional, Literal
from enum import Enum

from .tool_helpers import (
    get_tool_call_info_content,
    get_tool_response_message,
    get_tool_result_info_content,
    get_tool_run_message,
)
from app.modules.intelligence.provider.provider_service import (
    ProviderService,
)
from .crewai_agent import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger

from ..chat_agent import (
    ChatAgent,
    ChatAgentResponse,
    ChatContext,
    ToolCallEventType,
    ToolCallResponse,
)

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ModelResponse,
    TextPart,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = setup_logger(__name__)


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    FINALIZER = "finalizer"


class MultiAgentState(BaseModel):
    """State model for the multi-agent system"""

    current_role: AgentRole = Field(
        default=AgentRole.PLANNER, description="The current role in the agent workflow"
    )
    plan: str = Field(default="", description="The plan created by the planner agent")
    execution_results: str = Field(
        default="", description="The execution results from the executor agent"
    )
    verification_results: str = Field(
        default="", description="The verification results from the verifier agent"
    )
    final_output: str = Field(
        default="", description="The final output from the finalizer agent"
    )
    iterations: int = Field(
        default=0, description="Number of iterations through the verification loop"
    )
    verified: bool = Field(
        default=False, description="Whether the results have been verified"
    )


def create_prompt_for_role(
    role: AgentRole, state: MultiAgentState, task_description: str, ctx: ChatContext
) -> str:
    """Create a role-specific prompt based on the current state"""
    if ctx.node_ids is None:
        ctx.node_ids = []
    if isinstance(ctx.node_ids, str):
        ctx.node_ids = [ctx.node_ids]
    base_prompt = f"""
    
    CONTEXT:
    User Task: 
    {ctx.query}
    
    Project ID: {ctx.project_id}
    Node IDs: {" ,".join(ctx.node_ids)}
    Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

    Additional Context:
    {ctx.additional_context if ctx.additional_context != "" else "no additional context"}
    
    Current State:
    - Iterations: {state.iterations}
    """

    if role == AgentRole.PLANNER:
        return (
            base_prompt
            + """
        Your role is to create a detailed plan to accomplish the task.
        
        Break down the task into clear steps that can be executed systematically.
        Consider edge cases and potential challenges in your plan. Don't go deep into execution.
        Next steps will take care of executing your plan and verifying results based on your plan
        
        Create one more list of key requirements mentioned in the task, this will be used to verify
        the output against requirements. List of requirements should be exhaustive and cover all aspects of the task.
        The requirements should be in the form of a list. Add references from the task description to each requirement.
        
        Output a comprehensive plan with numbered steps
        
        IMPORTANT: Only gather information and create a high level plan, do not execute task. That will be taken
        care in the next steps
        
        SAY HELLO At the end of requirements list
        """
        )

    elif role == AgentRole.EXECUTOR:
        return (
            base_prompt
            + f"""
        The plan to execute is:
        {state.plan}
        
        Your role is to execute each step in the plan methodically.
        Show your work for each step.
        If you encounter issues with the plan, try to overcome them and note what adjustments were needed.
        
        Output the execution results for each step in the plan.
        """
        )

    elif role == AgentRole.VERIFIER:
        return (
            base_prompt
            + f"""
        The plan was:
        {state.plan}
        
        The execution results are:
        {state.execution_results}
        
        Your role is to verify if the execution successfully completed the task.
        Check if all steps in the plan were executed correctly.
        Identify any issues, errors, or missing elements in the execution.
        All the key requirements should be verified.

        Output:
        - Your answer **must** start with `VERIFIED: YES` if everything is correct, or `VERIFIED: NO` if there are problems.
        - After that, provide a short justification or feedback for the executor if you answer NO.
        - Mention the key requrements that weren't met

        Example:
        VERIFIED: YES
        All implementation steps were successful and match the plan.

        or

        VERIFIED: NO
        Step 3 did not complete as described. Please fix...
        """
        )

    elif role == AgentRole.FINALIZER:
        return (
            base_prompt
            + f"""
        The plan was:
        {state.plan}
        
        The execution results are:
        {state.execution_results}
        
        The verification results are:
        {state.verification_results}
        
        Your role is to produce the final output based on the execution results.
        Synthesize the information into a coherent, well-structured response.
        Format your response appropriately (markdown for text, proper code blocks for code, etc.)
        
        Output the final response that directly addresses the original task.
        """
        )

    return base_prompt


def create_system_prompt_for_role(
    role: AgentRole, config: AgentConfig, task_desc: str
) -> str:
    """Create a role-specific system prompt"""
    base_prompt = f"""
        Role: {config.role}
        Goal: {config.goal}
        Backstory: {config.backstory}
        
        CURRENT CONTEXT AND AGENT TASK OVERVIEW:
        {task_desc}
        
        You are part of a larger AI Workflow. Focus on your role in answering user query or executing user task
        """

    role_specific = {
        AgentRole.PLANNER: "You are a strategic planner. Break down complex tasks into actionable steps.",
        AgentRole.EXECUTOR: "You are a methodical executor. Follow plans precisely and show your work.",
        AgentRole.VERIFIER: "You are a critical verifier. Rigorously check work against requirements.",
        AgentRole.FINALIZER: "You are a skilled synthesizer. Create polished final outputs from execution results.",
    }

    return f"{base_prompt}\n\n{role_specific[role]}"


class PydanticMultiAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ):
        """Initialize the multi-agent system with configuration and tools"""
        self.tasks = config.tasks
        self.max_iter = config.max_iter
        self.config = config

        # tool name can't have spaces for pydantic agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Convert tools to pydantic-ai format
        self.pydantic_tools = [
            Tool(
                name=tool.name,
                description=tool.description,
                function=tool.func,  # type: ignore
            )
            for tool in tools
        ]

        # Create the provider model
        self.model = llm_provider.get_pydantic_model()

        # Create individual agents for each role
        self.agents = {}

        # Initialize the state
        self.state = MultiAgentState()

    def _init_agents(self, ctx: ChatContext):
        # Create individual agents for each role
        self.agents = {
            AgentRole.PLANNER: self._create_agent_for_role(AgentRole.PLANNER, ctx),
            AgentRole.EXECUTOR: self._create_agent_for_role(AgentRole.EXECUTOR, ctx),
            AgentRole.VERIFIER: self._create_agent_for_role(AgentRole.VERIFIER, ctx),
            AgentRole.FINALIZER: self._create_agent_for_role(AgentRole.FINALIZER, ctx),
        }

    def _create_agent_for_role(self, role: AgentRole, ctx: ChatContext) -> Agent:
        """Create a specialized agent for a specific role"""
        return Agent(
            model=self.model,
            tools=self.pydantic_tools,
            instructions=create_system_prompt_for_role(
                role, self.config, self._create_task_description(ctx)
            ),
            output_type=str,
            retries=3,
            defer_model_check=True,
            model_settings={"parallel_tool_calls": True, "max_tokens": 14000},
        )

    def _get_next_role(self, state: MultiAgentState) -> AgentRole:
        """Determine the next role based on the current state"""
        if state.current_role == AgentRole.PLANNER:
            return AgentRole.EXECUTOR
        elif state.current_role == AgentRole.EXECUTOR:
            return AgentRole.VERIFIER
        elif state.current_role == AgentRole.VERIFIER:
            if state.verified or state.iterations >= self.max_iter:
                return AgentRole.FINALIZER
            else:
                return AgentRole.EXECUTOR
        else:
            # If we're at the finalizer, we're done
            return AgentRole.FINALIZER

    def _create_task_description(
        self,
        ctx: ChatContext,
    ) -> str:
        """Create a task description from task configuration"""
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        return f"""
                CONTEXT:
                User Query: {ctx.query}
                Project ID: {ctx.project_id}
                Node IDs: {" ,".join(ctx.node_ids)}
                Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}

                Additional Context:
                {ctx.additional_context if ctx.additional_context != "" else "no additional context"}

                With above information execute the following task: 
                {ctx.query}
            """

    async def _execute_current_role(
        self, role: AgentRole, task: str, ctx: ChatContext
    ) -> str:
        """Execute the agent for the current role and return its result"""
        agent = self.agents[role]
        prompt = create_prompt_for_role(role, self.state, task, ctx)
        resp = await agent.run(
            user_prompt=prompt,
            message_history=[
                ModelResponse([TextPart(content=msg)]) for msg in ctx.history
            ],
        )
        return resp.data

    async def _update_state_for_role(self, role: AgentRole, result: str) -> None:
        """Update the state based on the role and result"""
        if role == AgentRole.PLANNER:
            self.state.plan = result
        elif role == AgentRole.EXECUTOR:
            self.state.execution_results = result
        elif role == AgentRole.VERIFIER:
            self.state.verification_results = result
            self.state.iterations += 1
            # Strict detection for "VERIFIED: YES" at the start of response
            match = re.match(r"^\s*VERIFIED:\s*YES", result.strip(), re.IGNORECASE)
            self.state.verified = bool(match)
        elif role == AgentRole.FINALIZER:
            self.state.final_output = result

        # Update to the next role
        self.state.current_role = self._get_next_role(self.state)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        logger.info("Running pydantic-ai multi-agent")
        try:
            task = self._create_task_description(ctx)
            self._init_agents(ctx)

            # Reset state
            self.state = MultiAgentState()

            # Execute the workflow
            while True:
                current_role = self.state.current_role
                result = await self._execute_current_role(current_role, task, ctx)
                await self._update_state_for_role(current_role, result)

                # If we're at the finalizer and have output, we're done
                if current_role == AgentRole.FINALIZER:
                    break

            return ChatAgentResponse(
                response=self.state.final_output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run asynchronously, stream the multi-agent process"""
        logger.info("Running pydantic-ai multi-agent stream")
        task = self._create_task_description(ctx)

        # Reset state
        self.state = MultiAgentState()

        try:
            self._init_agents(ctx)
            # Execute each role sequentially, streaming output
            while True:
                current_role = self.state.current_role
                current_agent = self.agents[current_role]
                current_prompt = create_prompt_for_role(
                    current_role, self.state, task, ctx
                )

                # Yield transition message
                if self.state.iterations > 0 and current_role == AgentRole.EXECUTOR:
                    yield ChatAgentResponse(
                        response=f"\n\n--- Verification indicates changes needed. Iterating execution (iteration {self.state.iterations})... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.PLANNER:
                    yield ChatAgentResponse(
                        response=f"\n\n--- Starting planning phase... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.EXECUTOR and self.state.iterations == 0:
                    yield ChatAgentResponse(
                        response="\n\n--- Planning complete. Moving to execution... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.VERIFIER:
                    yield ChatAgentResponse(
                        response=f"\n\n--- Execution complete. Moving to verification (iteration {self.state.iterations + 1})... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.FINALIZER:
                    yield ChatAgentResponse(
                        response="\n\n--- Verification complete. Finalizing output... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )

                # Stream the agent execution
                result = ""
                async with current_agent.iter(
                    user_prompt=current_prompt,
                    message_history=[
                        ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                    ],
                ) as agent_run:
                    async for node in agent_run:
                        if Agent.is_model_request_node(node):
                            # Stream model responses
                            async with node.stream(agent_run.ctx) as request_stream:
                                async for event in request_stream:
                                    if isinstance(event, PartStartEvent) and isinstance(
                                        event.part, TextPart
                                    ):
                                        yield ChatAgentResponse(
                                            response=f"[{current_role.value.upper()}] {event.part.content}",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        result += event.part.content
                                    if isinstance(event, PartDeltaEvent) and isinstance(
                                        event.delta, TextPartDelta
                                    ):
                                        yield ChatAgentResponse(
                                            thought=event.delta.content_delta,
                                            # response="",
                                            response=event.delta.content_delta,
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        result += event.delta.content_delta

                        elif Agent.is_call_tools_node(node):
                            # Stream tool calls
                            async with node.stream(agent_run.ctx) as handle_stream:
                                async for event in handle_stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        yield ChatAgentResponse(
                                            response="",
                                            tool_calls=[
                                                ToolCallResponse(
                                                    call_id=event.part.tool_call_id
                                                    or "",
                                                    event_type=ToolCallEventType.CALL,
                                                    tool_name=event.part.tool_name,
                                                    tool_response=get_tool_run_message(
                                                        event.part.tool_name
                                                    ),
                                                    tool_call_details={
                                                        "summary": get_tool_call_info_content(
                                                            event.part.tool_name,
                                                            event.part.args_as_dict(),
                                                        )
                                                    },
                                                )
                                            ],
                                            citations=[],
                                        )
                                    if isinstance(event, FunctionToolResultEvent):
                                        yield ChatAgentResponse(
                                            response="",
                                            tool_calls=[
                                                ToolCallResponse(
                                                    call_id=event.result.tool_call_id
                                                    or "",
                                                    event_type=ToolCallEventType.RESULT,
                                                    tool_name=event.result.tool_name
                                                    or "unknown tool",
                                                    tool_response=get_tool_response_message(
                                                        event.result.tool_name
                                                        or "unknown tool"
                                                    ),
                                                    tool_call_details={
                                                        "summary": get_tool_result_info_content(
                                                            event.result.tool_name
                                                            or "unknown tool",
                                                            event.result.content,
                                                        )
                                                    },
                                                )
                                            ],
                                            citations=[],
                                        )

                # Update state with the result
                await self._update_state_for_role(current_role, result)

                # If we're done, yield the final message and exit
                if current_role == AgentRole.FINALIZER:
                    yield ChatAgentResponse(
                        response="\n\n# Finalization complete. Final output:\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                    yield ChatAgentResponse(
                        response=self.state.final_output,
                        tool_calls=[],
                        citations=[],
                    )
                    break

        except Exception as e:
            logger.error(f"Error in run_stream method: {str(e)}", exc_info=True)
            raise Exception from e
