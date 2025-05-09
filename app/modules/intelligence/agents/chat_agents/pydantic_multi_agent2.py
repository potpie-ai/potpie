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


class Requirement(BaseModel):
    """Model for a single requirement"""

    description: str = Field(..., description="Description of the requirement")
    verified: bool = Field(
        default=False, description="Whether this requirement has been verified"
    )


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
    requirements: List[Requirement] = Field(
        default_factory=list, description="List of requirements to verify"
    )
    current_requirement_index: int = Field(
        default=0, description="Index of the current requirement being verified"
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
            Make sure you include verification steps and output requirements from the user query in the user
            If the user has explicitly mentioned steps include them in the list.
            Explore the problem and user provided task for few lines before starting with the plan

            IMPORTANT: Create a separate list of key requirements mentioned in the task. Each requirement should be
            clearly stated and must be verifiable. Number these requirements. These will be used to verify
            the output against requirements one by one. List of requirements should be EXHAUSTIVE and cover all aspects of the task.
            Include all the output formatting and do and don'ts in the user task as part of the requirements to verify later.
            Each requirement can span a few lines, but should be clear and concise with examples (include user provided examples if any)

            NOTE ON VERIFICATION: For each requirement you list, a separate verification step will be performed.
            The verifier agent's response for EACH requirement will start STRICTLY with 'VERIFIED: YES' or 'VERIFIED: NO',
            followed by specific feedback SOLELY about that single requirement. Your requirements should be specific
            enough to allow for this clear YES/NO verification per requirement.
            
            IMPORTANT: DO NOT have sub requirements or sub-steps in the requirements list.
            It can be few sentences long, but should not contain sub list. Make each point as clear and explicit as possible.
            
            Example output:
            # Task Analysis
Before diving into a plan for creating a user registration API endpoint, let's understand what this involves. We need a secure, robust RESTful API that handles user registration with proper validation, error handling, and follows best practices. This requires careful consideration of data validation, security measures, response formatting, and documentation.

# Detailed Plan
1. Design the API endpoint structure and request/response formats
2. Set up the development environment and project structure
3. Implement data models for user information
... so on ...
10. Deploy the endpoint to staging environment
11. Conduct security testing
12. Prepare for production deployment

# Key Requirements

1. The API endpoint must follow RESTful conventions with a POST request to "/api/v1/users" or "/api/v1/register" that accepts application/json content type.

2. The API must validate user input including: email in proper format (e.g., user@domain.com), password minimum 8 characters with at least one uppercase letter, one lowercase letter, one number, and one special character.

3. The implementation must hash passwords before storing them, using a modern algorithm like bcrypt or Argon2 with appropriate work factors.

4. The API must return appropriate HTTP status codes: 201 Created on success, 400 Bad Request for validation errors, 409 Conflict for existing users, and 500 for server errors.

... so on

Hello! I've prepared a detailed plan with clearly defined steps and an exhaustive list of specific, verifiable requirements as requested.

    Output a comprehensive plan with numbered steps followed by Key Requirements section with numbered steps. Make sure to include all the task requirements in the Key Requirements section.
    Especially focus on the requirements that are critical for the task and need to be verified later like output formatting, do's and don'ts etc.
    IMPORTANT: ONLY OUTPUT THE PLAN AND KEY REQUIREMENTS. DO NOT SOLVE THE PROBLEM
            """
        )

    elif role == AgentRole.EXECUTOR:
        return (
            base_prompt
            + f"""
        The plan to execute is:
        {state.plan}
        
        Requirements Status:
        {_format_requirements_status(state)}
        
        Your role is to execute each step in the plan methodically.
        Show your work for each step.
        If you encounter issues with the plan, try to overcome them and note what adjustments were needed.
        
        {"IMPORTANT: You are being called because the verification for a requirement FAILED. You need to fix the following specific issue:" if state.iterations > 0 else ""}
        {state.requirements[state.current_requirement_index].description if state.iterations > 0 and state.requirements else ""}
        
        {"Verification feedback: " + state.verification_results if state.iterations > 0 else ""}
        
        {"Focus specifically on addressing the issues mentioned in the verification feedback." if state.iterations > 0 else "Execute all steps in the plan thoroughly."}
        
        Output the execution results for each step in the plan.
        
        IMPORTANT: DO NOT VERIFY THE RESULTS. Your role is to execute the plan and show your work. VERIFICATION will be done separately later
        """
        )

    elif role == AgentRole.VERIFIER:
        # If this is not the first verification, focus on the current requirement
        if state.requirements:
            current_req = state.requirements[state.current_requirement_index]
            return (
                base_prompt
                + f"""    
            The execution results are:
            {state.execution_results}
            
            Requirements Status:
            {_format_requirements_status(state)}
            
            Your role is to verify if the execution successfully completed the CURRENT requirement:
            
            CURRENT REQUIREMENT: {current_req.description}
            
            Check thoroughly if this specific requirement was met in the execution results.
            Check user task/query for examples and further instructions regarding this requirement.
            
            IMPORTANT: Your response format is critical. You MUST strictly follow this format:
            1. Start your response with either "VERIFIED: YES" or "VERIFIED: NO"
            2. Then provide specific reasoning for your verification decision
            3. Focus ONLY on the current requirement - do not try to verify other requirements
            
            Examples of proper verification responses:
            
            VERIFIED: YES
            The requirement was successfully implemented because the code contains a proper RESTful API endpoint for user registration at '/api/users' which accepts POST requests with username, email, and password fields.
            
            OR
            
            VERIFIED: NO
            This requirement is not met because the email validation is missing. The current implementation only checks if an email is present but doesn't validate its format. The code needs to include email format validation using a regex pattern or validation library.
            """
            )
        else:
            # First verification pass, identify all requirements
            return (
                base_prompt
                + f"""
            The plan was:
            {state.plan}
            
            The execution results are:
            {state.execution_results}
            
            Requirements Status:
            {_format_requirements_status(state)}
            
            Your role is to verify if the execution successfully addressed the current requirement:
            
            CURRENT REQUIREMENT: {state.requirements[0].description if state.requirements else "No requirements defined yet"}
            
            Check thoroughly if this specific requirement was met in the execution results.
            
            IMPORTANT: Your response format is critical. You MUST strictly follow this format:
            1. Then provide specific reasoning for your verification decision
            2. respond with either "VERIFIED: YES" or "VERIFIED: NO"
            3. Focus ONLY on the current requirement - do not try to verify other requirements
            
            Examples of proper verification responses:
            
            VERIFIED: YES
            The requirement was successfully implemented because the code contains a proper RESTful API endpoint for user registration at '/api/users' which accepts POST requests with username, email, and password fields.
            
            OR
            
            VERIFIED: NO
            This requirement is not met because the email validation is missing. The current implementation only checks if an email is present but doesn't validate its format. The code needs to include email format validation using a regex pattern or validation library.
            
            
            IMPORTANT: Respond with "VERIFIED: YES" or "VERIFIED: NO". DO NOT TRY TO FIX THE PROBLEM, ONLY DO YOUR ROLE. FIXING PART WILL BE TAKEN CARE OF BY EXECUTOR AGENT
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
        
        Requirements Status:
        {_format_requirements_status(state)}
        
        Your role is to produce the final output based on the execution results.
        Synthesize the information into a coherent, well-structured response.
        Format your response appropriately (markdown for text, proper code blocks for code, etc.)
        
        Output the final response that directly addresses the original task.
        """
        )

    return base_prompt


def _format_requirements_status(state: MultiAgentState) -> str:
    """Format the requirements status for display in prompts"""
    if not state.requirements:
        return "No requirements have been identified yet."

    status_lines = []
    for i, req in enumerate(state.requirements):
        status = "✅" if req.verified else "❌"
        current = " (CURRENT)" if i == state.current_requirement_index else ""
        status_lines.append(f"{i+1}. {status} {req.description}{current}")

    return "\n".join(status_lines)


def create_system_prompt_for_role(
    role: AgentRole, config: AgentConfig, task_desc: str
) -> str:
    """Create a role-specific system prompt"""
    base_prompt = f"""
        Role: {config.role}
        Goal: {config.goal}
        Backstory: {config.backstory}
        
        {
           f'''CURRENT CONTEXT AND WAY TO HANDLE TASK:
           {task_desc}''' if role == AgentRole.EXECUTOR else ""
        }
        
        You are part of a larger AI Workflow. Focus on your role ({role}) in answering user query or executing user task
        """

    role_specific = {
        AgentRole.PLANNER: "You are a strategic planner. Break down complex tasks into actionable steps. Also identify key requirements that will be verified separately.",
        AgentRole.EXECUTOR: "You are a methodical executor. Follow plans precisely and show your work. Focus on fixing issues identified in verification.",
        AgentRole.VERIFIER: "You are a critical verifier. Focus on verifying one requirement at a time thoroughly against execution results.",
        AgentRole.FINALIZER: "You are a skilled synthesizer. Create polished final outputs from execution results.",
    }

    return f"{base_prompt}\n\n{role_specific[role]}"


def extract_requirements(plan_text: str) -> List[Requirement]:
    """
    Robustly extract individual requirements from the planner's output.
    - Looks for a heading "Key Requirements" or "Requirements" (case-insensitive)
    - Accepts numbered or bulleted lists
    - Cleans up whitespace, double prefixes, deduplicates, rejects empties
    """
    # 1. Locate the Key Requirements section (accepts "##", "#", or plain "Key Requirements" etc.)
    section_regex = (
        r"(?P<start>(?:^|\n)\s*(?:#+\s*)?(key requirements?|requirements?)\s*:?[\n\r]+)"
        r"(?P<body>[\s\S]+?)(?=\n\s*#+|\Z)"  # Until next markdown heading or end of text
    )
    m = re.search(section_regex, plan_text, flags=re.IGNORECASE)
    section_text = ""
    if m:
        section_text = m.group("body")
    else:
        # fallback: look for first bulleted/numbered list after the word "requirement"
        m2 = re.search(
            r"requirement[^\n]*\n+((?:[ \t]*[-*\d\.]+\s+.+\n*)+)",
            plan_text,
            flags=re.IGNORECASE,
        )
        if m2:
            section_text = m2.group(1)
        else:
            # fallback: the entire plan as list
            section_text = plan_text

    # 2. Extract list items: starts with "-", "*", or digit/period "1." etc.
    lines = re.findall(r"^[ \t]*(?:[-*]|\d+\.)\s+(.+)", section_text, re.MULTILINE)
    # fallback: non-list lines (if not detected above)
    if not lines:
        lines = [
            l.strip()
            for l in section_text.splitlines()
            if l.strip() and not l.strip().lower().startswith("key requirement")
        ]

    # 3. Deduplicate and clean
    clean_reqs = []
    seen = set()
    for l in lines:
        # Remove any leading numbering/bullet/colon accidentally captured
        l = re.sub(r"^(?:\d+\.\s*|[-*]\s*)+", "", l).strip(":-. \n\r\t")
        # Remove accidental "Requirement: ..." style
        l = re.sub(r"\b(?:requirement|requirements?)[:\s-]*", "", l, flags=re.I).strip()
        # Drop if empty
        if not l:
            continue
        # Drop if too short/generic
        if len(l) <= 3:
            continue
        # De-duplicate (ignore case, whitespace)
        sig = re.sub(r"\s+", " ", l).lower()
        if sig in seen:
            continue
        seen.add(sig)
        clean_reqs.append(Requirement(description=l))
    return clean_reqs


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
            # Check if the current requirement was verified
            current_req_verified = False
            if state.requirements and state.current_requirement_index < len(
                state.requirements
            ):
                current_req_verified = state.requirements[
                    state.current_requirement_index
                ].verified

            if current_req_verified:
                # Current requirement verified, check if there are more requirements to verify
                if (
                    state.requirements
                    and state.current_requirement_index < len(state.requirements) - 1
                ):
                    # More requirements to check - move to next requirement but stay in verifier role
                    state.current_requirement_index += 1
                    return AgentRole.VERIFIER
                else:
                    # All requirements verified, move to finalizer
                    return AgentRole.FINALIZER
            elif state.iterations >= self.max_iter:
                # Max iterations reached, go to finalizer
                return AgentRole.FINALIZER
            else:
                # Current requirement not verified, return to executor to fix it
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
            # Extract requirements from plan
            self.state.requirements = extract_requirements(result)
            # Reset current requirement index
            self.state.current_requirement_index = 0
            # Reset iteration counter
            self.state.iterations = 0

        elif role == AgentRole.EXECUTOR:
            self.state.execution_results = result

        elif role == AgentRole.VERIFIER:
            self.state.verification_results = result

            # Strict detection for "VERIFIED: YES" at the start of response
            match = re.match(r".*?\s*VERIFIED:\s*YES", result.strip(), re.IGNORECASE)
            current_req_verified = bool(match)

            # Update the current requirement's verification status
            if self.state.requirements and self.state.current_requirement_index < len(
                self.state.requirements
            ):
                self.state.requirements[
                    self.state.current_requirement_index
                ].verified = current_req_verified

            # Check if all requirements are verified
            all_verified = (
                all(req.verified for req in self.state.requirements)
                if self.state.requirements
                else False
            )
            self.state.verified = all_verified

            # Only increment iterations counter if we're going to re-do the same requirement
            if not current_req_verified:
                self.state.iterations += 1
            else:
                # Reset iterations when moving to a new requirement
                self.state.iterations = 0

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

                # Yield transition message based on role and state
                # For transition messages in run_stream method
                if current_role == AgentRole.PLANNER:
                    yield ChatAgentResponse(
                        response=f"\n\n--- Starting planning phase... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.EXECUTOR:
                    if self.state.iterations == 0:
                        yield ChatAgentResponse(
                            response="\n\n--- Planning complete. Starting execution... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                    else:
                        req_index = self.state.current_requirement_index + 1
                        total_reqs = len(self.state.requirements)
                        yield ChatAgentResponse(
                            response=f"\n\n--- Verification failed. Fixing requirement {req_index}/{total_reqs}: {self.state.requirements[self.state.current_requirement_index].description} (iteration {self.state.iterations})... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                elif current_role == AgentRole.VERIFIER:
                    req_index = self.state.current_requirement_index + 1
                    total_reqs = len(self.state.requirements)
                    if self.state.iterations == 0:
                        yield ChatAgentResponse(
                            response=f"\n\n--- Execution complete. Verifying requirement {req_index}/{total_reqs}... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                    else:
                        yield ChatAgentResponse(
                            response=f"\n\n--- Moving to next requirement. Verifying requirement {req_index}/{total_reqs}... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                elif current_role == AgentRole.FINALIZER:
                    yield ChatAgentResponse(
                        response="\n\n--- All requirements verified. Finalizing output... ---\n\n",
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
