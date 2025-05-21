import functools
import re
from typing import List, AsyncGenerator, Dict, Any, Optional, Literal
from enum import Enum
from pydantic_ai.usage import UsageLimits

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


def handle_exception(tool_func):
    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in tool function: {e}")
            return f"An internal error occurred. Please try again later."

    return wrapper


class AgentStage(str, Enum):
    CONTEXT = "context"
    SOLUTION_GEN = "solution_gen"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"


class Solution(BaseModel):
    """Model for a potential solution approach"""

    description: str = Field(..., description="Description of the solution approach")
    pros: List[str] = Field(
        default_factory=list, description="Advantages of this approach"
    )
    cons: List[str] = Field(
        default_factory=list, description="Disadvantages of this approach"
    )
    complexity: int = Field(default=5, description="Complexity rating (1-10)")


class Requirement(BaseModel):
    """Model for a single requirement"""

    description: str = Field(..., description="Description of the requirement")
    verified: bool = Field(
        default=False, description="Whether this requirement has been verified"
    )


class AgentState(BaseModel):
    """State model for the streamlined agent system"""

    current_stage: AgentStage = Field(
        default=AgentStage.CONTEXT,
        description="The current stage in the agent workflow",
    )
    context_analysis: str = Field(
        default="", description="Analysis of the issue and context"
    )
    issue_identified: str = Field(
        default="", description="The specific issue identified"
    )
    solution_approaches: List[Solution] = Field(
        default_factory=list, description="Different approaches to solve the issue"
    )
    chosen_solution_index: Optional[int] = Field(
        default=None, description="Index of the chosen solution"
    )
    implementation: str = Field(default="", description="The implemented solution")
    verification_result: str = Field(
        default="", description="Verification of the solution"
    )
    requirements: List[Requirement] = Field(
        default_factory=list, description="List of requirements to verify"
    )
    final_output: str = Field(default="", description="The final verified solution")


def create_system_prompt_for_stage(
    stage: AgentStage, config: AgentConfig, task_desc: str, ctx: ChatContext
) -> str:
    if ctx.node_ids is None:
        ctx.node_ids = []
    if isinstance(ctx.node_ids, str):
        ctx.node_ids = [ctx.node_ids]

    """Create a stage-specific system prompt"""
    base_prompt = f"""
    Role: {config.role}
    Goal: {config.goal}
    Backstory: {config.backstory}     
    
    Project ID: {ctx.project_id}
    Node IDs: {" ,".join(ctx.node_ids)}
    Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}
    
    TIPS TO HANDLE THE OVERALL CODEBASE TASKS: FOLLOW THE INSTRUCTIONS WHEREVER APPLICABLE
    {task_desc}
    
    You are part of a streamlined 4-step agent workflow:
    1. CONTEXT GATHERING & ISSUE IDENTIFICATION: Understand the codebase, explore the issue, identify the root cause
    2. SOLUTION GENERATION: Generate 3-4 different approaches to fix the issue
    3. IMPLEMENTATION: Choose the best approach and implement it
    4. VERIFICATION: Verify the solution works and meets all requirements
    
    IMPORTANT: Consider your current step and avoid unnecessary computation. Keep your answers focused on your current stage.
    """

    stage_specific = {
        AgentStage.CONTEXT: """
        You are currently in the CONTEXT GATHERING & ISSUE IDENTIFICATION stage.
        
        Your tasks:
        1. Thoroughly understand the issue by exploring the codebase
        2. Identify the specific root cause of the problem
        3. Document relevant code snippets, classes, and relationships
        4. Determine what requirements will need to be met for a successful fix
        
        Use available tools extensively to explore the codebase. Build a comprehensive understanding
        before proceeding. Look for patterns, inconsistencies, bugs, or missed requirements.
        
        Your output should include:
        1. A clear statement of the issue and its root cause
        2. Relevant code context (files, classes, functions)
        3. A list of 3-5 clear requirements that any solution must meet to be considered successful
        
        Be thorough but efficient. Focus on understanding the problem deeply.
        """,
        AgentStage.SOLUTION_GEN: """
        You are currently in the SOLUTION GENERATION stage.
        
        Your tasks:
        1. Generate 3-4 different approaches to solve the identified issue
        2. For each approach, describe:
           - The implementation method in detail
           - Pros and cons
           - Complexity rating (1-10)
           - How it addresses the requirements
        3. Don't implement any solution yet, just describe the approaches
        
        Be creative but practical. Consider different paradigms, techniques, or algorithms.
        Evaluate each approach objectively, considering factors like performance, maintainability,
        compatibility, and adherence to project patterns.
        
        Your output should be well-structured with each approach clearly delineated.
        """,
        AgentStage.IMPLEMENTATION: """
        You are currently in the IMPLEMENTATION stage.
        
        Your tasks:
        1. Review the solution approaches and select the best one
        2. Implement the chosen solution in detail
        3. Show all your work and reasoning
        4. Use appropriate tools to modify files and create patches
        
        Focus on quality implementation. Follow best practices, maintain code style consistency,
        and ensure your solution fully addresses the identified issue.
        
        Use the tools at your disposal to implement the changes. Make sure to:
        1. Generate proper patch diffs
        2. Update any related documentation/tests if needed
        3. Thoroughly explain your implementation
        4. Maintain a CURRENT CONTEXT DATA section with relevant information
        
        Be methodical and detail-oriented in your implementation.
        """,
        AgentStage.VERIFICATION: """
        You are currently in the VERIFICATION stage.
        
        Your tasks:
        1. Verify the implementation against each requirement
        2. Use verification tools to test the solution
        3. Make any necessary adjustments
        4. Produce the final verified solution
        
        Be rigorous in your verification. Check each requirement methodically and ensure the
        solution is robust. Use the VerifyDiffTool to validate your changes.
        
        Your output must include:
        1. A verification of each requirement (success or failure)
        2. Any adjustments made to fix verification issues
        3. The final verified solution
        
        The final output should be clean, professional, and ready for submission.
        
        IMPORTANT: Run the VerifyDiffTool to ensure the solution is correct. Fix any issues that
        arise and verify again until successful.
        """,
    }

    return f"{base_prompt}\n\n{stage_specific[stage]}"


def create_prompt_for_stage(stage: AgentStage, state: AgentState, query: str) -> str:
    """Create a stage-specific prompt based on the current state"""
    base_prompt = f"""
    Full Problem statement:
    {query}
    """

    if stage == AgentStage.CONTEXT:
        return (
            base_prompt
            + """
        You are in the CONTEXT GATHERING & ISSUE IDENTIFICATION stage.
        
        Use tools to explore the codebase and understand the issue in depth. Determine:
        1. What is the specific problem we need to solve?
        2. Which files/classes/functions are involved?
        3. What's the root cause of the issue?
        4. What requirements must a solution satisfy?
        
        Be thorough in your exploration. Use knowledge graph tools, file exploration, and code inspection.
        
        Format your response:
        
        ## Issue Analysis
        [Detailed analysis of the issue, including relevant code context]
        
        ## Root Cause
        [Clear statement of the root cause]
        
        ## Requirements
        [List of 3-5 specific, verifiable requirements that any solution must meet]
        
        ## Relevant Code Context
        [Important code snippets, file locations, and relationships]
        """
        )

    elif stage == AgentStage.SOLUTION_GEN:
        return (
            base_prompt
            + f"""
        You are in the SOLUTION GENERATION stage.
        
        Based on the previous analysis:
        
        Issue Identified: {state.issue_identified}
        
        Context Analysis:
        {state.context_analysis}
        
        Requirements:
        {[f"- {req.description}" for req in state.requirements]}
        
        Generate 3-4 different approaches to solve this issue. For each approach:
        1. Describe the implementation method in detail
        2. List pros and cons
        3. Rate complexity (1-10)
        4. Explain how it addresses each requirement
        
        Don't implement any solution yet - just describe the possible approaches.
        
        Format your response:
        
        ## Approach 1: [Name of Approach]
        [Detailed description]
        
        **Pros:**
        - [List of pros]
        
        **Cons:**
        - [List of cons]
        
        **Complexity:** [1-10]
        
        **Requirements Addressed:**
        [How this addresses each requirement]
        
        [Repeat for each approach]
        
        Be creative but practical in your suggestions.
        """
        )

    elif stage == AgentStage.IMPLEMENTATION:
        # Format the solution approaches for easy reading
        solution_text = ""
        for i, sol in enumerate(state.solution_approaches):
            solution_text += f"""
            ## Approach {i+1}: {sol.description.split(':')[0] if ':' in sol.description else sol.description[:20]+'...'}
            
            **Pros:** {', '.join(sol.pros)}
            **Cons:** {', '.join(sol.cons)}
            **Complexity:** {sol.complexity}/10
            
            """

        return (
            base_prompt
            + f"""
        You are in the IMPLEMENTATION stage.
        
        Based on the previous analysis:
        
        Issue Identified: {state.issue_identified}
        
        Context Analysis:
        {state.context_analysis}
        
        Requirements:
        {[f"- {req.description}" for req in state.requirements]}
        
        Solution Approaches:
        {solution_text}
        
        Choose the best approach from those listed above and implement it. Consider the trade-offs
        carefully. Then provide a detailed implementation of your chosen solution.
        
        Use appropriate tools to:
        1. Get current code if needed
        2. Generate patches for changes
        3. Test your implementation
        
        Format your response:
        
        ## Selected Approach
        [Which approach you selected and why]
        
        ## Implementation
        [Detailed implementation steps and code changes]
        
        ## Patch Diff
        [Include patch diffs for all changed files]
        
        ## CURRENT CONTEXT DATA
        [Important references and information]
        
        Be methodical and thorough in your implementation.
        """
        )

    elif stage == AgentStage.VERIFICATION:
        return (
            base_prompt
            + f"""
        You are in the VERIFICATION stage.
        
        Based on the implementation:
        
        Issue Identified: {state.issue_identified}
        
        Implementation Summary:
        {state.implementation[:300]}... [truncated]
        
        Requirements to Verify:
        {[f"- {req.description}" for req in state.requirements]}
        
        Verify that the implementation meets all requirements. Use the VerifyDiffTool to ensure
        the changes are valid. Fix any issues that arise during verification.
        
        For each requirement:
        1. Verify if it has been met
        2. Provide evidence of verification
        3. Mark as VERIFIED or NOT VERIFIED
        
        If any requirements are not met, fix the implementation and verify again.
        
        Format your response:
        
        ## Verification Results
        
        ### Requirement 1
        [Description of requirement]
        **Status:** VERIFIED / NOT VERIFIED
        **Evidence:** [Evidence of verification]
        
        [Repeat for each requirement]
        
        ## Final Solution
        [The complete verified solution, including any final adjustments]
        
        IMPORTANT: Run the VerifyDiffTool to ensure the solution is correct. The final solution
        must pass all verification checks.
        """
        )

    return base_prompt


def extract_requirements(analysis_text: str) -> List[Requirement]:
    """
    Extract requirements from the context analysis output
    """
    # Look for a requirements section using different possible formats
    section_regex = r"(?:^|\n)(?:#+\s*)?(?:requirements|key requirements|solution requirements)(?:\s*:)?(?:\s*\n+)([\s\S]+?)(?=\n\s*#+|\Z)"

    m = re.search(section_regex, analysis_text, flags=re.IGNORECASE)
    section_text = ""
    if m:
        section_text = m.group(1)
    else:
        # Try finding numbered or bulleted lists
        list_regex = r"(?:^|\n)(?:\d+\.|[-*])\s+.+(?:\n(?:\d+\.|[-*])\s+.+)*"
        list_match = re.search(list_regex, analysis_text)
        if list_match:
            section_text = list_match.group(0)

    # Extract list items
    list_items = re.findall(r"(?:^|\n)(?:\d+\.|[-*])\s+(.+)", section_text)

    # If still no requirements found, try a more aggressive approach
    if not list_items:
        # Look for sentences with requirements keywords
        sentences = re.findall(
            r"(?:^|\n| )(?:[^.\n])+must[^.\n]+\.", analysis_text, re.IGNORECASE
        )
        sentences.extend(
            re.findall(
                r"(?:^|\n| )(?:[^.\n])+should[^.\n]+\.", analysis_text, re.IGNORECASE
            )
        )
        sentences.extend(
            re.findall(
                r"(?:^|\n| )(?:[^.\n])+needs to[^.\n]+\.", analysis_text, re.IGNORECASE
            )
        )
        list_items = [s.strip() for s in sentences]

    # Create requirement objects
    requirements = []
    for item in list_items:
        item = item.strip()
        if item and len(item) > 10:  # Minimum length to filter out noise
            requirements.append(Requirement(description=item))

    # Ensure we have at least one requirement
    if not requirements:
        requirements.append(
            Requirement(
                description="Fix the identified issue while maintaining code integrity"
            )
        )

    # Add verification requirement
    verify_req_exists = any("verify" in req.description.lower() for req in requirements)
    if not verify_req_exists:
        requirements.append(
            Requirement(
                description="Use VerifyDiffTool to ensure the solution is valid and passes all verification checks"
            )
        )

    return requirements


def extract_solutions(solutions_text: str) -> List[Solution]:
    """
    Extract solution approaches from the generation output
    """
    # Look for approach sections
    approach_sections = re.split(r"(?:^|\n)#+\s*Approach \d+(?::)?", solutions_text)

    # Remove the text before the first approach (if any)
    if approach_sections and not approach_sections[0].strip().startswith("Approach"):
        approach_sections = approach_sections[1:]

    solutions = []
    for section in approach_sections:
        if not section.strip():
            continue

        # Extract description (first paragraph)
        description = section.strip().split("\n")[0]
        if not description:
            description = "Approach " + str(len(solutions) + 1)

        # Extract pros
        pros = []
        pros_section = re.search(
            r"(?:\*\*)?Pros(?:\*\*)?(?::|(?:\s*\n+))([\s\S]+?)(?=\*\*|\n\n|$)",
            section,
            re.IGNORECASE,
        )
        if pros_section:
            pros_text = pros_section.group(1)
            pros = re.findall(r"(?:^|\n)(?:[-*]|\d+\.)\s+(.+)", pros_text)
            if not pros:
                # Try splitting by newlines if no bullet points
                pros = [p.strip() for p in pros_text.split("\n") if p.strip()]

        # Extract cons
        cons = []
        cons_section = re.search(
            r"(?:\*\*)?Cons(?:\*\*)?(?::|(?:\s*\n+))([\s\S]+?)(?=\*\*|\n\n|$)",
            section,
            re.IGNORECASE,
        )
        if cons_section:
            cons_text = cons_section.group(1)
            cons = re.findall(r"(?:^|\n)(?:[-*]|\d+\.)\s+(.+)", cons_text)
            if not cons:
                # Try splitting by newlines if no bullet points
                cons = [c.strip() for c in cons_text.split("\n") if c.strip()]

        # Extract complexity
        complexity = 5  # Default middle value
        complexity_match = re.search(
            r"(?:\*\*)?Complexity(?:\*\*)?(?::|(?:\s*\n+))\s*(\d+)",
            section,
            re.IGNORECASE,
        )
        if complexity_match:
            try:
                complexity = int(complexity_match.group(1))
                # Ensure within range
                complexity = max(1, min(10, complexity))
            except ValueError:
                pass

        solutions.append(
            Solution(
                description=description,
                pros=pros if pros else ["Good approach"],
                cons=cons if cons else ["No specific disadvantages noted"],
                complexity=complexity,
            )
        )

    # If no solutions were extracted, create a default one
    if not solutions:
        solutions.append(
            Solution(
                description="Direct fix of the identified issue",
                pros=["Straightforward solution", "Directly addresses the problem"],
                cons=["May not be the most optimal approach"],
                complexity=5,
            )
        )

    return solutions


class PydanticStreamlinedAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ):
        """Initialize the streamlined agent system with configuration and tools"""
        self.tasks = config.tasks
        self.config = config

        # tool name can't have spaces for pydantic agents
        for i, tool in enumerate(tools):
            tools[i].name = re.sub(r" ", "", tool.name)

        # Convert tools to pydantic-ai format
        self.pydantic_tools = [
            Tool(
                name=tool.name,
                description=tool.description,
                function=handle_exception(tool.func),
            )
            for tool in tools
        ]

        # Create the provider model
        self.model = llm_provider.get_pydantic_model()

        # Agents will be initialized in _init_agents
        self.agents = {}

        # Initialize the state
        self.state = AgentState()

    def _init_agents(self, ctx: ChatContext):
        # Create individual agents for each stage
        self.agents = {
            AgentStage.CONTEXT: self._create_agent_for_stage(AgentStage.CONTEXT, ctx),
            AgentStage.SOLUTION_GEN: self._create_agent_for_stage(
                AgentStage.SOLUTION_GEN, ctx
            ),
            AgentStage.IMPLEMENTATION: self._create_agent_for_stage(
                AgentStage.IMPLEMENTATION, ctx
            ),
            AgentStage.VERIFICATION: self._create_agent_for_stage(
                AgentStage.VERIFICATION, ctx
            ),
        }

    def _create_agent_for_stage(self, stage: AgentStage, ctx: ChatContext) -> Agent:
        """Create a specialized agent for a specific stage"""
        return Agent(
            model=self.model,
            tools=self.pydantic_tools,
            instructions=create_system_prompt_for_stage(
                stage, self.config, self.tasks[0].description, ctx
            ),
            output_type=str,
            retries=2,
            defer_model_check=True,
            end_strategy="exhaustive",
        )

    def _get_next_stage(self, current_stage: AgentStage) -> AgentStage:
        """Determine the next stage in the workflow"""
        stage_sequence = {
            AgentStage.CONTEXT: AgentStage.SOLUTION_GEN,
            AgentStage.SOLUTION_GEN: AgentStage.IMPLEMENTATION,
            AgentStage.IMPLEMENTATION: AgentStage.VERIFICATION,
            AgentStage.VERIFICATION: AgentStage.VERIFICATION,  # Terminal state
        }
        return stage_sequence[current_stage]

    async def _update_state_for_stage(self, stage: AgentStage, result: str) -> None:
        """Update the state based on the stage and result"""
        if stage == AgentStage.CONTEXT:
            self.state.context_analysis = result
            # Extract the issue statement - usually the first paragraph after "Root Cause"
            root_cause_match = re.search(
                r"(?:^|\n)#+\s*Root Cause\s*\n+(.+?)(?:\n\n|\n#+|$)",
                result,
                re.IGNORECASE,
            )
            if root_cause_match:
                self.state.issue_identified = root_cause_match.group(1).strip()
            else:
                # Fallback: take the first paragraph that seems to describe an issue
                issue_statements = re.findall(
                    r"(?:issue|problem|bug|error|fail)(?:[^.!?]+[.!?])",
                    result,
                    re.IGNORECASE,
                )
                if issue_statements:
                    self.state.issue_identified = issue_statements[0].strip()
                else:
                    self.state.issue_identified = (
                        "Specific issue needs addressing based on code analysis"
                    )

            # Extract requirements
            self.state.requirements = extract_requirements(result)

        elif stage == AgentStage.SOLUTION_GEN:
            # Extract solutions
            self.state.solution_approaches = extract_solutions(result)

        elif stage == AgentStage.IMPLEMENTATION:
            self.state.implementation = result
            # Try to extract which solution was chosen
            chosen_approach_match = re.search(
                r"(?:^|\n)#+\s*Selected Approach\s*\n+(.+?)(?:\n\n|\n#+|$)",
                result,
                re.IGNORECASE,
            )
            if chosen_approach_match:
                chosen_text = chosen_approach_match.group(1).lower()
                # Try to match to one of our approaches by looking at words
                for i, approach in enumerate(self.state.solution_approaches):
                    approach_key_words = re.sub(
                        r"[^\w\s]", "", approach.description.lower()
                    ).split()
                    # If several key words match, assume this is the chosen approach
                    matches = sum(
                        1
                        for word in approach_key_words
                        if word in chosen_text and len(word) > 3
                    )
                    if matches >= 2 or f"approach {i+1}" in chosen_text:
                        self.state.chosen_solution_index = i
                        break

        elif stage == AgentStage.VERIFICATION:
            self.state.verification_result = result

            # Extract final output - usually the last section of the verification
            final_solution_match = re.search(
                r"(?:^|\n)#+\s*Final Solution\s*\n+([\s\S]+)(?:\Z)",
                result,
                re.IGNORECASE,
            )
            if final_solution_match:
                self.state.final_output = final_solution_match.group(1).strip()
            else:
                # Fallback: use the entire verification result
                self.state.final_output = result

            # Check which requirements were verified
            for i, req in enumerate(self.state.requirements):
                # Look for verification status for this requirement
                req_pattern = (
                    re.escape(req.description[:30])
                    if len(req.description) > 30
                    else re.escape(req.description)
                )
                req_section = re.search(
                    rf"Requirement\s+{i+1}|{req_pattern}(.+?)(?:Requirement|##|\Z)",
                    result,
                    re.IGNORECASE | re.DOTALL,
                )

                if req_section:
                    req_text = req_section.group(0)
                    verified = (
                        re.search(
                            r"VERIFIED|✅|Success|Passes", req_text, re.IGNORECASE
                        )
                        is not None
                    )
                    self.state.requirements[i].verified = verified

        # Update to the next stage
        self.state.current_stage = self._get_next_stage(self.state.current_stage)

    def _create_task_description(self, ctx: ChatContext) -> str:
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
        """

    async def _execute_current_stage(
        self, stage: AgentStage, task: str, ctx: ChatContext
    ) -> str:
        """Execute the agent for the current stage and return its result"""
        agent = self.agents[stage]
        prompt = create_prompt_for_stage(stage, self.state, ctx.query)
        resp = await agent.run(
            user_prompt=prompt,
            message_history=[
                ModelResponse([TextPart(content=msg)]) for msg in ctx.history
            ],
        )
        return resp.data

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Main execution flow"""
        logger.info("Running streamlined pydantic-ai agent")
        try:
            task = self._create_task_description(ctx)
            self._init_agents(ctx)

            # Reset state
            self.state = AgentState()

            # Execute each stage in sequence
            while (
                self.state.current_stage != AgentStage.VERIFICATION
                or not self.state.final_output
            ):
                current_stage = self.state.current_stage
                result = await self._execute_current_stage(current_stage, task, ctx)
                await self._update_state_for_stage(current_stage, result)

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
        """Run asynchronously, stream the agent process"""
        logger.info("Running streamlined pydantic-ai agent stream")
        task = self._create_task_description(ctx)

        # Reset state
        self.state = AgentState()

        try:
            self._init_agents(ctx)

            # Execute each stage sequentially, streaming output
            while (
                self.state.current_stage != AgentStage.VERIFICATION
                or not self.state.final_output
            ):
                current_stage = self.state.current_stage
                current_agent = self.agents[current_stage]
                current_prompt = create_prompt_for_stage(
                    current_stage, self.state, ctx.query
                )

                # Yield transition message based on stage
                if current_stage == AgentStage.CONTEXT:
                    yield ChatAgentResponse(
                        response=f"\n\n--- STAGE 1: Gathering Context & Identifying Issue... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_stage == AgentStage.SOLUTION_GEN:
                    yield ChatAgentResponse(
                        response=f"\n\n--- STAGE 2: Generating Solution Approaches... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_stage == AgentStage.IMPLEMENTATION:
                    yield ChatAgentResponse(
                        response=f"\n\n--- STAGE 3: Implementing Best Solution... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_stage == AgentStage.VERIFICATION:
                    yield ChatAgentResponse(
                        response=f"\n\n--- STAGE 4: Verifying Solution... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )

                # Stream the agent execution
                result = ""
                async with current_agent.iter(
                    user_prompt=current_prompt,
                    usage_limits=UsageLimits(request_limit=100),
                    # message_history=[
                    #     ModelResponse([TextPart(content=msg)]) for msg in ctx.history
                    # ],
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
                                            response=f"[{current_stage.value.upper()}] {event.part.content}",
                                            tool_calls=[],
                                            citations=[],
                                        )
                                        result += event.part.content
                                    if isinstance(event, PartDeltaEvent) and isinstance(
                                        event.delta, TextPartDelta
                                    ):
                                        yield ChatAgentResponse(
                                            thought=event.delta.content_delta,
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
                await self._update_state_for_stage(current_stage, result)

            # Yield the final message
            yield ChatAgentResponse(
                response="\n\n# Task Complete. Final Solution:\n\n",
                tool_calls=[],
                citations=[],
            )
            yield ChatAgentResponse(
                response=self.state.final_output,
                tool_calls=[],
                citations=[],
            )

        except Exception as e:
            logger.error(f"Error in run_stream method: {str(e)}", exc_info=True)
            raise Exception from e
