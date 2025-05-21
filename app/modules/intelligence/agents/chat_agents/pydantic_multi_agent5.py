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


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    FINALIZER = "finalizer"
    FIXER = "fixer"  # New FIXER role


class Requirement(BaseModel):
    """Model for a single requirement"""

    description: str = Field(..., description="Description of the requirement")
    verified: bool = Field(
        default=False, description="Whether this requirement has been verified"
    )


class PlanStep(BaseModel):
    """Model for a single step in the plan"""

    description: str = Field(..., description="Description of the step")
    executed: bool = Field(
        default=False, description="Whether this step has been executed"
    )
    execution_result: str = Field(
        default="", description="Result of executing this step"
    )
    # New fields to support looping
    is_loop: bool = Field(
        default=False, description="Whether this step should be executed in a loop"
    )
    max_iterations: int = Field(
        default=1, description="Maximum number of times to execute this step"
    )
    current_iteration: int = Field(
        default=0, description="Current iteration of this loop step"
    )
    iteration_results: List[str] = Field(
        default_factory=list, description="Results from each iteration of this step"
    )
    best_iteration: Optional[int] = Field(
        default=None, description="Index of the best iteration (if applicable)"
    )


class MultiAgentState(BaseModel):
    """State model for the multi-agent system"""

    current_role: AgentRole = Field(
        default=AgentRole.PLANNER, description="The current role in the agent workflow"
    )
    plan: str = Field(default="", description="The plan created by the planner agent")
    plan_steps: List[PlanStep] = Field(
        default_factory=list, description="List of steps in the plan to execute"
    )
    current_step_index: int = Field(
        default=0, description="Index of the current step being executed"
    )
    execution_results: List[str] = Field(
        default_factory=list,
        description="The execution results from the executor agent",
    )
    verification_results: str = Field(
        default="", description="The verification results from the verifier agent"
    )
    fixer_results: str = Field(
        default="", description="The fix results from the fixer agent"
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


def create_system_prompt_for_role(
    role: AgentRole, config: AgentConfig, task_desc: str, ctx: ChatContext
) -> str:
    if ctx.node_ids is None:
        ctx.node_ids = []
    if isinstance(ctx.node_ids, str):
        ctx.node_ids = [ctx.node_ids]
    """Create a role-specific system prompt"""
    base_prompt = f"""
        Role: {config.role}
        Goal: {config.goal}
        Backstory: {config.backstory}     
        
        Project ID: {ctx.project_id}
        Node IDs: {" ,".join(ctx.node_ids)}
        Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}
        
        
        TIPS TO HANDLE THE OVERALL CODEBASE TASKS: FOLLOW THE INSTRUCTIONS WHEREVER APPLICABLE
        {task_desc}
        
        
        You are part of a larger AI Workflow. Focus on your role ({role}) in answering user query or executing user task
        AI Workflow has many agents and follow the below ROUTE
        1. PLANNER AGENT
        2. EXECUTOR AGENT
        3. VERIFIER AGENT
        4. FIXER AGENT (this agent's response is routed to verifier)
        5. FINALIZER (Once all requirements are verified)
        
        IMPORTANT: Consider the ROUTE above and avoid unnecessary computation, keep your answers concise and DONT STRAY AWAY FROM YOUR ROLE
        """

    role_specific = {
        AgentRole.PLANNER: "You are a strategic planner. Break down complex tasks into actionable steps. Also identify key requirements that will be verified separately. You are part of the larger workflow, make sure you respond with necessary information/results for next steps to use",
        AgentRole.EXECUTOR: "You are a methodical executor. Follow plans precisely and show your work for the specific step you're assigned. You are part of the larger workflow, make sure you respond with necessary information/results for next steps to use",
        AgentRole.FIXER: "You are a specialized fixer. Focus specifically on fixing one requirement issue at a time based on verification feedback. Don't try to solve everything, just fix what's needed. You are part of the larger workflow, make sure you respond with necessary information/results for next steps to use",
        AgentRole.VERIFIER: "You are a critical verifier. Focus on verifying one requirement at a time thoroughly against execution or fixer results. You are part of the larger workflow, make sure you respond with necessary information/results for next steps to use",
        AgentRole.FINALIZER: "You are a skilled synthesizer. Create polished final outputs from execution results. You are part of the larger workflow, make sure you respond with necessary information/results for next steps to use",
    }

    return f"{base_prompt}\n\n{role_specific[role]}"


def create_prompt_for_role(role: AgentRole, state: MultiAgentState, query: str) -> str:
    """Create a role-specific prompt based on the current state, with loop awareness"""

    base_prompt = f"""
    
    Full Problem statement:
    {query}
    
    Current State:
    - Iterations: {state.iterations}
    """

    # Update executor prompt for loop steps
    if role == AgentRole.EXECUTOR:
        steps_status = _format_steps_status(state)
        current_step = (
            state.plan_steps[state.current_step_index]
            if state.plan_steps and state.current_step_index < len(state.plan_steps)
            else None
        )

        # Special handling for loop steps
        loop_context = ""
        if current_step and current_step.is_loop:
            loop_context = f"""
            LOOP STEP INFORMATION:
            - This is iteration {current_step.current_iteration + 1} of {current_step.max_iterations} for this step
            - Previous iterations: {len(current_step.iteration_results)}
            """

            # Add results from previous iterations if available
            if current_step.iteration_results:
                loop_context += "\nRESULTS FROM PREVIOUS ITERATIONS:\n"
                for i, result in enumerate(current_step.iteration_results):
                    loop_context += f"\n--- Iteration {i+1} result ---\n{result[:6000]}...(truncated)\n"

                loop_context += "\nFor this iteration, try a different approach than the previous ones to explore alternative solutions."

        return (
            base_prompt
            + f"""
        The overall plan to execute is:
        {state.plan}
        
        Steps Status:
        {steps_status}
        
        Your role is to execute ONE SPECIFIC STEP of the plan methodically.
        
        CURRENT STEP TO EXECUTE: {current_step.description if current_step else "No step defined yet"}
        
        {loop_context}
        
        Execute ONLY this specific step. Show your detailed work for this step only.
        If you encounter issues with this step, try to overcome them and note what adjustments were needed.
        
        IMPORTANT:
        1. Focus ONLY on executing the CURRENT step above - do not try to execute other steps
        2. Use the tools at your disposal to assist with the execution
        3. You are supposed to have full understanding of the codebase and the task
        4. Fetch any code and related context from the codebase that you need for this specific step
        5. Show all your work in detail for this step
        6. Use Knowledge graph tools to explore codebase and all the code relations in the project. Maintain node_ids and summary in the CURRENT CONTEXT DATA
        7. Use CURRENT CONTEXT DATA from previous step to access relevant data in the project, build CURRENT CONTEXT DATA and update it in current iteration
        8. Reuse existing helpers in the project, explore the project and helper files and reuse the helpers and already existing functions/classes etc 
                       
        Previous execution results that might be helpful context:
        {[f'''
        
        {result}
        
        ''' for result in (state.execution_results[-2:] if len(state.execution_results) > 1 else state.execution_results)]}
        
        Format your response:
        1. Start with a brief explanation of what you will do for this step
        2. Then provide the detailed implementation/solution for this step only
        3. End with a brief summary of what was accomplished in this step
        4. Provider a section (CURRENT CONTEXT DATA:), Update it from previous execution if any. Provide references and important results like code snippets, line numbers, node_ids, file references and how to access them for next steps to use. This is IMPORTANT.
            Just add whatever data was used in current iteration of execution step to the CURRENT CONTEXT DATA, don't go and fetch data unnecessarily, reuse data from previous CURRENT CONTEXT DATA
        5. Include key necessary CURRENT CONTEXT DATA from previous execution section aswell when generating it. Make sure you mention info in the repo and info in file changes (FileChangesManager) seperately
        
        IF you need to generate a patch diff, use the GeneratePatchDiff Tool to create it, don't send too many lines when changes are small. Keep adequate amount of context lines in the patch diff.
        
        IMPORTANT: Include CURRENT CONTEXT DATA in your response. CURRENT CONTEXT DATA should also carry all the relevant information from previous execution, Include the execution result also at the end of the current context data (basically attach the current progress of the update)
        IMPORTANT: Use nodes from knowledge graph tools extensively to understand what a piece of class/function etc really does, you can use GetCodeanddocstringFromProbableNodeName to fetch this data, use this recursively if needed
        """
        )

    # Other roles remain the same
    elif role == AgentRole.PLANNER:
        return (
            base_prompt
            + """
            Your role is to create a detailed plan to accomplish the task.
            
            Understand the task, use tools at your disposal to gather initial information regarding the task and then start with the plan
            Break down the task into clear steps that can be executed systematically.
            Consider edge cases and potential challenges in your plan. Don't go deep into execution.
            Next steps will take care of executing your plan and verifying results based on your plan
            Make sure you include verification steps and output requirements from the user query in the user
            If the user has explicitly mentioned steps include them in the list.
            Plan shouldn't be very strict, this is a overview multi step plan so that we can implement the solution in stages
            Keep room for exploration of codebase, understanding the context better, searching alternatives and verifying and comparing solutions
            
            IMPORTANT: Always follow the PLANNING GUIDELINES provided. Make sure the task analysis is done deeply with exploration of codebase and tool calls
            before start with the plan and requirements

            IMPORTANT: Create a separate list of key requirements mentioned in the task. Each requirement should be
            clearly stated and must be verifiable. Number these requirements. These will be used to verify
            the output against requirements one by one. List of requirements should be EXHAUSTIVE and cover all aspects of the task.
            Include all the output formatting and do and don'ts in the user task as part of the requirements to verify later.
            Each requirement can span a few sentence but should be a single line (single point in the list)
            
            IMPORTANT: All you requirements should be summarized within 4 points. 4 is max no. of requirements. Group similar requirements
            in the same line in same point (don't create sub-lists / sub points)

            NOTE ON VERIFICATION: For each requirement you list, a separate verification step will be performed.
            The verifier agent's response for EACH requirement will start STRICTLY with 'VERIFIED: YES' or 'VERIFIED: NO',
            followed by specific feedback SOLELY about that single requirement. Your requirements should be specific
            enough to allow for this clear YES/NO verification per requirement.
            
            IMPORTANT: DO NOT have sub requirements or sub-steps in the requirements list.
            It can be few sentences long, but should not contain sub list. Make each point as clear and explicit as possible.
            
            IMPORTANT: YOUR PLAN MUST BE IN A CLEAR NUMBERED LIST FORMAT. Each step will be executed one at a time. Each step will be prefixed with a number, Nudge the plan to use knowledge graphs and nodes effectively to explore the codebase and maintain node_ids for reference
            
            LOOPING STEPS:
            You can now designate certain steps to be executed multiple times in a loop by adding [LOOP:N] to the step description,
            where N is the number of times to repeat the step. This is useful for exploring multiple solutions or approaches.
            Example: "3. [LOOP:3] Implement different sorting algorithms to compare performance"
            This will cause the step to be executed 3 times, with results from each iteration preserved.
            
            Use loop steps when:
            1. You want to explore multiple alternative approaches to a problem
            2. You want to compare different implementations or algorithms
            3. You need to iteratively refine a solution through multiple attempts
            
            Example output:
--------------
            # Task Analysis
Before diving into a plan for creating a user registration API endpoint, let's understand what this involves. We need a secure, robust RESTful API that handles user registration with proper validation, error handling, and follows best practices. This requires careful consideration of data validation, security measures, response formatting, and documentation.

# Detailed Plan
1. Design the API endpoint structure and request/response formats
2. Set up the development environment and project structure
3. [LOOP:3] Implement and test different data validation approaches
... so on ...
10. Deploy the endpoint to staging environment
11. Conduct security testing
12. Prepare for production deployment

# Key Requirements

1. Fixes the string, make sure the class implements toString(), print Methods, 

2. The API must validate user input including: email in proper format (e.g., user@domain.com), password minimum 8 characters with at least one uppercase letter, one lowercase letter, one number, and one special character.

3. Make sure the output requirement is met, it should use the exact format as Result:'''{...}''' and only have utf-8 encoded characters

... so on

----------------------
    Note above plan is example for the format - Content of each step and key requirement must follow guidelines mentioned above the example
    Output a comprehensive plan with numbered steps followed by Key Requirements section with numbered steps. Make sure to include all the task requirements in the Key Requirements section.
    Especially focus on the requirements that are critical for the task and need to be verified later like output formatting, do's and don'ts etc.
    IMPORTANT: EXPLORE THE PROBLEM AND OUTPUT THE PLAN AND KEY REQUIREMENTS. DO NOT SOLVE THE PROBLEM. CLUB SIMILAR REQUIREMENTS TOGETHER, DON"T CREATE TOO MANY UNNECCESSARY REQUIREMENTS. DO NOT HAVE SUB-POINTS IN THE REQUIREMENTS (IT WON"T BE PARSED PROPERLY).
    IMPORTANT: Last step in the plan has to output the final result expected by the user task
    Do not update test files for the given changes. Only fix the issue
    
    In Planning, always include in steps to understand the purpose of codebase (use webresults for this too). Check if the issue is expected behaviour
            """
        )

    # Other roles remain unchanged from original implementation
    elif role == AgentRole.FIXER:
        # Special prompt for FIXER role (unchanged)
        return (
            base_prompt
            + f"""
        The original plan was:
        {state.plan}
        
        Requirements Status:
        {_format_requirements_status(state)}
        
        The last few execution results are (older to latest):
        {[f'''
        
        {result}
        
        ''' for result in (state.execution_results[-2:] if len(state.execution_results) > 1 else state.execution_results)]}
        This above result is the current state of the code and the output of the last execution
        
        Your role is to FIX a SPECIFIC ISSUE that was identified during verification.
        
        CURRENT REQUIREMENT WITH ISSUE: {state.requirements[state.current_requirement_index].description}
        
        Verification feedback on this requirement:
        {state.verification_results}
        {"" if state.verification_results == "" else "IMPORTANT: Fix the above issue from the verification feedback. This is the main cause for failure"}
        
        IMPORTANT INSTRUCTIONS:
        1. Focus ONLY on fixing the SPECIFIC issue mentioned in the verification feedback
        2. Use any tool calls necessary to fix the issue or asked in the verification feedback
        3. Reuse existing helpers in the project, explore the project and helper files and reuse the helpers and already existing functions/classes etc 
        
        Format your response:
        1. Start with a brief explanation of what needs to be fixed
        2. Then provide the fixed implementation/solution or All the tool calls and responses/results
        3. End with a brief explanation of how your fix addresses the verification issue
        4. Verify your fix
        4. Provider a section (CURRENT CONTEXT DATA:), Update it from previous execution if any. Provide references and important results like code snippets, line numbers, node_ids, file references and how to access them for next steps to use. This is IMPORTANT.
            Just add whatever data was used in current iteration of execution step to the CURRENT CONTEXT DATA, don't go and fetch data unnecessarily, reuse data from previous CURRENT CONTEXT DATA
        5. Include key necessary CURRENT CONTEXT DATA from previous execution section aswell when generating it. Make sure you mention info in the repo and info in file changes (FileChangesManager) seperately
        
        
        IMPORTANT:
        Use the tools at your disposal to assist with the execution.
        You are supposed to have full understanding of the codebase and the task. Exhaustively explore the codebase and the task to get the best results.
        This step is critical for the overall success of the task so make sure to do it thoroughly and take your time.
        Fetch the code and all the related context from the codebase.
        
        IF you need to generate a patch diff, use the GeneratePatchDiff Tool to create it, don't send too many lines when changes are small. Keep adequate context in the patch diff.
        
        IMPORTANT: Make sure the fix doesn't violate one of the previous requirements, ALWAYS RESPOND WITH THE FIXED RESULT AFTER FIXING
        IMPORTANT: Include CURRENT CONTEXT DATA in your response. CURRENT CONTEXT DATA should also carry all the relevant information from previous execution, Include the execution result also at the end of the current context data (basically attach the current progress of the update)
        """
        )

    elif role == AgentRole.VERIFIER:
        # Original verifier prompt (unchanged)
        current_req = state.requirements[state.current_requirement_index]
        return (
            base_prompt
            + f"""    
            
            {"The fixer results for the current requirement are:" if state.fixer_results else ""}
            {state.fixer_results if state.fixer_results else ""}
            
            Requirements Status:
            {_format_requirements_status(state)}
            
            The last few execution results are (older to latest):
            {[f'''
        
            {result}
        
        ''' for result in (state.execution_results[-2:] if len(state.execution_results) > 1 else state.execution_results)]}
            This above result is the current state of the code and the output of the last execution
            
            Your role is to verify if the {"execution" if not state.fixer_results else "fixer"} successfully completed the CURRENT requirement:
            
            CURRENT REQUIREMENT: {current_req.description}
            
            Check thoroughly if this specific requirement was met in the {"execution" if not state.fixer_results else "fixer"} results.
            Check user task/query for examples and further instructions regarding this requirement.
            
            IMPORTANT: Your response format is critical. You MUST strictly follow this format:
            1. Verify if the current requirement is met
            2. Provide specific reasoning for your verification decision
            3. Respond with "VERIFIED: YES" or "VERIFIED: NO", this has to a part of your response (make sure it's the exact string)
            4. Focus ONLY on the current requirement - do not try to verify other requirements
            
            Examples of proper verification responses:
            
            Checking endpoints, finding api documentation...
            VERIFIED: YES
            The requirement was successfully implemented because the code contains a proper RESTful API endpoint for user registration at '/api/users' which accepts POST requests with username, email, and password fields.
            
            OR
            
            Checking endpoints, finding api documentation...
            VERIFIED: NO
            This requirement is not met because the email validation is missing. The current implementation only checks if an email is present but doesn't validate its format. The code needs to include email format validation using a regex pattern or validation library.
            
            IMPORTANT: Respond with "VERIFIED: YES" or "VERIFIED: NO", this perticular string has to part of your response. DO NOT TRY TO FIX THE PROBLEM, ONLY DO YOUR ROLE. FIXING PART WILL BE TAKEN CARE OF BY FIXER AGENT
            IMPORTANT: If fixer has changed anything make sure it still satisfies all the previously verified requirements. We Don't want fixer to cause changes that impact previous requirements
            
            IMPORTANT: Never change the result after it has been verified as "VERIFIED: YES". Output the result as it is, Make sure to return the exact verified result
            
            """
        )

    elif role == AgentRole.FINALIZER:
        # Original finalizer prompt (unchanged)
        return (
            base_prompt
            + f"""
        The plan was:
        {state.plan}
        
        Steps Status:
        {_format_steps_status(state)}
        
        The execution result from the last step is:
        {state.execution_results[-1]}
        
        Requirements Status:
        {_format_requirements_status(state)}
        
        Your role is to produce the final output based on the execution results.
        Synthesize the information into a coherent, well-structured response.
        Format your response appropriately (markdown for text, proper code blocks for code, etc.)
        
        IMPORTANT: If the final output is a patch diff, make sure you don't change any information in the patch.
        Copy the patch as it is.
        
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


def _format_steps_status(state: MultiAgentState) -> str:
    """Format the steps status for display in prompts, with loop information"""
    if not state.plan_steps:
        return "No steps have been identified yet."

    status_lines = []
    for i, step in enumerate(state.plan_steps):
        if step.is_loop:
            # Special formatting for loop steps
            loop_status = f"({step.current_iteration}/{step.max_iterations} iterations)"
            status = (
                "✅" if step.executed else "⏳" if step.current_iteration > 0 else "❌"
            )
            current = " (CURRENT)" if i == state.current_step_index else ""
            status_lines.append(
                f"{i+1}. {status} [LOOP] {step.description} {loop_status}{current}"
            )
        else:
            # Regular step formatting
            status = "✅" if step.executed else "❌"
            current = " (CURRENT)" if i == state.current_step_index else ""
            status_lines.append(f"{i+1}. {status} {step.description}{current}")

    return "\n".join(status_lines)


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

    clean_reqs.append(
        Requirement(
            description="""Use VerifyDiffTool to make the result pass through for all the files in the hunk, 
            Run this tool again even if it has been previously run in the history. 
            Tool HAS to be used don't assume it will pass through, The result HAS to pass the VerifyDiffTool test with valid = True. 
            Fix any issues that arise from the test using FileChangesManager tools and generate diffs. 
            Make sure the final diff is exactly the one that was verified. Verify diff at the end everytime before responding as verified
            Respond with the exact final result that was verified at the end. Stop here (you can't edit the result anymore)""",
        )
    )

    return clean_reqs


def extract_plan_steps(plan_text: str) -> List[PlanStep]:
    """
    Extract individual steps from the planner's output, including loop steps and subpoints.
    - Look for the plan section by finding title or numbered list
    - Extract numbered steps from the plan, including their subpoints
    - Identify loop steps by looking for [LOOP:X] or similar pattern
    """
    # First try to locate the "Detailed Plan" or "Plan" section
    plan_section_regex = (
        r"(?P<start>(?:^|\n)\s*(?:#+\s*)?(detailed plan|plan)\s*:?[\n\r]+)"
        r"(?P<body>[\s\S]+?)(?=\n\s*#+|\Z)"  # Until next markdown heading or end of text
    )
    m = re.search(plan_section_regex, plan_text, flags=re.IGNORECASE)
    section_text = ""
    if m:
        section_text = m.group("body")
    else:
        # Fallback: look for first numbered list
        m2 = re.search(
            r"(?:^|\n)((?:[ \t]*\d+\.[ \t]+.+(?:\n+[ \t]*\*[ \t]+.+)*\n*)+)", plan_text
        )
        if m2:
            section_text = m2.group(1)
        else:
            # Last resort: use the whole text
            section_text = plan_text

    # Extract numbered steps with their subpoints
    steps = []

    # Split the text by numbered steps pattern
    step_blocks = re.split(r"(?=^[ \t]*\d+\.?[ \t]+)", section_text, flags=re.MULTILINE)

    for block in step_blocks:
        if not block.strip():
            continue

        # Extract the step number and initial description
        step_match = re.match(r"^[ \t]*(\d+\.?)[ \t]+(.+)$", block, re.MULTILINE)
        if not step_match:
            continue

        # Get the complete step content including all subpoints
        step_content = block.strip()

        # Look for loop indicator in the step description
        # Formats supported: [LOOP:3], [LOOP: 3], (LOOP:3), (LOOP: 3), LOOP(3), etc.
        loop_match = re.search(
            r"\[LOOP[:\s]*(\d+)\]|\(LOOP[:\s]*(\d+)\)|LOOP\((\d+)\)",
            step_content,
            re.IGNORECASE,
        )

        if loop_match:
            # Extract max iterations from the matched group (whichever group matched)
            max_iter = int(next(g for g in loop_match.groups() if g is not None))

            # Remove the loop indicator from the description
            clean_desc = re.sub(
                r"\[LOOP[:\s]*\d+\]|\(LOOP[:\s]*\d+\)|LOOP\(\d+\)", "", step_content
            ).strip()

            steps.append(
                PlanStep(description=clean_desc, is_loop=True, max_iterations=max_iter)
            )
        else:
            steps.append(PlanStep(description=step_content))

    # If no steps were found, try a more general approach
    if not steps:
        # Just break by line and look for lines that might be steps
        lines = [line.strip() for line in section_text.splitlines() if line.strip()]
        for line in lines:
            # Skip lines that are likely headings or other non-step content
            if line.startswith("#") or len(line) < 10:
                continue
            steps.append(PlanStep(description=line))

    return steps


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
                function=handle_exception(tool.func),
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
            AgentRole.FIXER: self._create_agent_for_role(AgentRole.FIXER, ctx),
            AgentRole.VERIFIER: self._create_agent_for_role(AgentRole.VERIFIER, ctx),
            AgentRole.FINALIZER: self._create_agent_for_role(AgentRole.FINALIZER, ctx),
        }

    def _create_agent_for_role(self, role: AgentRole, ctx: ChatContext) -> Agent:
        """Create a specialized agent for a specific role"""
        return Agent(
            model=self.model,
            tools=self.pydantic_tools,
            instructions=create_system_prompt_for_role(
                role, self.config, self.tasks[0].description, ctx
            ),
            output_type=str,
            retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
            # model_settings={"parallel_tool_calls": True},
        )

    # 3. Update the _get_next_role method to handle looping behavior
    def _get_next_role(self, state: MultiAgentState) -> AgentRole:
        """Determine the next role based on the current state, with support for looping steps"""
        if state.current_role == AgentRole.PLANNER:
            return AgentRole.EXECUTOR
        elif state.current_role == AgentRole.EXECUTOR:
            # Check if we're on a loop step and need to iterate
            if (
                state.plan_steps
                and state.current_step_index < len(state.plan_steps)
                and state.plan_steps[state.current_step_index].is_loop
            ):

                current_step = state.plan_steps[state.current_step_index]

                # Increment the current iteration counter
                current_step.current_iteration += 1

                # Check if we need to loop again
                if current_step.current_iteration < current_step.max_iterations:
                    # Store the current result in iteration_results but don't mark as executed yet
                    if current_step.execution_result:
                        current_step.iteration_results.append(
                            current_step.execution_result
                        )
                        # Reset execution_result for the next iteration
                        current_step.execution_result = ""

                    # Stay in EXECUTOR role but note we're looping
                    return AgentRole.EXECUTOR
                else:
                    # We've completed all iterations, store final result and proceed
                    if current_step.execution_result:
                        current_step.iteration_results.append(
                            current_step.execution_result
                        )

                    # Mark as executed now that all iterations are complete
                    current_step.executed = True

                    # Check if there are more steps to execute
                    if state.current_step_index < len(state.plan_steps) - 1:
                        # Move to next step but stay in executor role
                        state.current_step_index += 1
                        return AgentRole.EXECUTOR
                    else:
                        # All steps executed, move to verifier
                        return AgentRole.VERIFIER
            else:
                # Not a loop step, use original logic
                if (
                    state.plan_steps
                    and state.current_step_index < len(state.plan_steps) - 1
                ):
                    # More steps to execute - stay in executor role but move to next step
                    # First mark the current step as executed
                    if state.plan_steps and state.current_step_index < len(
                        state.plan_steps
                    ):
                        state.plan_steps[state.current_step_index].executed = True

                    state.current_step_index += 1
                    return AgentRole.EXECUTOR
                else:
                    # Mark the last step as executed
                    if state.plan_steps and state.current_step_index < len(
                        state.plan_steps
                    ):
                        state.plan_steps[state.current_step_index].executed = True

                    # All steps executed, move to verifier
                    return AgentRole.VERIFIER

        elif state.current_role == AgentRole.VERIFIER:
            # Original verifier logic
            current_req_verified = False
            if state.requirements and state.current_requirement_index < len(
                state.requirements
            ):
                current_req_verified = state.requirements[
                    state.current_requirement_index
                ].verified

            if current_req_verified:
                if (
                    state.requirements
                    and state.current_requirement_index < len(state.requirements) - 1
                ):
                    # More requirements to check
                    state.current_requirement_index += 1
                    state.fixer_results = ""
                    return AgentRole.VERIFIER
                else:
                    # All requirements verified, move to finalizer
                    return AgentRole.FINALIZER
            elif state.iterations >= self.max_iter:
                # Max iterations reached, go to finalizer
                return AgentRole.FINALIZER
            else:
                # Current requirement not verified, route to FIXER
                return AgentRole.FIXER

        elif state.current_role == AgentRole.FIXER:
            # After fixing, go back to verifier
            return AgentRole.VERIFIER
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
            """

    async def _execute_current_role(
        self, role: AgentRole, task: str, ctx: ChatContext
    ) -> str:
        """Execute the agent for the current role and return its result"""
        agent = self.agents[role]
        prompt = create_prompt_for_role(role, self.state, ctx.query)
        resp = await agent.run(
            user_prompt=prompt,
            message_history=[
                ModelResponse([TextPart(content=msg)]) for msg in ctx.history
            ],
        )
        return resp.data

    async def _update_state_for_role(self, role: AgentRole, result: str) -> None:
        """Update the state based on the role and result, with support for looping steps"""
        if role == AgentRole.PLANNER:
            # Original planner logic
            self.state.plan = result
            self.state.requirements = extract_requirements(result)
            self.state.plan_steps = extract_plan_steps(
                result
            )  # Now supports loop steps
            self.state.current_requirement_index = 0
            self.state.current_step_index = 0
            self.state.iterations = 0
            self.state.fixer_results = ""
            self.state.execution_results = []

        elif role == AgentRole.EXECUTOR:
            # Update the current step's execution status and result, with loop awareness
            if self.state.plan_steps and self.state.current_step_index < len(
                self.state.plan_steps
            ):
                current_step = self.state.plan_steps[self.state.current_step_index]

                # For loop steps, handle differently
                if current_step.is_loop:
                    # Don't mark as executed until all iterations are complete
                    current_step.execution_result = result

                    # Add the result to execution_results with a header showing which step and iteration
                    step_num = self.state.current_step_index + 1
                    total_steps = len(self.state.plan_steps)
                    iter_num = current_step.current_iteration + 1
                    max_iters = current_step.max_iterations

                    step_result = f"\n\n### Step {step_num}/{total_steps} (Iteration {iter_num}/{max_iters}): {current_step.description}\n{result}"
                    self.state.execution_results.append(step_result)
                else:
                    # Regular step, mark as executed
                    current_step.executed = True
                    current_step.execution_result = result

                    # Add the result to execution_results with a header
                    step_num = self.state.current_step_index + 1
                    total_steps = len(self.state.plan_steps)
                    step_result = f"\n\n### Step {step_num}/{total_steps}: {current_step.description}\n{result}"
                    self.state.execution_results.append(step_result)

            # Reset fixer results when executing a new step
            self.state.fixer_results = ""

        # Other roles remain unchanged
        elif role == AgentRole.FIXER:
            # Update fixer results
            self.state.fixer_results = result
            self.state.execution_results.append(
                f"\n\n### Fix for requirement {self.state.current_requirement_index + 1}:\n{result}"
            )

        elif role == AgentRole.VERIFIER:
            self.state.verification_results = result
            match = re.search(r"VERIFIED:\s*YES", result, re.IGNORECASE)
            current_req_verified = bool(match)

            if self.state.requirements and self.state.current_requirement_index < len(
                self.state.requirements
            ):
                self.state.requirements[
                    self.state.current_requirement_index
                ].verified = current_req_verified

            all_verified = (
                all(req.verified for req in self.state.requirements)
                if self.state.requirements
                else False
            )
            self.state.verified = all_verified

            if not current_req_verified:
                self.state.iterations += 1
            else:
                self.state.iterations = 0
                self.state.fixer_results = ""

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
                    current_role, self.state, ctx.query
                )

                # Yield transition message based on role and state
                if current_role == AgentRole.PLANNER:
                    yield ChatAgentResponse(
                        response=f"\n\n--- Starting planning phase... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.EXECUTOR:
                    if len(self.state.execution_results) == 0:
                        yield ChatAgentResponse(
                            response="\n\n--- Planning complete. Starting execution... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                    else:
                        # Show which step we're executing
                        step_index = self.state.current_step_index + 1
                        total_steps = len(self.state.plan_steps)
                        current_step = self.state.plan_steps[
                            self.state.current_step_index
                        ].description
                        yield ChatAgentResponse(
                            response=f"\n\n--- Executing step {step_index}/{total_steps}: {current_step}... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                elif current_role == AgentRole.FIXER:
                    req_index = self.state.current_requirement_index + 1
                    total_reqs = len(self.state.requirements)
                    yield ChatAgentResponse(
                        response=f"\n\n--- Verification failed. FIXER working on requirement {req_index}/{total_reqs}: {self.state.requirements[self.state.current_requirement_index].description} (iteration {self.state.iterations})... ---\n\n",
                        tool_calls=[],
                        citations=[],
                    )
                elif current_role == AgentRole.VERIFIER:
                    req_index = self.state.current_requirement_index + 1
                    total_reqs = len(self.state.requirements)
                    if self.state.iterations == 0 and not self.state.fixer_results:
                        yield ChatAgentResponse(
                            response=f"\n\n--- Execution complete. Verifying requirement {req_index}/{total_reqs}... ---\n\n",
                            tool_calls=[],
                            citations=[],
                        )
                    elif self.state.fixer_results:
                        yield ChatAgentResponse(
                            response=f"\n\n--- Fix applied. Re-verifying requirement {req_index}/{total_reqs}... ---\n\n",
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
