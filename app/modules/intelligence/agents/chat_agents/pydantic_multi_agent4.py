import functools
import re
from typing import List, AsyncGenerator, Dict, Any, Optional, Literal
from enum import Enum

from app.modules.intelligence.tools.tool_service import ToolService

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
from pydantic_ai.usage import UsageLimits
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
        """

    role_specific = {
        AgentRole.PLANNER: "You are a strategic planner. Break down complex tasks into actionable steps. Also identify key requirements that will be verified separately.",
        AgentRole.EXECUTOR: "You are a methodical executor. Follow plans precisely and show your work for the specific step you're assigned.",
        AgentRole.FIXER: "You are a specialized fixer. Focus specifically on fixing one requirement issue at a time based on verification feedback. Don't try to solve everything, just fix what's needed.",
        AgentRole.VERIFIER: "You are a critical verifier. Focus on verifying one requirement at a time thoroughly against execution or fixer results.",
        AgentRole.FINALIZER: "You are a skilled synthesizer. Create polished final outputs from execution results.",
    }

    return f"{base_prompt}\n\n{role_specific[role]}"


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
        
        FUNCTIONAL TEST FUNDAMENTALS:
        1. Functional tests are NOT unit tests - they test end-to-end flows through multiple services
        2. Tests should be located in src/functionalTest/ directory (not src/test/)
        3. Each test verifies complete workflows across service boundaries
        4. Tests validate business requirements and user-facing functionality
        
        FUNCTIONAL TEST STRUCTURE:
        1. Test Data Setup:
           - Store test data as JSON files in src/functionalTest/resources/ directory
           - Use subdirectories: json/, requestJson/, responseJson/, text/ for appropriate content
           - Use JsonFile utility class to load and parse test data
           - Follow template pattern for test data files
        
        2. External Service Mocking:
           - Use WireMock for HTTP service dependencies (SetWiremockStub class)
           - Use gRPC stubs for gRPC services (SetGrpcStub class)
           - Configure mock responses programmatically based on test requirements
           - Use helper methods to build consistent mock responses
        
        3. Assertions and Validations:
           - Use dedicated assertor classes for standardized validation
           - Follow hierarchical validation approach (status, entity, field)
           - Use descriptive assertion messages for clear failure reporting
           - Validate cross-service data consistency
        
        4. Context Setup:
           - Use @BeforeClass for test environment initialization
           - Create test profiles with TestProfileFactory
           - Manage test users with UserFactory
           - Set up HTTP request headers and authentication
           - Clean up resources with @AfterClass methods
        
        5. Test Configuration:
           - Use system properties and environment variables
           - Configure test parameters with helper methods
           - Organize tests by feature in appropriate directories
           - Follow JUnit 4 conventions for test structure
        """

    role_specific = {
        AgentRole.PLANNER: """You are a strategic planner. Break down complex tasks into actionable steps. Also identify key requirements that will be verified separately.

When planning for functional test creation or updates:
1. Identify all services and components involved in the test flows
2. Determine what test data needs to be created or modified
3. Plan for test profile creation and management
4. Include mocking of external service dependencies
5. Ensure test assertions validate end-to-end functionality
6. Include test cleanup and resource management
7. Consider error scenarios and edge cases
8. Plan for cross-service validation
""",
        AgentRole.EXECUTOR: """You are a methodical executor. Follow plans precisely and show your work for the specific step you're assigned.

When implementing functional tests:
1. Use the correct directory structure (src/functionalTest/)
2. Follow established patterns for test data management:
   - Load JSON from src/functionalTest/resources
   - Use JsonFile utility for parsing
   - Manage test data file organization
3. Implement proper service mocking:
   - Use SetWiremockStub for HTTP services
   - Use SetGrpcStub for gRPC services
   - Configure appropriate mock responses
4. Structure test assertions correctly:
   - Use dedicated assertor classes
   - Follow hierarchical assertion approach
   - Include descriptive assertion messages
5. Set up proper test context:
   - Initialize with @BeforeClass
   - Create test profiles with TestProfileFactory
   - Manage test users with UserFactory
   - Clean up with @AfterClass
6. Follow test naming and organization conventions
7. Include documentation for test scenarios and data
8. Create comprehensive tests covering positive cases, edge cases, and error scenarios
9. Validate end-to-end functionality across service boundaries
""",
        AgentRole.FIXER: """You are a specialized fixer. Focus specifically on fixing one requirement issue at a time based on verification feedback. Don't try to solve everything, just fix what's needed.

When fixing functional test issues:
1. Understand the specific validation failure
2. Check test data configuration
3. Verify mock service setup
4. Review assertion structure and expectations
5. Ensure test context is properly initialized
6. Fix only the specific issue identified
7. Maintain the existing test structure and patterns
8. Follow project-specific test conventions
""",
        AgentRole.VERIFIER: """You are a critical verifier. Focus on verifying one requirement at a time thoroughly against execution or fixer results.

When verifying functional tests:
1. Check for proper directory structure (src/functionalTest/)
2. Verify test data management follows patterns:
   - JSON files in resources directory
   - Proper data loading mechanisms
   - Appropriate test data organization
3. Confirm service mocking is implemented correctly:
   - WireMock for HTTP services
   - gRPC stubs for gRPC services
   - Appropriate mock responses
4. Validate assertion structure:
   - Dedicated assertor classes
   - Hierarchical assertion approach
   - Descriptive assertion messages
5. Ensure proper test context:
   - @BeforeClass initialization
   - Test profile creation
   - Resource cleanup with @AfterClass
6. Verify end-to-end flow validation
7. Check for error scenario and edge case coverage
8. Confirm test documentation completeness
""",
        AgentRole.FINALIZER: """You are a skilled synthesizer. Create polished final outputs from execution results.

When finalizing functional test implementations:
1. Ensure all tests are properly organized in src/functionalTest/
2. Confirm test data files are correctly placed in resources directories
3. Verify mock service configurations are complete
4. Check that assertions comprehensively validate functionality
5. Ensure proper test context setup and cleanup
6. Confirm error scenarios and edge cases are covered
7. Verify documentation completeness
8. Ensure tests follow project-specific conventions
9. Compile a summary of test coverage and functionality
""",
    }

    return f"{base_prompt}\n\n{role_specific[role]}"


def create_prompt_for_role(role: AgentRole, state: MultiAgentState, query: str) -> str:
    """Create a role-specific prompt based on the current state"""

    base_prompt = f"""
    
    Full Problem statement:
    {query}
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
            Plan shouldn't be very strict, this is a overview multi step plan so that we can implement the solution in stages
            Keep room for exploration of codebase, understanding the context better, searching alternatives and verifying and comparing solutions
            
            FUNCTIONAL TEST PLANNING GUIDELINES:
            1. Identify which service functionalities need to be tested
            2. Determine what test data needs to be created or updated
            3. Plan for test profile creation and management
            4. Include steps for mocking external service dependencies
            5. Ensure test assertions will validate end-to-end functionality
            6. Include test cleanup and resource management
            7. Consider error scenarios and edge cases
            8. Plan for cross-service validation

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
            
            FUNCTIONAL TEST KEY REQUIREMENTS GUIDANCE:
            1. Tests must be located in src/functionalTest/ directory
            2. Tests must follow project patterns for data management, mocking, assertions, setup/teardown
            3. Tests must validate end-to-end functionality across service boundaries
            4. Tests must include error scenarios and edge cases
            
            Example output:
--------------
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

1. Fixes the string, make sure the class implements toString(), print Methods, 

2. The API must validate user input including: email in proper format (e.g., user@domain.com), password minimum 8 characters with at least one uppercase letter, one lowercase letter, one number, and one special character.

3. Make sure the output requirement is met, it should use the exact format as Result:'''{...}''' and only have utf-8 encoded characters

... so on

----------------------
    Note above plan is example for the format - Content of each step and key requirement must follow guidelines mentioned above the example
    Output a comprehensive plan with numbered steps followed by Key Requirements section with numbered steps. Make sure to include all the task requirements in the Key Requirements section.
    Especially focus on the requirements that are critical for the task and need to be verified later like output formatting, do's and don'ts etc.
    IMPORTANT: ONLY OUTPUT THE PLAN AND KEY REQUIREMENTS. DO NOT SOLVE THE PROBLEM. CLUB SIMILAR REQUIREMENTS TOGETHER, DON"T CREATE TOO MANY UNNECCESSARY REQUIREMENTS. DO NOT HAVE SUB-POINTS IN THE REQUIREMENTS (IT WON"T BE PARSED PROPERLY).
    IMPORTANT: DO NOT FIX THE ISSUE OR TRY TO SOLVE THE PROBLEM. YOUR ROLE IS TO CREATE A PLAN AND LIST OF KEY REQUIREMENTS
    Your core task is to udpate the functional tests in src/functionalTest based on the PR change analysis, i.e. add new tests or update existing tests based on the PR changes.
            """
        )

    elif role == AgentRole.EXECUTOR:
        # Modified executor prompt to focus on the current step only
        steps_status = _format_steps_status(state)
        current_step = (
            state.plan_steps[state.current_step_index]
            if state.plan_steps and state.current_step_index < len(state.plan_steps)
            else None
        )

        return (
            base_prompt
            + f"""
        The overall plan to execute is:
        {state.plan}
        
        Steps Status:
        {steps_status}
        
        Your role is to execute ONE SPECIFIC STEP of the plan methodically.
        
        CURRENT STEP TO EXECUTE: {current_step.description if current_step else "No step defined yet"}
        
        Execute ONLY this specific step. Show your detailed work for this step only.
        If you encounter issues with this step, try to overcome them and note what adjustments were needed.
        
        FUNCTIONAL TEST IMPLEMENTATION GUIDELINES:
        1. Directory Structure:
           - All tests must be in src/functionalTest/
           - Test data must be in src/functionalTest/resources/ in appropriate subdirectories
        
        2. Test Data Management:
           - Use JsonFile utility to load and parse JSON test data
           - Follow the template pattern for test data files
           - Use appropriate subdirectories for different types of test data
        
        3. External Service Mocking:
           - Use SetWiremockStub for HTTP service dependencies
           - Use SetGrpcStub for gRPC service dependencies
           - Configure mock responses based on test requirements
        
        4. Assertions and Validations:
           - Use dedicated assertor classes for validation
           - Follow the hierarchical validation approach
           - Include descriptive assertion messages
           - Validate cross-service data consistency
        
        5. Test Context Management:
           - Use @BeforeClass for initialization
           - Create test profiles with TestProfileFactory
           - Manage test users with UserFactory
           - Clean up resources with @AfterClass
        
        6. Error and Edge Cases:
           - Include tests for service failures
           - Test timeout scenarios
           - Validate error propagation
           - Cover edge cases

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
        
        
        IMPORTANT: Include CURRENT CONTEXT DATA in your response. CURRENT CONTEXT DATA should also carry all the relevant information from previous execution, Include the execution result also at the end of the current context data (basically attach the current progress of the update)
        """
        )

    elif role == AgentRole.FIXER:
        # Special prompt for FIXER role
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
        
        FUNCTIONAL TEST FIXING GUIDELINES:
        1. Test Structure Issues:
           - Check directory structure (should be in src/functionalTest/)
           - Verify test data location (should be in resources/ subdirectories)
           - Ensure proper test class organization
        
        2. Test Data Management Issues:
           - Check JsonFile utility usage
           - Verify template pattern for test data
           - Ensure proper deserialization into POJOs
        
        3. External Service Mocking Issues:
           - Verify WireMock and gRPC stub configuration
           - Check mock response construction
           - Ensure proper mock service behavior
        
        4. Assertion and Validation Issues:
           - Check assertor class usage
           - Verify hierarchical validation approach
           - Ensure proper validation of cross-service data
        
        5. Test Context Issues:
           - Verify @BeforeClass initialization
           - Check test profile and user creation
           - Ensure proper @AfterClass cleanup
        
        6. Edge and Error Case Issues:
           - Check for service failure handling
           - Verify timeout scenario coverage
           - Ensure error propagation validation

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
        
        
        IMPORTANT: Make sure the fix doesn't violate one of the previous requirements
        IMPORTANT: Include CURRENT CONTEXT DATA in your response. CURRENT CONTEXT DATA should also carry all the relevant information from previous execution, Include the execution result also at the end of the current context data (basically attach the current progress of the update)
        """
        )

    elif role == AgentRole.VERIFIER:

        current_req = state.requirements[state.current_requirement_index]
        return (
            base_prompt
            + f"""    
            
            {"The fixer results for the current requirement are:" if state.fixer_results else ""}
            {state.fixer_results if state.fixer_results else ""}
            
            Requirements Status:
            {_format_requirements_status(state)}
            
            The last few execution results are (older to latest):
            {state.execution_results[-2:] if len(state.execution_results) > 1 else state.execution_results}
            This above result is the current state of the code and the output of the last execution
            
            Your role is to verify if the {"execution" if not state.fixer_results else "fixer"} successfully completed the CURRENT requirement:
            
            CURRENT REQUIREMENT: {current_req.description}
            
            FUNCTIONAL TEST VERIFICATION CRITERIA:
            1. Directory Structure:
               - Tests must be in src/functionalTest/
               - Test data must be in resources/ subdirectories
            
            2. Test Data Management:
               - Must use JsonFile utility for loading data
               - Must follow template pattern for test data
               - Must properly deserialize into POJOs
            
            3. External Service Mocking:
               - Must use WireMock and gRPC stubs
               - Must configure proper mock responses
               - Must handle service behavior correctly
            
            4. Assertions and Validations:
               - Must use dedicated assertor classes
               - Must follow hierarchical validation approach
               - Must validate cross-service data consistency
            
            5. Test Context Management:
               - Must use @BeforeClass for initialization
               - Must create proper test profiles and users
               - Must clean up resources with @AfterClass
            
            6. Error and Edge Cases:
               - Must handle service failures
               - Must test timeout scenarios
               - Must validate error propagation
               - Must cover edge cases
            
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
            """
        )

    elif role == AgentRole.FINALIZER:
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
        
        FUNCTIONAL TEST FINALIZATION CHECKLIST:
        1. Directory Structure:
           - All tests are in src/functionalTest/
           - Test data is in resources/ subdirectories
        
        2. Test Data Management:
           - JsonFile utility is used correctly
           - Template pattern is followed
           - Data is properly organized and managed
        
        3. External Service Mocking:
           - WireMock and gRPC stubs are configured properly
           - Mock responses are appropriate for test scenarios
           - Service behavior is correctly simulated
        
        4. Assertions and Validations:
           - Dedicated assertor classes are used
           - Hierarchical validation approach is followed
           - Cross-service data is properly validated
        
        5. Test Context Management:
           - @BeforeClass initializes properly
           - Test profiles and users are created correctly
           - Resources are cleaned up with @AfterClass
        
        6. Error and Edge Cases:
           - Service failures are handled
           - Timeout scenarios are tested
           - Error propagation is validated
           - Edge cases are covered
        
        Your role is to produce the final output based on the execution results.
        Synthesize the information into a coherent, well-structured response.
        Format your response appropriately (markdown for text, proper code blocks for code, etc.)
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
    """Format the steps status for display in prompts"""
    if not state.plan_steps:
        return "No steps have been identified yet."

    status_lines = []
    for i, step in enumerate(state.plan_steps):
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

    return clean_reqs


def extract_plan_steps(plan_text: str) -> List[PlanStep]:
    """
    Extract individual steps from the planner's output.
    - Look for the plan section by finding title or numbered list
    - Extract numbered steps from the plan
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
        m2 = re.search(r"(?:^|\n)((?:[ \t]*\d+\.[ \t]+.+\n*)+)", plan_text)
        if m2:
            section_text = m2.group(1)
        else:
            # Last resort: use the whole text
            section_text = plan_text

    # Extract numbered steps
    steps = []
    step_matches = re.findall(r"^[ \t]*(\d+\.?)[ \t]+(.+)$", section_text, re.MULTILINE)

    if step_matches:
        for _, step_desc in step_matches:
            # Clean up the step description
            step_desc = step_desc.strip()
            if step_desc:
                steps.append(PlanStep(description=step_desc))

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
        tool_service: ToolService,
    ):
        """Initialize the multi-agent system with configuration and tools"""
        self.tasks = config.tasks
        self.max_iter = config.max_iter
        self.config = config
        self.tool_service = tool_service

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
            model_settings={"parallel_tool_calls": True, "max_tokens": 14000, "temperature": 0.4, "extra_body": {"max_tokens": 14000, "temperature": 0.3}},
        )

    def _get_next_role(self, state: MultiAgentState) -> AgentRole:
        """Determine the next role based on the current state"""
        if state.current_role == AgentRole.PLANNER:
            return AgentRole.EXECUTOR
        elif state.current_role == AgentRole.EXECUTOR:
            # Check if all steps are executed
            if (
                state.plan_steps
                and state.current_step_index < len(state.plan_steps) - 1
            ):
                # More steps to execute - stay in executor role but move to next step
                state.current_step_index += 1
                return AgentRole.EXECUTOR
            else:
                # All steps executed, move to verifier
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
                    # Reset fixer results when moving to a new requirement
                    state.fixer_results = ""
                    return AgentRole.VERIFIER
                else:
                    # All requirements verified, move to finalizer
                    return AgentRole.FINALIZER
            elif state.iterations >= self.max_iter:
                # Max iterations reached, go to finalizer
                return AgentRole.FINALIZER
            else:
                # Current requirement not verified, route to FIXER instead of EXECUTOR
                return AgentRole.FIXER
        elif state.current_role == AgentRole.FIXER:
            # After fixing, go back to verifier to check if the fix worked
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
        """Update the state based on the role and result"""
        if role == AgentRole.PLANNER:
            self.state.plan = result
            # Extract requirements from plan
            self.state.requirements = extract_requirements(result)
            # Extract plan steps
            self.state.plan_steps = extract_plan_steps(result)
            # Reset current requirement and step indices
            self.state.current_requirement_index = 0
            self.state.current_step_index = 0
            # Reset iteration counter
            self.state.iterations = 0
            # Reset fixer results
            self.state.fixer_results = ""
            # Reset execution results
            self.state.execution_results = []

        elif role == AgentRole.EXECUTOR:
            # Update the current step's execution status and result
            if self.state.plan_steps and self.state.current_step_index < len(
                self.state.plan_steps
            ):
                current_step = self.state.plan_steps[self.state.current_step_index]
                current_step.executed = True
                current_step.execution_result = result

                # Add the result to execution_results with a header showing which step it belongs to
                step_num = self.state.current_step_index + 1
                total_steps = len(self.state.plan_steps)
                step_result = f"\n\n### Step {step_num}/{total_steps}: {current_step.description}\n{result}"
                self.state.execution_results.append(step_result)

            # Reset fixer results when executing a new step
            self.state.fixer_results = ""

        elif role == AgentRole.FIXER:
            # Update fixer results
            self.state.fixer_results = result

            # Append fixer results to execution results to maintain context
            self.state.execution_results.append(
                f"\n\n### Fix for requirement {self.state.current_requirement_index + 1}:\n{result}"
            )

        elif role == AgentRole.VERIFIER:
            self.state.verification_results = result

            # Strict detection for "VERIFIED: YES" at the start of response
            match = re.search(r"VERIFIED:\s*YES", result, re.IGNORECASE)
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
                # Reset fixer results when verification succeeds
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
            
        logger.info("Running pydantic-ai multi-agent stream")
        context = await self.tool_service.process_large_pr_tool.arun({"project_id": ctx.project_id, "base_branch": "prev-2"})
        prompt = f"""    
        Based on a provided PR change summary, you will create or update functional tests that verify new features and changes work correctly across service boundaries.
        Input Instructions
        I will provide you with a detailed summary of changes made in a PR. Review this carefully to understand:

        New classes and data models added
        API changes and endpoint updates
        Business logic modifications
        Feature flags and configurations

        Output Instructions
        Generate ONLY the code for functional tests that need to be created or updated. Include:

        Complete test class files with proper imports, setup, and teardown methods
        Any necessary test data setup
        Required assertions
        Test data files (JSON) if needed
        Helper methods for test setup

        DO NOT include explanations, comments about what you're doing, or any text outside of the actual code files.
        Functional Test Structure Information
        Understand the following about the project's functional test structure:

        Test Data Setup:

        Test data is stored as JSON files in src/functionalTest/resources
        Use JsonFile utility to load and parse JSON files
        Test files follow template patterns (e.g., single_item_no_addon_no_variant_template.json)
        Data is deserialized into Java POJOs for manipulation


        Integration Testing Approach:

        Functional tests are essentially integration tests
        Tests should call across service boundaries when possible
        Only mock external dependencies that cannot be directly called
        Use SetWiremockStub for mocking necessary external services
        Create realistic mock responses that match production behavior


        Assertions and Validations:

        Use dedicated assertor classes for standardized assertions
        Follow hierarchical approach: status codes, headers, entity-level, field-by-field
        Use both standard JUnit methods and custom assertion methods
        Include descriptive assertion messages for clarity


        Context Setup:

        Each test class has @BeforeClass (init()) and @AfterClass (destroy()) methods
        Create test profiles using TestProfileFactory
        Set configuration options relevant to the feature being tested
        Establish default headers for API calls
        Each test file runs with a different user


        Test Configuration:

        Set test parameters using methods like setCheckoutOptions()
        Tests are organized by feature in subdirectories
        Follow JUnit 4 conventions with @Test annotations
        Configure dynamic mock responses based on test requirements



        Task Steps

        Analyze the provided PR changes to identify required test updates
        Identify new features, endpoints, fields, or behaviors that need testing
        Create test cases that verify each new feature works correctly
        Ensure tests validate both success paths and error scenarios
        Follow existing test patterns for setup, execution, and assertions

        Important: Create functional (integration) tests that verify features work across service boundaries. Do not create unit tests - these should be comprehensive end-to-end tests that validate complete flows.
        Based on the PR changes summary, generate ONLY the code for all required functional test updates. Include filenames at the top of each file.
        
        {context}
        """
        query = ctx.query + prompt
        ctx.query = query
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
                    usage_limits=UsageLimits(
                        response_tokens_limit=14000,
                    ),
                    # message_hist`ory=[
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
