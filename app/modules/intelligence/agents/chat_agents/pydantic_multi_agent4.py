import functools
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
        
        PR Analysis Summary:\n\n<change_analysis>\n1. Change: Addition of new code coverage exclusions for two new packages:\n   - **com/swiggy/api/services/cart/cross_sell/CrossSellInfo.java**\n   - **com/swiggy/api/services/cart/cross_sell/CrossSellItemTab.java**\n\n   This change appears twice in the pipeline stages under the "test" and "functional" code coverage exclusions sections.\n\n   Category: Other (Test configuration update)\n\n   Description:\n   - The new classes related to cross-sell functionality in the cart service are added to the list of exclusions for code coverage reports.\n   - This means these files will be excluded from coverage metrics during unit and functional test runs.\n\n   Implications for Testing:\n   - No direct business logic or functional change is introduced here.\n   - However, the addition indicates that new cross-sell related components (CrossSellInfo and CrossSellItemTab) exist and are currently excluded from coverage.\n   - This may imply these components are newly introduced or undergoing development and not yet fully covered by tests.\n   - Test plans should consider adding or updating tests for these cross-sell components in the future.\n   - Current test coverage reports will not reflect coverage for these files, so test completeness should be verified separately.\n\n   Downstream Dependencies:\n   - Not visible in this file; these are class names likely representing data models or UI components related to cross-sell in the cart.\n\n   Modified Inputs/Outputs:\n   - None in this file; only test exclusion configuration changed.\n\nSummary:\n- The only change is an update to test coverage exclusion lists to include new cross-sell related classes.\n\nNo other changes to business logic, API, data model, or infra are present in this diff.\n\n</change_analysis>\n\nFile: checkout-service.yaml  \nChange Type: Other (Test configuration update)  \nSummary: Added new cross-sell related classes to code coverage exclusion lists in unit and functional test pipelines.  \nEndpoint/Function: N/A (pipeline test configuration)  \nNew Behavior: The classes `CrossSellInfo` and `CrossSellItemTab` under the cart cross_sell package are excluded from code coverage reports during unit and functional testing.  \nTest Impact: Low (no functional change, but indicates new components requiring future test coverage)  \nDownstream Dependencies: Not applicable (configuration only)  \nModified Inputs: None  \nModified Outputs: None  \nContext Needed: Source code or test suites for `CrossSellInfo` and `CrossSellItemTab` classes to assess test coverage needs.  \nSuggested Test Scenarios:  \n- Verify that code coverage reports correctly exclude these new classes.  \n- Plan and design unit and integration tests for cross-sell functionality represented by these classes.  \n- Validate that excluding these classes from coverage does not mask missing tests for critical business logic.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Change: Addition of a new feature flag/config parameter "pilled_cross_sell_widget" to the existing comma-separated list in the environment variable XP_CONFIG_PARAMS.\n\n- Category: Business logic update (feature flag/configuration change affecting feature toggling)\n- Endpoint/Function/Component: XP_AB_SERVER_URL related feature toggling/configuration system; specifically impacts the checkout service\'s feature flag evaluation for cross-sell widgets.\n- New Behavior: The system now includes a new experimental or feature-flagged variant "pilled_cross_sell_widget" as part of the XP (experimentation/feature flag) configuration parameters. This likely enables or controls a new variant or UI/UX behavior related to cross-sell widgets in the checkout flow.\n- Test Implications:\n  - Tests need to verify that the new feature flag "pilled_cross_sell_widget" is recognized and correctly toggled by the system.\n  - Validation that enabling/disabling this flag affects the checkout behavior as expected.\n  - Regression tests to ensure no disruption to existing XP_CONFIG_PARAMS features.\n  - Potential testing of UI changes or business logic changes triggered by this flag.\n- Downstream Dependencies: Any service or component that reads or interprets XP_CONFIG_PARAMS for feature toggling, especially those related to cross-sell widgets in the checkout flow.\n- Modified Inputs: XP_CONFIG_PARAMS environment variable now includes an additional flag "pilled_cross_sell_widget".\n- Modified Outputs: None directly from this file; indirect effect on feature toggling behavior.\n- Context Needed: Code or configuration that reads and acts upon XP_CONFIG_PARAMS, especially feature flag evaluation logic for cross-sell widgets; checkout UI or backend components that consume this flag.\n\nSummary: The only change is the addition of a new feature flag "pilled_cross_sell_widget" to the XP_CONFIG_PARAMS environment variable, enabling a new variant or feature related to cross-sell widgets in the checkout service.\n\nNo other changes or modifications to endpoints, API parameters, or business logic are present in this diff.\n\n</change_analysis>\n\nFile: [environment configuration file]\nChange Type: Business logic update (feature flag/configuration change)\nSummary: Added a new feature flag "pilled_cross_sell_widget" to the XP_CONFIG_PARAMS environment variable to enable a new cross-sell widget variant in the checkout service.\nEndpoint/Function: Feature flag evaluation component consuming XP_CONFIG_PARAMS; checkout service cross-sell widget feature\nNew Behavior: The checkout service will now recognize and potentially activate the "pilled_cross_sell_widget" feature flag, enabling new cross-sell widget behavior or UI variant.\nTest Impact: Medium\nDownstream Dependencies: Feature flag evaluation logic, checkout cross-sell widget components (UI/backend)\nModified Inputs: XP_CONFIG_PARAMS environment variable (added "pilled_cross_sell_widget")\nModified Outputs: None directly; indirect changes in feature toggling behavior\nContext Needed: Feature flag evaluation code, checkout cross-sell widget implementation, XP server integration\nSuggested Test Scenarios:\n- Verify that the "pilled_cross_sell_widget" flag is correctly parsed and recognized by the feature flag system.\n- Test checkout flows with the flag enabled and disabled to confirm expected behavior changes.\n- Regression test existing XP_CONFIG_PARAMS flags to ensure no side effects.\n- Validate UI rendering and backend logic changes related to cross-sell widgets when the flag is active.\n- Edge case: Behavior when the flag is malformed or missing.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Change: Addition of a new feature flag/config parameter `pilled_cross_sell_widget` to the existing `XP_CONFIG_PARAMS` environment variable.\n\n- Category: Business logic update (feature flag/configuration change impacting feature toggling)\n- Component affected: XP_AB_SERVER_URL feature flag configuration, which likely controls feature toggles or A/B testing parameters related to checkout or cross-sell functionality.\n- New behavior: The system will now recognize and potentially enable or test a new feature or variant called `pilled_cross_sell_widget` as part of the XP (experience platform) configuration parameters. This could enable new UI elements or business logic related to cross-selling items in the checkout flow.\n- Modified inputs: The environment variable `XP_CONFIG_PARAMS` now includes an additional flag `pilled_cross_sell_widget`.\n- Modified outputs: No direct output change in this file, but downstream services or components that read `XP_CONFIG_PARAMS` will behave differently.\n- Downstream dependencies: Any service or component that reads `XP_CONFIG_PARAMS` to enable/disable features, especially those related to cross-sell widgets or checkout UI/logic.\n- Test implications: \n  - Tests need to verify that the new feature flag `pilled_cross_sell_widget` is correctly recognized and toggled.\n  - Functional tests should cover scenarios where this flag is enabled and disabled.\n  - Regression tests to ensure existing XP_CONFIG_PARAMS flags continue to work as expected.\n  - Integration tests to verify that cross-sell widget behavior changes according to this flag.\n  - Edge cases where the flag is missing, malformed, or combined with other flags.\n- No API, data model, or security changes are evident from this diff.\n- No direct UI code changes shown here, but the flag likely impacts UI behavior downstream.\n- No error handling or performance changes are indicated.\n\nSummary: This change introduces a new feature flag `pilled_cross_sell_widget` to the XP configuration parameters, potentially enabling a new cross-sell widget feature in the checkout experience.\n\n</change_analysis>\n\nFile: [environment configuration file]\nChange Type: Business logic update (feature flag addition)\nSummary: Added a new feature flag `pilled_cross_sell_widget` to the XP configuration parameters to enable or test a new cross-sell widget feature.\nEndpoint/Function: XP_AB_SERVER_URL / XP_CONFIG_PARAMS feature flag configuration (affects checkout cross-sell widget behavior)\nNew Behavior: The system will now consider the `pilled_cross_sell_widget` flag in its feature toggling logic, potentially enabling new cross-sell widget functionality in checkout flows.\nTest Impact: Medium\nDownstream Dependencies: Components/services that read and act on `XP_CONFIG_PARAMS` flags, especially checkout and cross-sell related modules.\nModified Inputs: XP_CONFIG_PARAMS environment variable (added `pilled_cross_sell_widget`)\nModified Outputs: None directly in this file; downstream behavior changes expected.\nContext Needed: Functions or modules that parse and use `XP_CONFIG_PARAMS` flags, especially those controlling cross-sell widgets or checkout UI features.\nSuggested Test Scenarios:\n- Verify that the system correctly detects and enables the `pilled_cross_sell_widget` flag.\n- Test checkout flows with the flag enabled and disabled to observe differences in cross-sell widget behavior.\n- Validate that other XP_CONFIG_PARAMS flags continue to function as expected.\n- Test system behavior when the flag is missing or malformed.\n- Regression tests for checkout and cross-sell features to ensure no unintended side effects.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of new feature flag constants related to "Pilled Cross Sell Widget":\n   - New constants added:\n     - PILLED_CROSS_SELL_WIDGET_SKIP_XP\n     - PILLED_CROSS_SELL_WIDGET_DEFAULT_VARIANT\n     - PILLED_CROSS_SELL_WIDGET_DEFAULT_VARIANT_DEFAULT_VALUE ("non_tabbed")\n     - PILLED_CROSS_SELL_WIDGET_TABBED_VARIANT ("tabbed")\n     - PILLED_CROSS_SELL_WIDGET_ENABLED_USER_IDS\n\n   Category: Business logic update (feature flag and variant handling)\n\n2. Addition of two new public methods related to the "Pilled Cross Sell Widget":\n   - isPilledCrossSellWidgetXpSkipped():\n     - Returns a boolean indicating if the XP call for the pilled cross sell widget should be skipped.\n     - Reads from the checkoutOptions feature flag PILLED_CROSS_SELL_WIDGET_SKIP_XP with default false.\n\n   - getPilledCrossSellWidgetDefaultVariant(String customerId):\n     - Returns a String representing the default variant for the pilled cross sell widget.\n     - Checks if the given customerId is in a whitelist (PILLED_CROSS_SELL_WIDGET_ENABLED_USER_IDS).\n     - If whitelisted, returns "tabbed" variant.\n     - Otherwise, returns the configured default variant or the default value "non_tabbed".\n\n   Category: Business logic update (feature flag evaluation and variant selection)\n\n3. Functional impact:\n   - These methods introduce a new feature toggle and variant selection logic for a "pilled cross sell widget" feature.\n   - The logic includes:\n     - Ability to skip XP evaluation for this widget.\n     - Variant selection based on user whitelist or default configuration.\n   - This affects how the UI or business logic might render or behave for cross-sell widgets in the cart or checkout flow.\n\n4. Test implications:\n   - New feature flag behavior needs to be tested:\n     - When XP call is skipped (true) vs not skipped (false).\n   - Variant selection logic:\n     - For whitelisted user IDs, the variant should be "tabbed".\n     - For non-whitelisted users, the variant should be the configured default or fallback to "non_tabbed".\n   - Edge cases:\n     - Null or empty customerId.\n     - Empty or malformed whitelist.\n     - Missing or empty default variant config.\n   - Integration with XP platform calls if applicable (though skipping XP is a boolean flag).\n   - Impact on UI rendering or business logic that consumes these methods.\n\n5. Downstream dependencies:\n   - These new methods do not call other functions within this file.\n   - They rely on checkoutOptions to fetch configuration values.\n   - Utility methods used: Utility.isNotEmpty (for customerId check).\n   - No changes to parameters or return types of existing methods.\n\n6. Modified inputs/outputs:\n   - New method getPilledCrossSellWidgetDefaultVariant takes String customerId as input.\n   - Returns String variant name.\n   - isPilledCrossSellWidgetXpSkipped returns boolean, no inputs.\n\n7. No API signature changes or data model changes.\n8. No error handling or performance changes.\n9. No direct UI code changes visible here, but this likely affects UI rendering downstream.\n\n10. Context needed for further testing:\n    - Where and how these methods are called in the application (e.g., UI components, cart rendering).\n    - The XP platform integration for cross sell widgets.\n    - The configuration management for checkoutOptions (how flags are set).\n    - Any UI or service code that consumes the variant returned by getPilledCrossSellWidgetDefaultVariant.\n\n</change_analysis>\n\nFile: FeatureFlagService.java  \nChange Type: Business logic update  \nSummary: Added feature flags and methods to control skipping XP calls and variant selection for a new "pilled cross sell widget" feature.  \nEndpoint/Function:  \n- isPilledCrossSellWidgetXpSkipped()  \n- getPilledCrossSellWidgetDefaultVariant(String customerId)  \n\nNew Behavior:  \n- Ability to skip XP evaluation for the pilled cross sell widget via a feature flag.  \n- Variant selection for the widget based on whether the user is whitelisted (returns "tabbed") or else returns a configured default variant or fallback default ("non_tabbed").  \n\nTest Impact: Medium  \n\nDownstream Dependencies:  \n- checkoutOptions.getOptionAsBoolean  \n- checkoutOptions.getOption  \n- Utility.isNotEmpty  \n\nModified Inputs:  \n- getPilledCrossSellWidgetDefaultVariant(String customerId)  \n\nModified Outputs:  \n- isPilledCrossSellWidgetXpSkipped() returns boolean  \n- getPilledCrossSellWidgetDefaultVariant() returns String variant  \n\nContext Needed:  \n- Usage locations of these methods in the application (UI or service layers)  \n- Configuration of checkoutOptions for these new flags  \n- XP platform integration related to cross sell widgets  \n- UI rendering logic for cross sell widgets  \n\nSuggested Test Scenarios:  \n- Verify isPilledCrossSellWidgetXpSkipped returns correct boolean based on feature flag.  \n- Verify getPilledCrossSellWidgetDefaultVariant returns "tabbed" for whitelisted user IDs.  \n- Verify getPilledCrossSellWidgetDefaultVariant returns configured default variant when user is not whitelisted.  \n- Verify fallback to "non_tabbed" default variant when no config is set.  \n- Test behavior with null, empty, or invalid customerId.  \n- Test behavior with empty or missing whitelist.  \n- Test integration with UI rendering or business logic that consumes these methods.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Change: Addition of a new import for `CrossSellTab` and modification of the fallback and exception handling return statements in `cartSuggestions` and `cartSuggestionFallback` methods to include an additional parameter - an empty list of `CrossSellTab`.\n   - Category: API change, Data model change, Error handling change\n   - Endpoint/Function affected: `cartSuggestions` method and `cartSuggestionFallback` method in `CartSuggestionController`\n   - New Behavior: Previously, when an exception occurred or the Hystrix fallback was triggered, the response returned by `generateSuggestionsResponse` included only a boolean flag and an empty list of `CrossSellSuggestionV2`. Now, the response also includes an empty list of `CrossSellTab`. This implies that the API response structure has been extended to include an additional data model element (`CrossSellTab`), even in failure/fallback scenarios.\n   - Test Implications:\n     - Tests must verify that the API response now consistently includes the new `CrossSellTab` list in all cases, including success, failure, and fallback.\n     - Validation of the response schema must be updated to include this new field.\n     - Tests for fallback and error scenarios must confirm that the new field is present and correctly initialized (empty list).\n     - Potential backward compatibility tests to ensure clients handle the extended response properly.\n   - Downstream Dependencies: \n     - `generateSuggestionsResponse` method (imported statically from `CrossSellSuggestionHelper`) is modified to accept and handle the additional `CrossSellTab` parameter.\n     - `crossSellSuggestionService.getCrossSellSuggestion` remains unchanged in this file but is the main service call.\n   - Modified Inputs/Parameters: None in the controller method signature or request input.\n   - Modified Outputs/Return Values: The `GetCrossSellSuggestionResponse` now includes an additional list of `CrossSellTab` in the response payload.\n   - Additional Notes: The change is limited to the controller layer\'s error/fallback response construction; no changes to the main successful service call or request handling are visible here.\n\nSummary:\n- The main functional change is the extension of the API response structure to include an additional data model element (`CrossSellTab`), particularly in fallback and error scenarios.\n- This affects the contract of the `/api/v1/cart/suggestions` POST endpoint.\n- Testing must be updated to validate this extended response structure and ensure consistent behavior across success, failure, and fallback paths.\n</change_analysis>\n\nFile: CartSuggestionController.java  \nChange Type: API change, Data model change, Error handling change  \nSummary: The API response for cart suggestions now includes an additional `CrossSellTab` list in fallback and error responses.  \nEndpoint/Function: `/api/v1/cart/suggestions` POST endpoint, specifically `cartSuggestions` and `cartSuggestionFallback` methods.  \nNew Behavior: On failure or fallback, the response includes an empty list of `CrossSellTab` alongside the existing empty list of `CrossSellSuggestionV2`, extending the response schema.  \nTest Impact: Medium  \nDownstream Dependencies: `generateSuggestionsResponse` (from `CrossSellSuggestionHelper`), `crossSellSuggestionService.getCrossSellSuggestion`  \nModified Inputs: None  \nModified Outputs: `GetCrossSellSuggestionResponse` now contains an additional list of `CrossSellTab` in error and fallback responses.  \nContext Needed:  \n- Implementation of `generateSuggestionsResponse` in `CrossSellSuggestionHelper` to understand how the new parameter is handled.  \n- Definition and usage of `CrossSellTab` data model.  \n- `GetCrossSellSuggestionResponse` class definition to confirm schema changes.  \nSuggested Test Scenarios:  \n- Verify successful response still returns expected data (no change visible here, but good to confirm).  \n- Verify error scenario response includes empty `CrossSellSuggestionV2` list and empty `CrossSellTab` list.  \n- Verify fallback scenario response includes empty `CrossSellSuggestionV2` list and empty `CrossSellTab` list.  \n- Validate response schema includes `CrossSellTab` field and is correctly serialized.  \n- Confirm that metrics increment on failure as before.  \n- Test client handling of the extended response schema for backward compatibility.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new field `crossSellWidgetData` of type `CrossSellWidgetData` to the `Cart` class.\n\n- Category: Data model change (schema)\n- Component affected: `Cart` data model class used in the cart API responses and internal representations.\n- New behavior: The `Cart` object now includes an additional property `cross_sell_widget_data` which holds data related to cross-selling widgets. This likely means the cart response or internal cart representation can carry cross-sell related UI or business data.\n- Test implications:\n  - Serialization/deserialization tests need to verify the new field is correctly handled in JSON input/output.\n  - Backward compatibility tests to ensure older clients or services not expecting this field do not break.\n  - Functional tests to verify that when `crossSellWidgetData` is populated, the cart behaves as expected (e.g., cross-sell UI widgets appear or cross-sell logic triggers).\n  - Validation tests to ensure that invalid or missing `crossSellWidgetData` does not cause failures.\n  - Integration tests to ensure that the upstream services populating this field and downstream consumers using it work correctly.\n- Downstream dependencies: No functions are directly modified or called in this diff; this is purely a data model addition.\n- Modified inputs/parameters: None explicitly modified; however, the serialized JSON representation of `Cart` now includes a new optional field.\n- Modified outputs/return values: `Cart` JSON responses now may include `cross_sell_widget_data`.\n- Context needed: The definition and usage of `CrossSellWidgetData` class, any services or endpoints that populate or consume this field, and any UI components rendering cross-sell widgets.\n\nSummary: The only change is the addition of a new data model field `crossSellWidgetData` to the `Cart` class to support cross-selling widget data.\n\nNo other changes or business logic updates are present in the diff.\n\n</change_analysis>\n\nFile: com/swiggy/api/data/model/cart/presenter/Cart.java  \nChange Type: Data model change (schema)  \nSummary: Added a new field `crossSellWidgetData` to the `Cart` class to include cross-selling widget data in the cart representation.  \nEndpoint/Function: Cart data model class used in cart-related APIs and services  \nNew Behavior: The `Cart` object now optionally contains cross-sell widget data, enabling the cart to carry information needed for cross-selling UI or business logic.  \nTest Impact: Medium (due to serialization, backward compatibility, and integration considerations)  \nDownstream Dependencies: None visible in this file (no functions called or modified)  \nModified Inputs: None explicitly; JSON input may now include `cross_sell_widget_data`  \nModified Outputs: JSON output of `Cart` may now include `cross_sell_widget_data`  \nContext Needed:  \n- `CrossSellWidgetData` class definition and usage  \n- Services or endpoints that populate or consume `crossSellWidgetData`  \n- UI components or business logic that use cross-sell widget data  \nSuggested Test Scenarios:  \n- Serialization/deserialization of `Cart` with and without `crossSellWidgetData`  \n- Backward compatibility tests with clients not expecting this field  \n- Functional tests verifying cart behavior when `crossSellWidgetData` is present  \n- Validation tests for malformed or missing `crossSellWidgetData`  \n- Integration tests covering upstream and downstream usage of this new field\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of new imports related to cross-selling data models and utilities:\n   - Imports for classes like CatalogItem, CrossSellSuggestionV2, CrossSellTab, PricingInfo, VegClassifier, GetCrossSellSuggestionResponse, SuggestedItemCategory, CrossSellWidgetData, CrossSellItemTab, and MoneyUtils.\n   - Category: Data model change (new data structures introduced), Business logic update (support for cross-selling data).\n   - Implication: Introduction of new data structures implies new data handling and presentation logic that needs testing.\n\n2. Addition of a new private method `populateCrossSellWidgetData(BaseCart baseCart, Cart cart)`:\n   - This method extracts cross-sell suggestion data from `baseCart.getGetCrossSellSuggestionResponse()`.\n   - It builds a `CrossSellWidgetData` object, populates it with tabs and items converted from cross-sell suggestions.\n   - It handles two cases:\n     a) When `crossSellSuggestions` list is non-empty, it creates a single tab with popular items.\n     b) When `crossSellTabs` list is non-empty, it iterates over tabs and converts their suggestions.\n   - It also increments custom metrics for success/failure.\n   - Category: Business logic update (new feature to populate cross-sell widget data in cart response), Error handling change (metrics on success/failure).\n   - Implication: This adds new business logic for cross-sell presentation, requiring tests for correct data mapping, handling empty/null cases, and error scenarios.\n\n3. Invocation of `populateCrossSellWidgetData(baseCart, cart)` inside `basicCartPresenter` method before returning the cart:\n   - Ensures that cross-sell widget data is included in the cart response if available.\n   - Category: Business logic update (integration of new cross-sell data into cart presentation).\n   - Implication: Tests should verify that cross-sell data is correctly attached to the cart response under various conditions.\n\n4. Addition of a public method `convertCrossSellItemToCartMenuItem(List<CrossSellSuggestionV2> crossSellSuggestions)`:\n   - Converts a list of `CrossSellSuggestionV2` objects into a list of `Item` objects used in the cart.\n   - Maps fields like id, name, description, image, veg classifier, base price, final price, and subtotal.\n   - Uses `MoneyUtils.moneyToFloat` for price conversions.\n   - Contains a TODO comment indicating a hack for cross-selling widget enhancement.\n   - Category: Business logic update (data transformation logic for cross-sell items).\n   - Implication: Tests should cover correct conversion of cross-sell suggestions to cart items, including price and veg classification.\n\n5. Addition of error handling and logging in `populateCrossSellWidgetData` method:\n   - Logs errors and increments failure metrics if exceptions occur.\n   - Category: Error handling change.\n   - Implication: Tests should cover failure scenarios and verify metrics/logging behavior.\n\nSummary of impact:\n- New cross-sell widget data integration into cart presentation.\n- New data models and transformation logic for cross-sell suggestions.\n- Metrics tracking for cross-sell data population success/failure.\n- Changes primarily affect cart presentation logic and response structure.\n\nDownstream dependencies (functions called by changed code):\n- `convertCrossSellItemToCartMenuItem`\n- `CheckoutCustomMetrics.CROSS_SELL_SUGGESTIONS_METRICS.labels().inc()`\n- `cart.setCrossSellWidgetData()`\n\nModified inputs/parameters:\n- `BaseCart` now expected to potentially contain `GetCrossSellSuggestionResponse` with cross-sell data.\n\nModified outputs/return values:\n- `Cart` object now includes a new field `crossSellWidgetData` populated with cross-sell tabs and items.\n\nPotential test implications:\n- Verify that when `GetCrossSellSuggestionResponse` is present with cross-sell suggestions, the cart response includes correctly populated `crossSellWidgetData`.\n- Verify behavior when cross-sell suggestions are empty or null.\n- Verify correct mapping of cross-sell suggestion fields to cart item fields.\n- Verify that metrics increment correctly on success and failure.\n- Verify error handling and logging when exceptions occur during cross-sell data population.\n- Verify that existing cart presentation remains unaffected when no cross-sell data is present.\n\n</change_analysis>\n\nFile: com/swiggy/api/data/model/cart/presenter/CartPresenter.java  \nChange Type: Business logic update, Data model change, Error handling change  \nSummary: Added support for populating cross-sell widget data in the cart response by converting cross-sell suggestions into cart items and integrating them into the cart presentation, along with metrics tracking and error handling.  \nEndpoint/Function: `basicCartPresenter(CartOutput cartOutput, Headers headers)`, `populateCrossSellWidgetData(BaseCart baseCart, Cart cart)`, `convertCrossSellItemToCartMenuItem(List<CrossSellSuggestionV2> crossSellSuggestions)`  \nNew Behavior: When cross-sell suggestion data is available in the base cart, the cart response includes a `crossSellWidgetData` field containing cross-sell tabs and items converted to cart item format; metrics are tracked for success and failure of this population.  \nTest Impact: Medium  \nDownstream Dependencies: `convertCrossSellItemToCartMenuItem`, `CheckoutCustomMetrics.CROSS_SELL_SUGGESTIONS_METRICS.labels().inc()`, `cart.setCrossSellWidgetData()`  \nModified Inputs: `BaseCart.getGetCrossSellSuggestionResponse()` (new data source for cross-sell suggestions)  \nModified Outputs: `Cart.crossSellWidgetData` (new field added to cart response)  \nContext Needed: `GetCrossSellSuggestionResponse` class structure, `CrossSellSuggestionV2` and related cross-sell data models, `Cart` class changes for crossSellWidgetData field  \nSuggested Test Scenarios:  \n- Happy path: BaseCart contains cross-sell suggestions; verify cart response includes correctly populated crossSellWidgetData with tabs and items.  \n- Edge case: BaseCart contains empty cross-sell suggestions list; verify cart response has empty or null crossSellWidgetData.  \n- Edge case: BaseCart contains cross-sell tabs with multiple tabs and suggestions; verify all tabs and items are correctly converted and included.  \n- Error case: Simulate exception during cross-sell data processing; verify error is logged and failure metric is incremented.  \n- Verify that when no cross-sell data is present, cart response does not include crossSellWidgetData or it is empty.  \n- Verify price conversion correctness using MoneyUtils.  \n- Verify veg classification mapping from VegClassifier enum to integer flag.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new constant endpoint:\n   - Change: Added a new static final String constant named PILLED_CROSS_SELL_WIDGET_TABBED_ENDPOINT with the value "/v13".\n   - Category: API change (new endpoint addition)\n   - Component affected: DspServiceConfig class, specifically the configuration of API endpoints related to cross-sell widgets.\n   - New behavior: This change introduces a new API endpoint path presumably to be used for a "pilled cross-sell widget tabbed" feature or functionality. This suggests an expansion of the DSP service’s cross-sell capabilities with a new endpoint version.\n   - Test implications:\n     * Tests need to verify that this new endpoint is correctly referenced and used where applicable.\n     * Integration tests should confirm that calls to "/v13" behave as expected.\n     * Regression tests should ensure existing endpoints remain unaffected.\n     * If this endpoint corresponds to new or updated business logic, tests should cover the specific behavior behind this endpoint.\n   - Downstream dependencies: None visible in this file; this is a configuration constant only.\n   - Modified inputs: None (no parameters changed)\n   - Modified outputs: None (no return values changed)\n   - Context needed: Other classes or services that consume this endpoint constant, especially any service or controller classes that make HTTP calls or route requests to this endpoint.\n\nNo other changes are present in the diff.\n\nSummary: The change is a simple addition of a new API endpoint constant to the configuration class, enabling support for a new cross-sell widget tabbed feature.\n\n</change_analysis>\n\nFile: DspServiceConfig.java  \nChange Type: API change (new endpoint addition)  \nSummary: Added a new API endpoint constant for the pilled cross-sell widget tabbed feature.  \nEndpoint/Function: PILLED_CROSS_SELL_WIDGET_TABBED_ENDPOINT ("/v13") in DspServiceConfig  \nNew Behavior: Enables the DSP service to reference and use a new "/v13" endpoint for pilled cross-sell widget tabbed functionality.  \nTest Impact: Medium (new endpoint integration and usage need to be validated)  \nDownstream Dependencies: None visible in this file; likely other service or controller classes that use this constant.  \nModified Inputs: None  \nModified Outputs: None  \nContext Needed: Classes or services that utilize DspServiceConfig endpoints, particularly those handling cross-sell widget API calls.  \nSuggested Test Scenarios:  \n- Verify that the new endpoint constant is correctly loaded and accessible.  \n- Integration tests for API calls routed to "/v13" endpoint, verifying expected responses and behavior.  \n- Regression tests to ensure existing endpoints ("/v4", "/v9", "/v6") continue to function correctly.  \n- Edge cases around feature toggling or conditional usage of this new endpoint, if applicable.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new metric counter: CROSS_SELL_SUGGESTIONS_TAB_METRICS\n   - Category: Other (Metrics/Monitoring enhancement)\n   - Component affected: CheckoutCustomMetrics class (metrics collection for cross sell suggestions)\n   - Description: A new Prometheus Counter metric named "cross_sell_suggestions_tab_metrics" has been added. This metric tracks tab metrics for cross sell suggestions, specifically counting the number of items in each tab for both tabbed and fallback non-tabbed UI presentations. It includes two labels: "variant" and "tab".\n   - New Behavior: The system will now collect and expose metrics related to cross sell suggestion tabs, providing insights into item counts per tab variant.\n   - Test Implications:\n     * Verify that the new metric is correctly registered and increments as expected.\n     * Validate that the labels "variant" and "tab" are correctly applied and populated.\n     * Confirm that the metric increments in scenarios where cross sell suggestions tabs are rendered or processed.\n     * Since this is a metrics addition, no direct API or business logic changes are implied, but integration tests for metrics collection and monitoring dashboards might be needed.\n   - Downstream Dependencies: None visible in this file; this is a static metric definition.\n   - Modified Inputs: None (new metric added, no input parameters changed)\n   - Modified Outputs: None (no return values or API responses changed)\n   - Context Needed: The usage context of this metric in the codebase (e.g., where this counter is incremented) to understand triggering scenarios.\n   - Suggested Test Scenarios:\n     * Happy path: Confirm metric increments when cross sell suggestion tabs are displayed/processed.\n     * Edge cases: Validate behavior when variant or tab labels are empty or unexpected.\n     * Metrics registration and exposure correctness.\n</change_analysis>\n\nFile: CheckoutCustomMetrics.java  \nChange Type: Other (Metrics/Monitoring enhancement)  \nSummary: Added a new Prometheus counter metric to track item counts in cross sell suggestion tabs with variant and tab labels.  \nEndpoint/Function: CheckoutCustomMetrics class (new static Counter: CROSS_SELL_SUGGESTIONS_TAB_METRICS)  \nNew Behavior: The system will track and expose metrics for cross sell suggestion tabs, including item counts per tab and variant, aiding monitoring and analysis of cross sell UI components.  \nTest Impact: Low (metrics addition, no direct functional or API changes)  \nDownstream Dependencies: None visible in this file (metric definition only)  \nModified Inputs: None  \nModified Outputs: None  \nContext Needed: Code locations where CROSS_SELL_SUGGESTIONS_TAB_METRICS is incremented or used to understand operational impact.  \nSuggested Test Scenarios:  \n- Verify metric registration and availability in Prometheus metrics endpoint.  \n- Confirm metric increments correctly with appropriate labels during cross sell suggestion tab rendering or processing.  \n- Test label handling with various variant and tab values, including empty or unexpected inputs.\n--------------------------------------------------------------------------------\nError: Could not get content for file src/main/java/com/swiggy/api/pojo/CMS/crossselling/CrossSellTab.java\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new field `crossSellTabs` of type `List<CrossSellTab>` in the `GetCrossSellSuggestionResponse` data model.\n\n- Category: Data model change (schema, relationships)\n- Component affected: `GetCrossSellSuggestionResponse` class, which likely represents the response payload for an API endpoint returning cross-sell suggestions.\n- New behavior: The response now includes an additional list of cross-sell tabs, presumably to provide a categorized or tabbed view of cross-sell suggestions alongside the existing flat list of `crossSellSuggestions`.\n- Modified inputs/parameters: None (this is a response model change).\n- Modified outputs/return values: The response JSON now includes a new property `"cross_sell_tabs"` representing a list of `CrossSellTab` objects.\n- Downstream dependencies: The new field depends on the `CrossSellTab` class (imported from `com.swiggy.api.pojo.CMS.crossselling.CrossSellTab`). The actual usage or processing of this new field is not visible in this file, but downstream consumers of this response model will now receive this additional data.\n- Test implications:\n  - Serialization/deserialization tests to ensure the new field is correctly handled.\n  - API contract tests to verify that the response payload includes the new `cross_sell_tabs` field when applicable.\n  - Functional tests to validate the behavior of any UI or service components that consume this new field (e.g., rendering tabs in UI, logic that depends on tabs).\n  - Edge cases where `cross_sell_tabs` is empty, null, or contains multiple entries.\n  - Backward compatibility tests to ensure clients not expecting this field do not break.\n  - Validation of the contents and structure of `CrossSellTab` objects within the list.\n- Since this is a data model change, no direct business logic or algorithm changes are visible here, but the addition implies new functionality or UI behavior elsewhere.\n- No changes to error handling, performance, or security are evident from this diff.\n\nSummary: The addition of `crossSellTabs` extends the response model to include a new list of cross-sell tabs, likely enabling enhanced UI presentation or categorization of cross-sell suggestions.\n\n</change_analysis>\n\nFile: GetCrossSellSuggestionResponse.java  \nChange Type: Data model change (schema, relationships)  \nSummary: Added a new field `crossSellTabs` to the cross-sell suggestion response model to include a list of cross-sell tabs.  \nEndpoint/Function: `GetCrossSellSuggestionResponse` class (response model for cross-sell suggestions API)  \nNew Behavior: The API response now includes an additional property `cross_sell_tabs` containing a list of `CrossSellTab` objects, enabling clients to receive tabbed cross-sell data alongside the existing flat list of suggestions.  \nTest Impact: Medium  \nDownstream Dependencies: `CrossSellTab` class (used in the new field)  \nModified Inputs: None  \nModified Outputs: Added `cross_sell_tabs` field in the response JSON  \nContext Needed:  \n- `CrossSellTab` class definition and its fields  \n- The API endpoint/controller/service that returns `GetCrossSellSuggestionResponse`  \n- UI components or services that consume this response and render/use `crossSellTabs`  \nSuggested Test Scenarios:  \n- Verify that the response JSON includes the `cross_sell_tabs` field when tabs are present.  \n- Validate correct serialization/deserialization of `crossSellTabs` with various data sets (empty list, multiple tabs).  \n- Test backward compatibility with clients that do not expect `cross_sell_tabs`.  \n- Functional tests to ensure UI correctly renders cross-sell tabs based on this new data.  \n- Edge cases such as null or missing `cross_sell_tabs` field.  \n- Data integrity tests for the contents of each `CrossSellTab` object.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new field `category` of type `String` to the `SuggestedItem` data model class.\n   - Category: Data model change (schema)\n   - Component affected: `SuggestedItem` POJO class in the cross-selling API module.\n   - New behavior: The `SuggestedItem` class now includes a `category` attribute that is mapped to/from JSON via the `@JsonProperty("category")` annotation.\n   - Test implications:\n     - Serialization and deserialization tests need to verify that the `category` field is correctly handled in JSON payloads.\n     - Tests that create or consume `SuggestedItem` objects should validate the presence and correctness of the `category` field.\n     - If the `category` field is used downstream (e.g., filtering, display, scoring), those usages need to be tested for correct behavior with the new field.\n   - Downstream dependencies: None visible in this file; the class is a POJO without methods.\n   - Modified inputs/parameters: None explicitly, but the data model now expects an additional field in JSON input/output.\n   - Modified outputs/return values: The JSON representation of `SuggestedItem` now includes the `category` field.\n   - Context needed: Other classes or services that consume or produce `SuggestedItem` instances to understand how `category` is used.\n\nSummary:\n- The only change is the addition of a new data field `category` to the `SuggestedItem` class, impacting JSON serialization/deserialization and potentially downstream logic that uses this class.\n\nNo other changes were made to methods, logic, or API endpoints.\n\n</change_analysis>\n\nFile: SuggestedItem.java  \nChange Type: Data model change (schema)  \nSummary: Added a new `category` field to the `SuggestedItem` data model to capture item category information.  \nEndpoint/Function: `SuggestedItem` POJO class in the cross-selling API module  \nNew Behavior: The `SuggestedItem` class now includes a `category` attribute that will be serialized/deserialized as part of JSON payloads, enabling category information to be stored and transferred.  \nTest Impact: Medium  \nDownstream Dependencies: None visible in this file; likely used by services or controllers handling suggested items.  \nModified Inputs: JSON payloads representing `SuggestedItem` now include an optional `category` string field.  \nModified Outputs: JSON responses including `SuggestedItem` will now include the `category` field.  \nContext Needed: Classes or services that create, consume, or manipulate `SuggestedItem` instances (e.g., cross-selling recommendation services, API controllers, UI components).  \nSuggested Test Scenarios:  \n- Verify JSON serialization and deserialization correctly handle the `category` field (present, missing, null, empty string).  \n- Validate that `SuggestedItem` instances can be created with and without the `category` field without errors.  \n- Test downstream components that consume `SuggestedItem` for correct handling of the new `category` field.  \n- Edge cases: category with special characters, very long strings, or unexpected values.  \n- Backward compatibility: ensure older clients or services that do not send `category` still function correctly.\n--------------------------------------------------------------------------------\nError: Could not get content for file src/main/java/com/swiggy/api/pojo/crossselling/SuggestedItemCategory.java\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new field `suggestedItemsWithCategory` of type `List<SuggestedItem>` to the `Suggestions` class.\n   - Category: Data model change (schema, validation, relationships)\n   - Description: The data model for suggestions now includes a new list that holds suggested items along with their categories, presumably to provide richer information compared to the existing list of suggested item strings.\n   - Implications for testing:\n     - Tests need to verify correct serialization and deserialization of the new field.\n     - Validation tests should confirm that the new field can handle empty, null, and populated lists.\n     - Integration tests should verify that endpoints or services returning or consuming `Suggestions` handle the new field correctly.\n     - Backward compatibility tests to ensure existing clients that use only `suggestedItems` are not broken.\n     - Tests should verify that the `SuggestedItem` class is correctly defined and integrated.\n   - Modified Inputs: None explicitly in this file, but the data model now accepts an additional field.\n   - Modified Outputs: The `Suggestions` object now includes an additional property `suggestedItemsWithCategory`.\n   - Downstream Dependencies: The `SuggestedItem` class (not shown here) is a new dependency that needs to be validated.\n   - Functions called: None in this file; this is a POJO class.\n   - Context Needed: Definition and structure of the `SuggestedItem` class, any services or endpoints that produce or consume `Suggestions`.\n\nNo other changes are present in this diff.\n\nSummary: This change extends the `Suggestions` data model by adding a new field to hold suggested items along with their categories, enabling richer suggestion data to be passed through the API or internal services.\n\n</change_analysis>\n\nFile: Suggestions.java  \nChange Type: Data model change (schema, validation, relationships)  \nSummary: Added a new field `suggestedItemsWithCategory` to the `Suggestions` class to include suggested items with their categories.  \nEndpoint/Function: `Suggestions` POJO class (used in cross-selling API components)  \nNew Behavior: The `Suggestions` object now supports an additional list of `SuggestedItem` objects, allowing suggestions to carry category information alongside item identifiers.  \nTest Impact: Medium  \nDownstream Dependencies: `SuggestedItem` class (needs to be reviewed and tested)  \nModified Inputs: None directly in this class, but the data model now accepts `suggestedItemsWithCategory` as input.  \nModified Outputs: `Suggestions` instances now include `suggestedItemsWithCategory` in their serialized form.  \nContext Needed: Definition of `SuggestedItem` class, API endpoints or services using `Suggestions` class, serialization/deserialization logic for `Suggestions`.  \nSuggested Test Scenarios:  \n- Serialization and deserialization of `Suggestions` with and without `suggestedItemsWithCategory`.  \n- Validation of empty, null, and populated `suggestedItemsWithCategory` lists.  \n- Backward compatibility tests ensuring existing clients handle the new field gracefully.  \n- Integration tests verifying that services producing or consuming `Suggestions` correctly handle the new field.  \n- Tests verifying the correctness and completeness of the `SuggestedItem` data structure.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new private field in the BaseCart class:\n   - Field: private GetCrossSellSuggestionResponse getCrossSellSuggestionResponse;\n   - Category: Data model change (schema, validation, relationships)\n   - Description: The BaseCart class now holds an additional data member of type GetCrossSellSuggestionResponse, which appears related to cross-selling suggestions.\n   - Implications for testing:\n     - Tests involving serialization/deserialization of BaseCart objects need to verify the presence and correct handling of this new field.\n     - Any business logic or UI components that consume BaseCart objects may need to be tested for correct behavior when this field is set or unset.\n     - Validation and null handling tests for this new field should be considered.\n   - Downstream dependencies: None visible in this file; no methods using this field are shown.\n   - Modified inputs/outputs: No method signatures changed; this is an internal data model addition.\n\n2. Addition of import statement for GetCrossSellSuggestionResponse:\n   - Category: Other (supporting import for new data model field)\n   - No direct functional impact; supports the new field.\n\nNo changes were made to methods, parameters, return types, or business logic in this file. The static method fromCartInput remains unchanged.\n\nSummary: The only functional change is the addition of a new data member in the BaseCart class to hold cross-selling suggestion response data, likely for enhanced cross-selling features.\n\nTesting considerations focus on data model integrity, serialization, and any downstream usage of this new field in other parts of the system (not visible here).\n\n</change_analysis>\n\nFile: BaseCart.java  \nChange Type: Data model change  \nSummary: Added a new field getCrossSellSuggestionResponse to BaseCart for holding cross-selling suggestion response data.  \nEndpoint/Function: BaseCart class (no method signature changes)  \nNew Behavior: BaseCart instances can now store and provide access to GetCrossSellSuggestionResponse data, potentially enabling richer cross-selling features.  \nTest Impact: Low (primarily data model and serialization tests)  \nDownstream Dependencies: None visible in this file; likely used elsewhere in the codebase.  \nModified Inputs: None  \nModified Outputs: None  \nContext Needed:  \n- Usage of getCrossSellSuggestionResponse in other classes or services  \n- Serialization/deserialization logic for BaseCart  \n- Any UI or API layers consuming BaseCart objects  \nSuggested Test Scenarios:  \n- Verify BaseCart serialization and deserialization includes getCrossSellSuggestionResponse correctly.  \n- Validate behavior when getCrossSellSuggestionResponse is null vs populated.  \n- Test integration points where BaseCart is used to ensure no regressions due to the new field.  \n- Confirm that adding this field does not break backward compatibility or cause unexpected side effects.\n--------------------------------------------------------------------------------\nError: Could not get content for file src/main/java/com/swiggy/api/services/cart/cross_sell/CrossSellItemTab.java\n--------------------------------------------------------------------------------\n<change_analysis>\n1. **Addition of new imports and data models**  \n   - Categories: Data model change, Business logic update  \n   - Description: New imports related to `CrossSellTab`, `Item`, `User`, `AuthenticationContext`, `LocationContext`, `RequestContext`, `Timestamp`, `CartBlob`, and utility classes were added. This indicates new data structures and utilities are introduced for enhanced request building and response handling.  \n   - Test Implications: Tests need to cover the new data models and their integration in request/response flows.\n\n2. **Enhanced `convertDdbResponseToSuggestedItems` method**  \n   - Category: Business logic update  \n   - Function: `convertDdbResponseToSuggestedItems`  \n   - Change: Previously, only `suggestedItems` (a list of item IDs) were extracted from the DDB blob. Now, the method also extracts `suggestedItemsWithCategory` which includes item ID, score, and category. If `suggestedItemsWithCategory` is present and non-empty, it is returned instead of the older list.  \n   - Test Implications: Tests must verify correct parsing of both simple item lists and categorized items, and correct prioritization of returning categorized suggestions when available.\n\n3. **Enhanced `convertSuggestedItemsToDdbBlob` method**  \n   - Category: Business logic update  \n   - Function: `convertSuggestedItemsToDdbBlob`  \n   - Change: Along with storing a simple list of item IDs, the method now also serializes and stores `suggestedItemsWithCategory` (item ID, category, score) into the DDB blob.  \n   - Test Implications: Tests should verify that the DDB blob correctly serializes both simple and categorized suggested items, and that the TTL and keys are set correctly.\n\n4. **Modification of `generateSuggestionsResponse` method signature and behavior**  \n   - Category: API change, Business logic update  \n   - Function: `generateSuggestionsResponse`  \n   - Change: Added a new parameter `List<CrossSellTab> tabbedSuggestions` and set it in the response object. This adds a new field `crossSellTabs` to the response.  \n   - Test Implications: Tests must verify that the new field is correctly set and returned in the response, and that existing behavior remains unaffected.\n\n5. **Addition of a new overloaded method `getCrossSellSuggestionRequest`**  \n   - Category: Business logic update, API change  \n   - Function: `getCrossSellSuggestionRequest(Headers headers, CartBlob cartBlob)`  \n   - Change: This new method builds a `GetCrossSellSuggestionRequest` object from HTTP headers and a cart blob, extracting and mapping multiple contexts: device, authentication, items, restaurant ID, request context, analytics session context, and location context. It uses utility methods to extract values and sets defaults or empty values where necessary. It also handles exceptions gracefully by logging warnings.  \n   - Test Implications: This is a significant new code path for request creation, requiring tests for correct mapping from headers and cart blob, handling of null/empty values, and exception scenarios. Also, validation of all nested contexts and their fields is needed.\n\n6. **Minor change in `isValidSession` method**  \n   - Category: Business logic update, Security change  \n   - Function: `isValidSession`  \n   - Change: Added redundant parentheses around token equality check; no functional change but could be a code cleanup.  \n   - Test Implications: No new tests needed.\n\n7. **New static imports and utility usage**  \n   - Category: Business logic update  \n   - Change: Use of `Utility.getPlatformFromUserAgent` and other utility methods for extracting platform, device ID, version code, user agent, TID, session info, city ID, request channel, etc.  \n   - Test Implications: Tests should verify that utility methods are correctly integrated and that their outputs are correctly mapped into the request.\n\n8. **Additional null and empty checks with Optional usage**  \n   - Category: Business logic update, Error handling change  \n   - Change: The new request builder method uses Java Optional extensively to avoid null pointer exceptions and to conditionally set fields.  \n   - Test Implications: Tests should verify behavior with missing or partial data in headers and cart blob, ensuring no exceptions and correct defaults.\n\n**Downstream dependencies:**  \n- `Utility.getDeviceId()`, `Utility.getVersionCode()`, `Utility.getUserAgentAsSentFromClient()`, `Utility.getAgentType()`, `Utility.getTid()`, `Utility.getSessionInfo()`, `Utility.getCityIdFromRequest()`, `Utility.getRequestChannel()`  \n- `getPlatformFromUserAgent()`  \n- `convertDdbResponseToSuggestedItems()`  \n- `convertSuggestedItemsToDdbBlob()`  \n- `generateSuggestionsResponse()`  \n\n**Modified inputs/parameters:**  \n- `convertDdbResponseToSuggestedItems(CrossSellSuggestionsDdbBlob response)` now handles additional fields in `response`.  \n- `convertSuggestedItemsToDdbBlob(List<SuggestedItem> suggestedItems, String pk, String sk, int crossSellTtl)` now serializes additional fields.  \n- `generateSuggestionsResponse(boolean allSuggestionsAdded, List<CrossSellSuggestionV2> suggestions, List<CrossSellTab> tabbedSuggestions)` new parameter added.  \n- New method `getCrossSellSuggestionRequest(Headers headers, CartBlob cartBlob)` added.\n\n**Modified outputs/return values:**  \n- `convertDdbResponseToSuggestedItems` returns a list that may include categorized suggested items.  \n- `convertSuggestedItemsToDdbBlob` returns a blob including categorized suggested items.  \n- `generateSuggestionsResponse` returns response including cross-sell tabs.  \n- New request object constructed by `getCrossSellSuggestionRequest(Headers, CartBlob)`.\n\n**Summary of test implications:**  \n- Validate parsing and serialization of categorized suggested items in DDB blobs.  \n- Validate new response field `crossSellTabs` is correctly set and returned.  \n- Validate new request building from headers and cart blob, including all nested contexts and optional fields.  \n- Validate error handling and logging on malformed or incomplete inputs.  \n- Validate backward compatibility with old request/response formats.\n\n</change_analysis>\n\nFile: CrossSellSuggestionHelper.java  \nChange Type: Business logic update, API change, Data model change, Error handling change, Security change  \nSummary: Added support for categorized suggested items in DDB response and blob serialization, introduced cross-sell tabs in response, and implemented a new method to build cross-sell suggestion requests from HTTP headers and cart data with enhanced context mapping and validation.  \nEndpoint/Function:  \n- `convertDdbResponseToSuggestedItems`  \n- `convertSuggestedItemsToDdbBlob`  \n- `generateSuggestionsResponse`  \n- `getCrossSellSuggestionRequest(Headers headers, CartBlob cartBlob)` (new method)  \nNew Behavior:  \n- The system now supports suggested items with categories and scores in addition to simple item lists, storing and retrieving them from DDB blobs.  \n- The suggestion response includes a new field for cross-sell tabs to support tabbed UI or categorization.  \n- A new request builder method creates a fully populated `GetCrossSellSuggestionRequest` from HTTP headers and cart data, including device, authentication, location, analytics, and request context information, using utility methods and Optional to handle missing data gracefully.  \nTest Impact: High  \nDownstream Dependencies:  \n- `Utility.getDeviceId()`, `Utility.getVersionCode()`, `Utility.getUserAgentAsSentFromClient()`, `Utility.getAgentType()`, `Utility.getTid()`, `Utility.getSessionInfo()`, `Utility.getCityIdFromRequest()`, `Utility.getRequestChannel()`  \n- `getPlatformFromUserAgent()`  \n- `convertDdbResponseToSuggestedItems()`  \n- `convertSuggestedItemsToDdbBlob()`  \n- `generateSuggestionsResponse()`  \nModified Inputs:  \n- `CrossSellSuggestionsDdbBlob response` (added fields in JSON)  \n- `List<SuggestedItem> suggestedItems` (with category and score)  \n- New inputs: `Headers headers`, `CartBlob cartBlob` for request builder method  \nModified Outputs:  \n- List of `SuggestedItem` with category and score  \n- `CrossSellSuggestionsDdbBlob` with categorized suggested items serialized  \n- `GetCrossSellSuggestionResponse` with `crossSellTabs` field  \n- `GetCrossSellSuggestionRequest` built from headers and cart data  \nContext Needed:  \n- Definitions of `CrossSellTab`, `Item`, `Headers`, `CartBlob`, `Utility` class methods, and related POJOs for request and response objects  \nSuggested Test Scenarios:  \n- Happy path: Build request from valid headers and cart blob, verify all contexts populated correctly  \n- Edge case: Missing or empty headers/cart blob fields, verify defaults and no exceptions  \n- Parsing DDB blobs with only simple suggested items, only categorized suggested items, and both  \n- Serialization of suggested items with categories into DDB blob, verify JSON structure and TTL  \n- Generating response with and without cross-sell tabs, verify response correctness  \n- Validation of request context fields including requestId, clientId, timestamp, and channel  \n- Authentication context validation with missing or invalid tokens and TIDs  \n- Device context validation with missing or malformed user agent, platform, deviceId, or versionCode  \n- Location context validation with missing lat/lng or cityId  \n- Logging and exception handling when building request fails due to unexpected input  \n- Backward compatibility tests ensuring existing flows are unaffected by new fields and methods\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Introduction of Tabbed Cross-Sell Widget Support\n   - Category: Business logic update, API change\n   - A new concept of "tabbed" cross-sell widget is introduced, controlled by XP configuration and feature flags.\n   - The main method `getCrossSellSuggestion` now supports returning tabbed suggestions (`List<CrossSellTab> tabbedSuggestions`) alongside the existing flat list of suggestions.\n   - Pagination logic for suggested items is now conditional: if the widget is tabbed, the old pagination logic (fixed sizes 5,7,8) is bypassed.\n   - New private methods related to tabbed widget:\n     - `isCrossSellWidgetTabbed` determines if the request should use tabbed UI based on XP config.\n     - `populateCrossSellTabs` and related helpers organize suggested items into tabs by category.\n     - Logic to remove duplicates across tabs, fallback to non-tabbed if insufficient tabs/items.\n     - New constants for max/min items per tab and minimum required tabs.\n   - The `getSuggestionApiEndpoint` method now returns a different DSP endpoint if the widget is tabbed.\n   - The `getSuggestedItems` method now conditionally calls DSP service if DDB suggestions lack category info for tabbed users.\n   - New asynchronous method `getCrossSellSuggestionsAsync` added to support async fetching.\n   - New metrics for tabbed suggestions are recorded.\n   - New utility method `getPilledCrossSellWidgetVariant` fetches XP variant with fallback to feature flag default.\n   - New helper methods for filtering suggested items by category, availability, cart presence, and for building tabs.\n   - Modified `generateSuggestionsResponse` calls now include the new tabbed suggestions list.\n   - Modified imports and added new classes like `CrossSellTab`, `SuggestedItemCategory`, and utility classes.\n\n   Test implications:\n   - Need to test both tabbed and non-tabbed flows.\n   - Verify correct endpoint selection based on XP config and feature flags.\n   - Validate correct pagination and item limits in tabbed vs non-tabbed modes.\n   - Test fallback logic when insufficient tabs or items are present.\n   - Verify metrics are recorded correctly for tabbed and non-tabbed variants.\n   - Test async method behavior and exception handling.\n   - Validate category parsing and filtering logic.\n   - Test interaction with DDB and DSP services, especially category presence logic.\n   - Verify cart item removal and availability filtering in tabbed context.\n   - Validate XP config fetching and fallback behavior.\n\n2. Changes to `getSuggestionApiEndpoint`\n   - Category: Business logic update, API change\n   - Now checks if widget is tabbed and returns a specific tabbed endpoint.\n   - Otherwise, falls back to existing logic for enhancement or Pepsi upsell endpoints.\n   - This changes the DSP endpoint used for fetching suggestions.\n\n   Test implications:\n   - Test endpoint selection for tabbed vs non-tabbed users.\n   - Test behavior when XP config is missing or feature flag skips XP call.\n\n3. Changes to `getSuggestedItems`\n   - Category: Business logic update, Error handling change\n   - When fetching from DDB, if suggestions exist but lack category info and user is tabbed, DSP service is called to get fresh suggestions.\n   - This adds a fallback mechanism to ensure category data is present for tabbed users.\n   - Exception handling and metrics recording remain similar.\n\n   Test implications:\n   - Test DDB fetch with and without category data.\n   - Test fallback to DSP call for tabbed users.\n   - Test exception scenarios from DSP call.\n\n4. Pagination and Filtering Logic in `getCrossSellSuggestion`\n   - Category: Business logic update\n   - Pagination is skipped for tabbed widgets.\n   - Cart item removal and availability filtering are adapted for tabbed suggestions.\n   - If tabbed, suggestions are grouped into tabs; otherwise, old flat list logic applies.\n\n   Test implications:\n   - Test pagination behavior for tabbed and non-tabbed.\n   - Test cart item removal and availability filtering in both modes.\n   - Test empty and partial cart scenarios.\n\n5. New Helper Methods for Tabbed Suggestions\n   - Category: Business logic update\n   - Methods to categorize suggested items, limit items per tab, remove duplicates, fallback to non-tabbed if insufficient tabs.\n   - Methods to map SuggestedItems to MenuItems.\n   - Methods to set tab availability flags based on minimum items.\n   - Metrics collection for tabbed and fallback non-tabbed variants.\n\n   Test implications:\n   - Unit tests for helper methods to verify correct categorization, filtering, and tab construction.\n   - Test edge cases like all items in popular tab, insufficient items per tab, duplicates across tabs.\n\n6. New Async API Method `getCrossSellSuggestionsAsync`\n   - Category: API change, Performance improvement\n   - Provides asynchronous fetching of cross-sell suggestions.\n   - Handles exceptions and logs metrics on failure.\n\n   Test implications:\n   - Test async method correctness and exception handling.\n   - Test integration with callers expecting async response.\n\n7. Changes to `generateSuggestionsResponse` Calls\n   - Category: API change\n   - Now includes an additional parameter for tabbed suggestions list.\n   - Implies change in response data model or at least response construction.\n\n   Test implications:\n   - Verify response structure includes tabbed suggestions when applicable.\n   - Test backward compatibility or impact on clients consuming the response.\n\n8. New Constants and Feature Flags\n   - Category: Business logic update\n   - Constants for max/min items per tab, minimum tabs required.\n   - New feature flag checks for skipping XP calls or default variants.\n\n   Test implications:\n   - Test behavior under different feature flag states.\n   - Test boundary conditions for min/max items and tabs.\n\nSummary of impacted functions:\n- `getCrossSellSuggestion` (major changes)\n- `getSuggestionApiEndpoint` (endpoint selection logic)\n- `getSuggestedItems` (DDB and DSP fallback logic)\n- `populateCrossSellTabs` and related helpers (new tabbed UI logic)\n- `getCrossSellSuggestionsAsync` (new async API)\n- `getPilledCrossSellWidgetVariant` (XP config fetching)\n- Various filtering and mapping helpers for suggested items and tabs.\n\nModified inputs:\n- `getCrossSellSuggestion` now accepts same request but internally uses new XP config and feature flags to decide tabbed vs non-tabbed.\n- `getSuggestionApiEndpoint` input unchanged but output endpoint string changes based on tabbed logic.\n- `getSuggestedItems` input unchanged but internal logic changes based on tabbed and category presence.\n\nModified outputs:\n- `getCrossSellSuggestion` now returns `GetCrossSellSuggestionResponse` including tabbed suggestions list.\n- `generateSuggestionsResponse` calls now include tabbed suggestions list.\n- New async method returns `CompletableFuture<GetCrossSellSuggestionResponse>`.\n\nPotential downstream dependencies to test/integrate:\n- `CrossSellSuggestionHelper` methods (some usage unchanged)\n- `dspService.getItemSuggestion`\n- `crossSellSuggestionRepository.get` and `.save`\n- `cmsService.getMenuItems`\n- `availabilityService.getRegularItemAvailability`\n- `xpPlatformUtil.getXPDataForConfig`\n- `featureFlagService` methods\n\nOverall, this is a significant business logic update introducing a new tabbed cross-sell widget variant with associated changes in suggestion fetching, filtering, pagination, and response structure.\n\n</change_analysis>\n\nFile: CrossSellSuggestionService.java  \nChange Type: Business logic update, API change, Error handling change, Performance improvement  \nSummary: Introduces support for a new tabbed cross-sell widget variant with category-based suggestion tabs, conditional endpoint selection, and asynchronous fetching.  \nEndpoint/Function: `getCrossSellSuggestion`, `getSuggestionApiEndpoint`, `getSuggestedItems`, `getCrossSellSuggestionsAsync`, `populateCrossSellTabs` and related helpers  \nNew Behavior: The service now supports returning cross-sell suggestions organized into tabs by category when the tabbed widget variant is enabled via XP config and feature flags; it selects different DSP endpoints accordingly, falls back to DSP calls if DDB lacks category data, and provides an async API method. Pagination and filtering logic adapts based on widget variant.  \nTest Impact: High  \nDownstream Dependencies:  \n- `CrossSellSuggestionHelper.convertDdbResponseToSuggestedItems`  \n- `CrossSellSuggestionHelper.convertSuggestedItemsToDdbBlob`  \n- `CrossSellSuggestionHelper.convertSuggestionResponse`  \n- `CrossSellSuggestionHelper.generateSuggestionsResponse`  \n- `dspService.getItemSuggestion`  \n- `crossSellSuggestionRepository.get` and `.save`  \n- `cmsService.getMenuItems`  \n- `availabilityService.getRegularItemAvailability`  \n- `xpPlatformUtil.getXPDataForConfig`  \n- `featureFlagService` methods  \nModified Inputs:  \n- `GetCrossSellSuggestionRequest` (used with new XP config and feature flag logic)  \nModified Outputs:  \n- `GetCrossSellSuggestionResponse` now includes `List<CrossSellTab>` for tabbed suggestions  \n- New async method returns `CompletableFuture<GetCrossSellSuggestionResponse>`  \nContext Needed:  \n- `CrossSellSuggestionHelper` class and its methods  \n- `CrossSellSuggestionResponse` and `CrossSellTab` data models  \n- `SuggestedItemCategory` enum and related classes  \n- Feature flag configurations and XP platform utility  \nSuggested Test Scenarios:  \n- Happy path for non-tabbed widget: verify pagination, filtering, and response structure  \n- Happy path for tabbed widget: verify tab creation, category assignment, item limits per tab, and response structure  \n- Verify fallback to non-tabbed when insufficient tabs or items  \n- Test endpoint selection logic for tabbed and non-tabbed variants  \n- Test DDB fetch with and without category data, and DSP fallback for tabbed users  \n- Test cart item removal and availability filtering in both widget modes  \n- Test metrics recording for tabbed and non-tabbed suggestions  \n- Test async API method for success and exception scenarios  \n- Test XP config fetching and feature flag toggles affecting widget variant  \n- Edge cases: empty suggestions, all items already in cart, unavailable items, invalid categories in suggested items  \n- Performance and concurrency tests for async method and parallel API calls\n--------------------------------------------------------------------------------\nError: Could not get content for file src/main/java/com/swiggy/api/services/cart/cross_sell/CrossSellWidgetData.java\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of Cross-Sell Suggestion Integration in Cart Update Flow\n   - Category: Business logic update\n   - A new asynchronous call to fetch cross-sell suggestions is added in the updateCart method:\n     - A CompletableFuture<GetCrossSellSuggestionResponse> is created by invoking crossSellSuggestionService.getCrossSellSuggestionsAsync with a request built from headers and cartBlob.\n     - After cartOutput is prepared, the cross-sell suggestion response is populated into the baseCart via populateGetCrossSellSuggestionResponse.\n     - This method sets the cross-sell suggestion response on the baseCart and also populates XP data related to the "pilled cross sell widget" variant.\n     - Metrics are tracked for success and failure of this population.\n   - Functions affected/added:\n     - updateCart (modified to add cross-sell suggestion fetching and population)\n     - getCrossSellSuggestionResponseCompletableFuture (new helper method to fetch suggestions asynchronously)\n     - populateGetCrossSellSuggestionResponse (new helper method to populate response into cartOutput)\n     - populateXpDataForPilledCrossSellWidget (new helper method to set XP data related to cross-sell widget variant)\n   - Inputs modified:\n     - updateCart now internally calls crossSellSuggestionService with GetCrossSellSuggestionRequest derived from headers and cartBlob.\n   - Outputs modified:\n     - The BaseResponse returned by updateCart now includes cross-sell suggestion data embedded in the baseCart.\n   - Test implications:\n     - High: This is a significant business logic addition that impacts the cart update response payload by adding cross-sell suggestion data.\n     - Tests need to verify that cross-sell suggestions are fetched and correctly populated in the cart response.\n     - Error handling paths where cross-sell suggestions fail to fetch should be tested to ensure fallback behavior and metrics increment.\n     - XP data population for the pilled cross-sell widget variant needs validation.\n     - Interaction with feature flags or conditions controlling cross-sell invocation (if any) should be tested.\n   - Downstream dependencies:\n     - crossSellSuggestionService.getCrossSellSuggestionsAsync\n     - getCrossSellSuggestionRequest (static helper)\n     - Utility.getUserId()\n     - crossSellSuggestionService.getPilledCrossSellWidgetVariant\n\n2. Constructor Injection of CrossSellSuggestionService\n   - Category: Other (dependency injection update)\n   - The constructor of CartServiceImpl is updated to accept CrossSellSuggestionService and assign it to a new private final field.\n   - This enables the new cross-sell functionality.\n   - Test implications:\n     - Low: Mostly setup; ensure that the service is properly injected and null checks are handled.\n\n3. Imports and Static Imports Added\n   - Category: Other\n   - New imports related to cross-sell suggestion request/response and service added.\n   - Static imports for helper methods related to cross-sell suggestion.\n   - No direct test impact but necessary for compilation and functionality.\n\nSummary of impact:\n- The primary functional change is the integration of cross-sell suggestion data fetching and embedding into the cart update response.\n- This affects the updateCart method\'s behavior and response.\n- New asynchronous call and response handling added with error handling and metrics.\n- XP data related to cross-sell widget variant is populated in the cart.\n- No changes to API signatures or parameters exposed externally.\n- No changes to error handling or validation outside cross-sell integration.\n- No UI changes directly visible here but cross-sell data is likely consumed by UI downstream.\n\nPotential test areas:\n- Successful fetching and embedding of cross-sell suggestions in updateCart.\n- Behavior when cross-sell service fails or returns null.\n- Correct XP data population for pilled cross-sell widget variant.\n- Metrics increment on success and failure.\n- Interaction with existing cart update flow and no regressions.\n- Dependency injection correctness for the new service.\n\n</change_analysis>\n\nFile: CartServiceImpl.java  \nChange Type: Business logic update  \nSummary: Added asynchronous fetching and embedding of cross-sell suggestions into the cart update response.  \nEndpoint/Function: updateCart  \nNew Behavior: During cart update, the service asynchronously fetches cross-sell suggestions based on the cart and headers, populates the response into the baseCart, sets related XP data for the pilled cross-sell widget variant, and tracks success/failure metrics.  \nTest Impact: High  \nDownstream Dependencies:  \n- crossSellSuggestionService.getCrossSellSuggestionsAsync  \n- getCrossSellSuggestionRequest (static helper)  \n- populateGetCrossSellSuggestionResponse  \n- populateXpDataForPilledCrossSellWidget  \n- Utility.getUserId()  \n- crossSellSuggestionService.getPilledCrossSellWidgetVariant  \n\nModified Inputs:  \n- updateCart now internally uses Headers and CartBlob to build GetCrossSellSuggestionRequest  \nModified Outputs:  \n- BaseResponse from updateCart now includes cross-sell suggestion data embedded in baseCart  \nContext Needed:  \n- CrossSellSuggestionService class and its async method  \n- getCrossSellSuggestionRequest helper method  \n- CartOutput and BaseCart classes to verify new field for cross-sell suggestion response  \n- XP data handling in BaseCart  \nSuggested Test Scenarios:  \n- Happy path: updateCart returns cart with populated cross-sell suggestion data  \n- Cross-sell service returns null or empty response: cart update still succeeds without cross-sell data  \n- Cross-sell service throws exception: error is logged, metrics incremented, cart update succeeds gracefully  \n- XP data for pilled cross-sell widget variant is correctly populated when variant is present  \n- Metrics for cross-sell suggestion population success and failure are incremented appropriately  \n- Dependency injection of CrossSellSuggestionService works correctly  \n- Verify no regressions in existing cart update functionality and response structure\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new overloaded method:\n   - Method: `public static float moneyToFloat(com.swiggy.api.pojo.Money money)`\n   - Change Type: \n     - API change (new method overload)\n     - Error handling change (try-catch for NumberFormatException)\n   - Description:\n     - This method accepts a different Money class (`com.swiggy.api.pojo.Money`) which appears to have `String` types for units and an `int` for nanos, and converts it into the existing Google Money Proto format before calling the existing `moneyToFloat(Money money)` method.\n     - It includes error handling for `NumberFormatException` when parsing the units string to a long, logging the error and returning 0 in case of failure.\n   - Modified Inputs:\n     - Input parameter changed from `com.google.type.Money` to `com.swiggy.api.pojo.Money` (new method overload).\n   - Modified Outputs:\n     - Returns a float value representing the monetary amount, or 0 if parsing fails.\n   - Downstream Dependencies:\n     - Calls `moneyToFloat(Money money)` (existing method in the same class).\n     - Calls `Money.newBuilder()` from Google Money Proto.\n     - Uses logging via `log.error`.\n   - Test Implications:\n     - New tests needed to verify correct conversion from `com.swiggy.api.pojo.Money` to float.\n     - Tests to cover valid inputs (correct string units and nanos).\n     - Tests to cover invalid inputs (non-numeric units string) to verify error handling and that 0 is returned.\n     - Verify that logging occurs on error.\n     - Ensure consistency with existing `moneyToFloat(Money)` behavior.\n     - No changes to existing methods, but regression testing recommended to ensure no side effects.\n   - Additional Notes:\n     - The addition of Lombok\'s `@Slf4j` annotation enables logging, which is new for this class.\n     - No changes to existing business logic or calculations.\n     - No changes to currency code handling or precision.\n</change_analysis>\n\nFile: MoneyUtils.java  \nChange Type: API change, Error handling change  \nSummary: Added a new overloaded method to convert a different Money object (`com.swiggy.api.pojo.Money`) to float with error handling for invalid numeric conversion.  \nEndpoint/Function: `moneyToFloat(com.swiggy.api.pojo.Money money)`  \nNew Behavior: Converts a `com.swiggy.api.pojo.Money` instance (with String units) to a float by parsing the units string and nanos, returning 0 and logging an error if parsing fails.  \nTest Impact: Medium  \nDownstream Dependencies: `moneyToFloat(Money money)`, `Money.newBuilder()`, logging via `log.error`  \nModified Inputs: Added new method with input parameter of type `com.swiggy.api.pojo.Money` (units as String, nanos as int)  \nModified Outputs: Returns float value of money amount or 0 on parsing error  \nContext Needed: `com.swiggy.api.pojo.Money` class definition, existing `moneyToFloat(Money)` method, logging configuration  \nSuggested Test Scenarios:  \n- Valid `com.swiggy.api.pojo.Money` with numeric units string and valid nanos → returns correct float value  \n- Invalid `com.swiggy.api.pojo.Money` with non-numeric units string → returns 0 and logs error  \n- Boundary tests with zero, negative, and large values in units and nanos  \n- Consistency check with existing `moneyToFloat(Money)` method for equivalent values  \n- Verify no exceptions propagate outside the method on invalid input\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new static method `getPlatformFromUserAgent(String userAgent, UserAgentType userAgentType)` in the `Utility` class.\n\n   - Change Type: Business logic update\n   - Endpoint/Function affected: `Utility.getPlatformFromUserAgent`\n   - New Behavior:\n     - The method determines the platform (Android, iOS, Web, Dweb, or Invalid) based on the input parameters `userAgent` (String) and `userAgentType` (UserAgentType enum).\n     - Priority is given to the `userAgent` string to identify Android and iOS platforms.\n     - If the `userAgent` does not correspond to Android or iOS, the method uses `userAgentType` to determine if the platform is MWEB (mapped to PLATFORM_WEB) or DWEB (mapped to PLATFORM_DWEB).\n     - If none of the conditions match, it returns PLATFORM_INVALID.\n   - Modified Inputs:\n     - `userAgent` (String)\n     - `userAgentType` (UserAgentType)\n   - Modified Outputs:\n     - Returns a `Platform` enum value indicating the platform type.\n   - Downstream Dependencies:\n     - Uses `Utility.isNotEmpty` method to check if `userAgent` is not empty.\n     - Uses `UserAgentType` enum\'s `getValue()` method.\n     - Returns values from `Platform` enum.\n   - Test Implications:\n     - New method requires unit tests to verify correct platform determination based on various combinations of `userAgent` and `userAgentType`.\n     - Tests should cover:\n       - `userAgent` values "Swiggy-Android" and "Swiggy-iOS" returning PLATFORM_ANDROID and PLATFORM_IOS respectively.\n       - `userAgent` values other than Android/iOS with `userAgentType` as MWEB and DWEB returning PLATFORM_WEB and PLATFORM_DWEB respectively.\n       - Null or empty `userAgent` and null `userAgentType` returning PLATFORM_INVALID.\n       - Edge cases such as unknown `userAgent` strings and unknown `userAgentType` values.\n     - Since this is a utility method, it may be used downstream in platform-specific logic; testing should verify integration points where this method is called.\n   - Context Needed:\n     - `Platform` enum/class definition (to understand possible platform values).\n     - `UserAgentType` enum/class definition.\n     - Any existing usages of this method or similar platform detection logic in the codebase.\n</change_analysis>\n\nFile: Utility.java  \nChange Type: Business logic update  \nSummary: Added a new utility method to determine platform type from user agent string and user agent type enum.  \nEndpoint/Function: `Utility.getPlatformFromUserAgent(String userAgent, UserAgentType userAgentType)`  \nNew Behavior: Returns a platform enum value (PLATFORM_ANDROID, PLATFORM_IOS, PLATFORM_WEB, PLATFORM_DWEB, or PLATFORM_INVALID) based on input user agent string and user agent type.  \nTest Impact: Medium  \nDownstream Dependencies: `Utility.isNotEmpty`, `UserAgentType.getValue()`, `Platform` enum values  \nModified Inputs: `userAgent` (String), `userAgentType` (UserAgentType)  \nModified Outputs: Returns `Platform` enum value  \nContext Needed: `Platform` enum/class, `UserAgentType` enum/class, usages of platform detection logic  \nSuggested Test Scenarios:  \n- Input userAgent = "Swiggy-Android", userAgentType = null → expect PLATFORM_ANDROID  \n- Input userAgent = "Swiggy-iOS", userAgentType = null → expect PLATFORM_IOS  \n- Input userAgent = "someOtherAgent", userAgentType = MWEB → expect PLATFORM_WEB  \n- Input userAgent = "someOtherAgent", userAgentType = DWEB → expect PLATFORM_DWEB  \n- Input userAgent = null or empty, userAgentType = null → expect PLATFORM_INVALID  \n- Input userAgent = unknown string, userAgentType = unknown enum → expect PLATFORM_INVALID\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of two new test methods related to "Pilled Cross Sell Widget" feature:\n   - getPilledCrossSellWidgetDefaultVariantTestForWhitelistedUserIds\n   - getPilledCrossSellWidgetDefaultVariantTestForNonWhitelistedUserIds\n\nCategory: Business logic update\n\nComponent affected: FeatureFlagService class, specifically the method getPilledCrossSellWidgetDefaultVariant (implied, though method implementation is not shown in the diff, tests indicate existence).\n\nNew Behavior:\n- The feature flag service now includes logic to determine the default variant of the "Pilled Cross Sell Widget" based on user whitelist membership.\n- For whitelisted user IDs, the variant returned is a specific constant: PILLED_CROSS_SELL_WIDGET_TABBED_VARIANT.\n- For non-whitelisted user IDs, the variant returned is a default value fetched from configuration, with a fallback to "non_tabbed".\n\nTest Implications:\n- New test coverage is added for the user whitelist-based variant selection for the pilled cross sell widget.\n- Tests should verify correct variant is returned for both whitelisted and non-whitelisted user IDs.\n- Edge cases such as empty or null user IDs, or empty whitelist configurations, should be considered (not shown in the diff but implied for completeness).\n- Integration tests or end-to-end tests may need to verify UI behavior or feature toggling based on this variant.\n\nDownstream Dependencies:\n- getPilledCrossSellWidgetDefaultVariant (method under test)\n- checkoutOptions.getOption (to fetch whitelist and default variant config)\n- FeatureFlagService constants: PILLED_CROSS_SELL_WIDGET_TABBED_VARIANT, PILLED_CROSS_SELL_WIDGET_DEFAULT_VARIANT_DEFAULT_VALUE\n\nModified Inputs:\n- User ID string passed to getPilledCrossSellWidgetDefaultVariant\n\nModified Outputs:\n- String representing the default variant of the pilled cross sell widget\n\nContext Needed:\n- Implementation of getPilledCrossSellWidgetDefaultVariant method in FeatureFlagService\n- Constants related to pilled cross sell widget variants in FeatureFlagService\n- Configuration keys: "pilled_cross_sell_widget_enabled_user_ids", "pilled_cross_sell_widget_default_variant"\n\nSuggested Test Scenarios:\n- Verify variant returned is PILLED_CROSS_SELL_WIDGET_TABBED_VARIANT for user IDs in whitelist\n- Verify variant returned is default variant for user IDs not in whitelist\n- Verify behavior when whitelist is empty or null\n- Verify behavior when default variant config is missing or null\n- Verify behavior for null or empty user ID input\n- Verify integration with UI or feature toggling that depends on this variant\n</change_analysis>\n\nFile: FeatureFlagServiceTest.java  \nChange Type: Business logic update  \nSummary: Added tests for user whitelist-based selection of default variant for the pilled cross sell widget feature.  \nEndpoint/Function: FeatureFlagService.getPilledCrossSellWidgetDefaultVariant  \nNew Behavior: Returns a specific variant for whitelisted users and a default variant for others based on configuration.  \nTest Impact: Medium  \nDownstream Dependencies: FeatureFlagService.getPilledCrossSellWidgetDefaultVariant, checkoutOptions.getOption  \nModified Inputs: User ID string parameter to getPilledCrossSellWidgetDefaultVariant  \nModified Outputs: String variant representing the pilled cross sell widget default variant  \nContext Needed: Implementation of getPilledCrossSellWidgetDefaultVariant, constants and configuration keys related to pilled cross sell widget  \nSuggested Test Scenarios:  \n- Whitelisted user ID returns tabbed variant  \n- Non-whitelisted user ID returns default variant  \n- Empty or null whitelist configuration  \n- Missing or empty default variant configuration  \n- Null or empty user ID input  \n- Integration with UI or feature toggling mechanisms\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of import: \n   - Added import for `CrossSellTab` class.\n   - Category: Data model change (new data model class usage).\n   - Implication: Indicates that `CrossSellTab` is now part of the response model and should be tested for presence and correctness.\n\n2. Addition of new test method `cartSuggestionFallbackTest()`:\n   - A new test case was added to test the `cartSuggestionFallback` method of `cartSuggestionController`.\n   - Category: Error handling change (fallback behavior).\n   - Endpoint/Function affected: `cartSuggestionFallback` method in `CartSuggestionController`.\n   - New behavior: When fallback is triggered, the controller returns an empty `GetCrossSellSuggestionResponse` object.\n   - Test implications: Need to verify fallback behavior returns expected empty response, ensuring graceful degradation when main suggestion service fails.\n   - Inputs: `GetCrossSellSuggestionRequest` (same as existing request mocks).\n   - Outputs: Empty `GetCrossSellSuggestionResponse` with empty lists for suggestions and tabs, and `allCrossSellSuggestionsAdded` false.\n\n3. Modification in `mockEmptyResponse()` method:\n   - Added initialization of `crossSellTabs` to an empty list in the empty response mock.\n   - Category: Data model change (response schema extended).\n   - Implication: The response model now includes `crossSellTabs` field, which must be accounted for in tests verifying empty or fallback responses.\n   - Outputs: `GetCrossSellSuggestionResponse` now includes empty `crossSellTabs` list in addition to existing fields.\n\nSummary of key changes:\n- The response model `GetCrossSellSuggestionResponse` has been extended to include a new field `crossSellTabs` (likely a list of `CrossSellTab` objects).\n- A new fallback method `cartSuggestionFallback` has been introduced in the controller with a corresponding test, which returns an empty response with empty suggestions and tabs.\n- Existing tests for error handling verify that on exceptions, an empty response including empty `crossSellTabs` is returned.\n\nTest implications:\n- Existing tests need to be updated to verify the presence and correctness of the new `crossSellTabs` field in responses.\n- New tests should be added for the fallback method to ensure it returns the correct empty response structure.\n- Tests should verify that in error scenarios, the response includes empty `crossSellTabs`.\n- Tests should verify serialization/deserialization of the new `crossSellTabs` field.\n- Since this is a data model extension and fallback addition, no direct API signature changes are visible here, but the response schema is extended.\n\nDownstream dependencies visible in the file:\n- `cartSuggestionFallback` method of `cartSuggestionController` (new test added).\n- `mockEmptyResponse()` method (modified to include `crossSellTabs`).\n- `cartSuggestions` method (existing tests use it, no change shown here).\n\nModified inputs:\n- No changes to input parameters or request objects.\n\nModified outputs:\n- `GetCrossSellSuggestionResponse` now includes `crossSellTabs` list in empty and fallback responses.\n\nContext needed for further testing:\n- Implementation of `cartSuggestionFallback` method in `CartSuggestionController` (not shown here).\n- Definition and usage of `CrossSellTab` class and its role in the response.\n- Behavior of `crossSellSuggestionService.getCrossSellSuggestion()` in relation to `crossSellTabs`.\n- Any other controller methods that populate or use `crossSellTabs`.\n\n</change_analysis>\n\nFile: CartSuggestionControllerTest.java  \nChange Type:  \n- Data model change (addition of `crossSellTabs` field in response)  \n- Error handling change (addition of fallback method and test)  \n\nSummary:  \nAdded support for a new `crossSellTabs` field in the cross-sell suggestion response and introduced a fallback method in the controller returning an empty response including this new field.  \n\nEndpoint/Function:  \n- `cartSuggestionFallback` method in `CartSuggestionController`  \n- `mockEmptyResponse()` method (test helper)  \n\nNew Behavior:  \nThe controller now supports a fallback method that returns an empty `GetCrossSellSuggestionResponse` including empty `crossSellTabs`. The response model has been extended to include `crossSellTabs` in empty and fallback responses.  \n\nTest Impact:  \nMedium — requires new tests for fallback behavior and updates to existing tests to verify the presence and correctness of the new `crossSellTabs` field in responses, especially in empty and error scenarios.  \n\nDownstream Dependencies:  \n- `cartSuggestionFallback` method  \n- `mockEmptyResponse()` method  \n- `cartSuggestions` method (existing)  \n\nModified Inputs:  \n- None  \n\nModified Outputs:  \n- `GetCrossSellSuggestionResponse` now includes `crossSellTabs` list in empty and fallback responses.  \n\nContext Needed:  \n- Implementation details of `cartSuggestionFallback` in `CartSuggestionController`  \n- Definition and usage of `CrossSellTab` class  \n- How `crossSellTabs` is populated in normal and fallback scenarios  \n\nSuggested Test Scenarios:  \n- Happy path: Verify normal `cartSuggestions` response includes `crossSellTabs` when populated.  \n- Fallback path: Verify `cartSuggestionFallback` returns empty response with empty `crossSellTabs`.  \n- Error handling: Verify that when `crossSellSuggestionService` throws exceptions, the controller returns an empty response including empty `crossSellTabs`.  \n- Serialization: Verify JSON serialization/deserialization of `crossSellTabs` in responses.  \n- Edge cases: Empty input items, missing contexts, and verify fallback behavior.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of imports related to cross-selling:\n   - New imports added for classes like CatalogItem, CrossSellSuggestionV2, CrossSellTab, PricingInfo, VegClassifier, Money, GetCrossSellSuggestionResponse, SuggestedItemCategory, and CrossSellItemTab.\n   - Category: Data model change (new data structures introduced).\n   - Implication: The system now handles new data models related to cross-selling suggestions and tabs.\n\n2. Addition of multiple new test methods related to cross-selling widget presentation:\n   - testUserCartPresenterForPilledCrossSellWidgetTabbed\n   - testUserCartPresenterForPilledCrossSellWidgetNonTabbed\n   - testUserCartPresenterForPilledCrossSellWidgetNonTabbedNonXp\n   - These tests create mock cross-sell suggestion responses with multiple tabs or non-tabbed suggestions.\n   - They set the XP data for "pilled_cross_sell_widget" to "tabbed" or "non_tabbed" and verify the cart\'s cross-sell widget data accordingly.\n   - Category: Business logic update (handling of cross-sell widget data and XP data integration).\n   - Endpoint/Function affected: userCartPresenter.basicCartPresenter (main cart presentation logic).\n   - New Behavior: The cart presentation now supports rendering cross-sell widgets in tabbed or non-tabbed formats based on XP data and feature flag checks.\n   - The logic also respects a feature flag to skip XP data for pilled cross-sell widgets.\n   - Modified Inputs: BaseCart.getGetCrossSellSuggestionResponse (new cross-sell suggestion data), BaseCart.getXpData (with "pilled_cross_sell_widget" key).\n   - Modified Outputs: Cart.getCrossSellWidgetData (populated with tabs and suggestions), Cart.getXpData (used to determine widget format).\n   - Downstream dependencies: userCartPresenter.basicCartPresenter, featureFlagService.isPilledCrossSellWidgetXpSkipped().\n   - Test Implications: High - new data models and logic paths added for cross-sell widgets, requiring tests for tabbed/non-tabbed rendering, XP flag effects, and data correctness.\n\n3. Changes in test assertions:\n   - Assertions check that the cross-sell tabs and suggestions are correctly mapped from the new data structures.\n   - IDs are converted from String to Integer in assertions (e.g., popularCrossSellSuggestionV2.getCatalogItem().getId() is "1" in input but asserted as 1 in output).\n   - This implies a transformation in the presentation layer converting string IDs to integers.\n   - Category: Business logic update (data transformation).\n   - Test Implications: Medium - tests must verify correct data transformation and mapping.\n\nSummary of overall change:\n- Introduction of a new cross-sell widget feature that supports both tabbed and non-tabbed UI presentations.\n- The cart presenter now processes a new cross-sell suggestion response model and populates the cart with cross-sell tabs and suggestions accordingly.\n- Feature flag and XP data control whether the widget is shown and in which format.\n- Tests added to verify these behaviors and data mappings.\n\nNo changes to existing API signatures or error handling were observed in the diff.\n\nNo UI code is shown, but the cross-sell widget data is likely consumed by UI layers.\n\nNo security or performance changes evident.\n\nPotential test implications are mostly around validating new data models, feature flag behavior, XP data influence, and correct population of cart cross-sell widget data.\n\n</change_analysis>\n\nFile: BasicCartPresenterTest.java  \nChange Type: Business logic update, Data model change  \nSummary: Added support and tests for a new cross-sell widget feature with tabbed and non-tabbed presentations controlled by XP data and feature flags.  \nEndpoint/Function: userCartPresenter.basicCartPresenter (cart presentation logic)  \nNew Behavior: The cart presenter processes new cross-sell suggestion data structures to populate cross-sell widget data in the cart, supporting tabbed and non-tabbed layouts based on XP data and feature flags.  \nTest Impact: High  \nDownstream Dependencies: userCartPresenter.basicCartPresenter, featureFlagService.isPilledCrossSellWidgetXpSkipped()  \nModified Inputs: BaseCart.getGetCrossSellSuggestionResponse (new cross-sell suggestion response), BaseCart.getXpData (with "pilled_cross_sell_widget" key)  \nModified Outputs: Cart.getCrossSellWidgetData (populated cross-sell tabs and suggestions), Cart.getXpData (used for widget format decision)  \nContext Needed: userCartPresenter.basicCartPresenter method implementation, featureFlagService.isPilledCrossSellWidgetXpSkipped method, data model classes: GetCrossSellSuggestionResponse, CrossSellTab, CrossSellSuggestionV2, CatalogItem, CrossSellItemTab  \nSuggested Test Scenarios:  \n- Verify cart presentation with cross-sell widget in tabbed mode with multiple tabs and suggestions.  \n- Verify cart presentation with cross-sell widget in non-tabbed mode with flat suggestion list.  \n- Verify behavior when XP data indicates skipping the pilled cross-sell widget.  \n- Validate correct mapping and transformation of cross-sell suggestion data (IDs, names, veg classifiers, pricing).  \n- Edge cases: empty cross-sell suggestions, missing tabs, null XP data, feature flag disabled.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new test method `getItemSuggestionWithCategorySuccess()` in `DspServiceTest`:\n   - Category: Business logic update\n   - Description: This test verifies that the `getItemSuggestion` method of `DspService` correctly handles and returns a `SuggestedItem` that includes a new data field `category` along with existing fields `itemId` and `score`.\n   - Implications for testing: Tests must now verify that the `category` field is correctly populated and returned in the response object. This implies the underlying service or client may have been updated to support this new attribute.\n   - Modified outputs: The `ItemSuggestionResponse` now includes `SuggestedItem` objects with an additional `category` field.\n   - Downstream dependencies: `dspClient.getItemSuggestions()`, `dspService.getItemSuggestion()`\n   - Modified inputs: None explicitly changed in the test; mocks still use generic `Mockito.any()` for parameters.\n   - Modified outputs: `ItemSuggestionResponse` now includes `SuggestedItem.category` in addition to `itemId` and `score`.\n\n2. Addition of a new helper method `mockItemSuggestionWithCategoryResponse()`:\n   - Category: Data model change\n   - Description: This method creates a mock `ItemSuggestionResponse` with `SuggestedItem` objects that include the new `category` field set to `"Beverage"`.\n   - Implications for testing: Mock data now reflects the updated data model with category information, enabling tests to validate this new attribute.\n   - Modified outputs: Mock response includes `category` field in `SuggestedItem`.\n\n3. Minor refactoring of `mockItemSuggestionResponse()`:\n   - Category: Other (code style/refactoring)\n   - Description: The method now explicitly sets fields on `SuggestedItem` using setters rather than constructor, aligning with the style used in the new mock method.\n   - Implications for testing: No functional change, but improved consistency in mock object creation.\n\nSummary of overall impact:\n- The core business logic of `DspService.getItemSuggestion()` is expected to handle and return `SuggestedItem` objects that now include a `category` attribute.\n- Tests must be updated or added to verify the presence and correctness of this new field.\n- No changes to API parameters or error handling are evident.\n- No UI or performance changes.\n- The addition suggests an enhancement in the data model and business logic to support item categorization in suggestions.\n\nTesting implications:\n- New tests are required to validate the `category` field in successful responses.\n- Existing tests should be reviewed to ensure they do not break due to the new field.\n- Edge cases where `category` might be null or missing should be tested if applicable.\n- Integration tests may need to confirm that the DSP client and service correctly propagate this new field.\n\n</change_analysis>\n\nFile: DspServiceTest.java  \nChange Type: Business logic update, Data model change  \nSummary: Added support and testing for a new `category` field in `SuggestedItem` returned by `getItemSuggestion`.  \nEndpoint/Function: `DspService.getItemSuggestion()`, `DspServiceTest.getItemSuggestionWithCategorySuccess()`  \nNew Behavior: The service now returns suggested items with an additional `category` attribute, allowing consumers to receive item category information alongside item ID and score.  \nTest Impact: Medium  \nDownstream Dependencies: `dspClient.getItemSuggestions()`, `dspService.getItemSuggestion()`  \nModified Inputs: None (test inputs remain generic mocks)  \nModified Outputs: `ItemSuggestionResponse` now includes `SuggestedItem.category` in the response payload  \nContext Needed: `DspService.getItemSuggestion()`, `SuggestedItem` class definition, `DspClient.getItemSuggestions()`  \nSuggested Test Scenarios:  \n- Verify successful retrieval of item suggestions including the `category` field (happy path).  \n- Validate behavior when `category` is missing or null in the response.  \n- Confirm existing fields (`itemId`, `score`) remain unaffected.  \n- Test failure scenarios remain unchanged but should be revalidated to ensure no regression.  \n- Integration test to verify end-to-end propagation of the `category` field from client to service to consumer.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of a new static response constant RESPONSE_WITH_CATEGORIES_SUCCESS that includes an additional field "category" for each item in the cross_sell_cart_scores array.\n   - Category: Data model change\n   - Description: The response JSON now can include a "category" attribute per item, which was not present before.\n   - Test implications: Tests need to verify that the client correctly parses and includes the new "category" field when present, and that it handles null or missing categories gracefully.\n   - Modified Outputs: The JSON response now includes "category" fields in the cross_sell_cart_scores list.\n\n2. Modification of the existing test getResponse_Success to expect "category":null for each item in the response.\n   - Category: Data model change\n   - Description: The existing success test now expects the "category" field to be present but null, indicating that the data model has been extended to always include this field, even if empty.\n   - Test implications: Existing tests must be updated to expect the new field, ensuring backward compatibility and correct handling of missing category data.\n\n3. Addition of a new test method getResponseWithCategories_Success that:\n   - Uses the new RESPONSE_WITH_CATEGORIES_SUCCESS stub.\n   - Calls the client with endpoint "v1/predict/cart_cross_sell/v13" (note the version change from v9 to v13).\n   - Asserts that the response includes the "category" field with non-null values.\n   - Category: API change (endpoint version), Data model change, Business logic update (handling new field)\n   - Description: This test verifies that the client can handle a newer version of the API response which includes categories per item.\n   - Test implications: New tests must validate correct parsing and mapping of the "category" field, and that the client correctly handles the new endpoint version.\n   - Modified Inputs: Endpoint changed from "v1/predict/cart_cross_sell/v9" to "v1/predict/cart_cross_sell/v13"\n   - Modified Outputs: Response now includes "category" fields with string values.\n\nSummary of test impact:\n- Existing tests updated to expect the new "category" field as null.\n- New tests added to verify handling of the "category" field with actual values.\n- Tests should cover both old and new API versions to ensure backward compatibility.\n- Tests should verify correct deserialization of the new field and proper error handling if the field is missing or malformed.\n\nDownstream dependencies:\n- client.getItemSuggestions(String endpoint, ItemSuggestionRequest request)\n- ObjectMapper.writeValueAsString(Object)\n\nNo changes were made to error handling, performance, or security in this diff.\n\n</change_analysis>\n\nFile: DspClientTest.java  \nChange Type: Data model change, API change  \nSummary: Added support for a new "category" field in item suggestions and introduced a new API version endpoint to handle this enhanced response.  \nEndpoint/Function: DspClient.getItemSuggestions (tested via DspClientTest), endpoints "v1/predict/cart_cross_sell/v9" and "v1/predict/cart_cross_sell/v13"  \nNew Behavior: The client now expects and correctly parses an additional "category" attribute per item in the cross_sell_cart_scores list; the new API version v13 returns this enhanced data, while v9 returns null categories.  \nTest Impact: Medium  \nDownstream Dependencies: DspClient.getItemSuggestions(), ObjectMapper.writeValueAsString()  \nModified Inputs: Endpoint string parameter changed/added ("v1/predict/cart_cross_sell/v9" and "v1/predict/cart_cross_sell/v13")  \nModified Outputs: ItemSuggestionResponse now includes a "category" field per item, which can be null or a string value  \nContext Needed: DspClient class (to verify parsing logic for the new "category" field), ItemSuggestionResponse POJO (to confirm data model changes), API specification for v9 and v13 endpoints  \nSuggested Test Scenarios:  \n- Happy path: successful response from v9 endpoint with null categories  \n- Happy path: successful response from v13 endpoint with non-null categories  \n- Backward compatibility: ensure client handles missing "category" field gracefully  \n- Error handling: malformed "category" field or unexpected data types  \n- Integration test: verify correct deserialization and mapping of the new field in the response object  \n- Version handling: confirm client behavior when switching between v9 and v13 endpoints\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of import and usage of the method `convertDdbResponseToSuggestedItems` from `CrossSellSuggestionHelper`.\n   - Category: Business logic update\n   - Function affected: `testConvertSuggestedItemsToDdbBlob` (modified), new test `testConvertDdbResponseToSuggestedItemsWithCategory` added.\n   - New behavior: The serialization format of suggested items in the DDB blob now includes an additional JSON field `suggestedItemsWithCategory` which holds detailed suggested items with their `item_id`, `score`, and `category` fields. The test for `convertSuggestedItemsToDdbBlob` was updated to assert this new enriched JSON structure.\n   - The new test `testConvertDdbResponseToSuggestedItemsWithCategory` verifies that the method `convertDdbResponseToSuggestedItems` correctly parses this enriched JSON back into a list of `SuggestedItem` objects, including category information.\n   - Modified Inputs: The input to `convertSuggestedItemsToDdbBlob` remains the same (a list of `SuggestedItem`), but the output JSON structure now contains additional detailed information.\n   - Modified Outputs: The JSON string returned by `convertSuggestedItemsToDdbBlob` now includes the `suggestedItemsWithCategory` array with detailed objects.\n   - Downstream Dependencies: `convertSuggestedItemsToDdbBlob`, `convertDdbResponseToSuggestedItems`.\n   - Test implications: \n     - Existing tests for serialization need to be updated to expect the enriched JSON structure.\n     - New tests are required to verify deserialization from the new JSON format, including category fields.\n     - Edge cases where categories might be null or non-null should be tested.\n     - Validation of correct ordering and mapping of items in both serialization and deserialization.\n     - Backward compatibility or handling of older JSON formats (if applicable) should be considered.\n   - This change impacts the business logic around how suggested items are serialized/deserialized for storage or communication, adding category metadata.\n\nNo other changes to business logic, API, or data models were observed in the diff.\n\nSummary:\n- The main functional change is the enhancement of the suggested items serialization format to include category information and the addition of a corresponding deserialization method test.\n\n</change_analysis>\n\nFile: CrossSellSuggestionHelperTest.java  \nChange Type: Business logic update  \nSummary: Enhanced serialization and deserialization of suggested items to include category metadata in the JSON representation.  \nEndpoint/Function: `convertSuggestedItemsToDdbBlob`, `convertDdbResponseToSuggestedItems`, and their corresponding test methods `testConvertSuggestedItemsToDdbBlob` and `testConvertDdbResponseToSuggestedItemsWithCategory`.  \nNew Behavior: Suggested items are now serialized into a JSON string that includes both a simple list of item IDs and a detailed list with item IDs, scores, and categories; deserialization correctly reconstructs the detailed suggested items list including categories.  \nTest Impact: Medium - existing serialization tests need updating, and new deserialization tests must be added; tests should cover category presence/absence and correct mapping of fields.  \nDownstream Dependencies: `convertSuggestedItemsToDdbBlob`, `convertDdbResponseToSuggestedItems`  \nModified Inputs: None (input to serialization remains a list of SuggestedItem objects)  \nModified Outputs: JSON string format of suggested items now includes `suggestedItemsWithCategory` with detailed item info including category.  \nContext Needed: Implementation of `convertSuggestedItemsToDdbBlob` and `convertDdbResponseToSuggestedItems` methods in `CrossSellSuggestionHelper` class.  \nSuggested Test Scenarios:  \n- Serialization of suggested items with null categories.  \n- Serialization of suggested items with non-null categories.  \n- Deserialization of JSON with detailed suggested items including categories.  \n- Deserialization of JSON missing the `suggestedItemsWithCategory` field (backward compatibility).  \n- Verification of order preservation in serialization and deserialization.  \n- Handling of empty suggested items lists.\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of Cross-Sell Tabbed UI Support and Logic:\n   - Category: Business logic update, UI change\n   - A new concept of "CrossSellTab" and "SuggestedItemCategory" has been introduced and integrated into the cross-sell suggestion service.\n   - Multiple new tests have been added to verify the tabbed cross-sell suggestion behavior, including scenarios where:\n     - Tabs are shown when they have one or more items.\n     - Tabs are not shown if items are unavailable or if tabs have fewer than or equal to one item.\n     - Tabs are removed if their items overlap with popular items.\n     - Fallback to non-tabbed approach occurs under certain conditions (e.g., insufficient items in tabs, removal of tabs due to overlap).\n     - Sorting of items within tabs based on scores.\n   - The new behavior involves grouping cross-sell suggestions into tabs based on categories (like Dessert, Beverage, Side), and conditionally showing or hiding these tabs depending on item availability, count, and overlap with popular items.\n   - The response model now includes a list of CrossSellTabs (`crossSellTabs`), in addition to the existing crossSellSuggestions list.\n   - The mockDspResponse method was updated to include category information for SuggestedItems.\n   - The mockEmptyResponse method now initializes an empty list of CrossSellTabs.\n   - Feature flags related to "pilledCrossSellWidget" and default variant ("tabbed") are used to enable this new behavior.\n   - Downstream dependencies include:\n     - dspService.getItemSuggestion()\n     - cmsService.getMenuItems()\n     - availabilityService.getRegularItemAvailability()\n     - crossSellSuggestionRepository.get() and save()\n     - xpPlatformUtil methods for attributes and config\n   - Inputs modified: The request now can include multiple items with categories, and feature flags influence behavior.\n   - Outputs modified: The response now includes `crossSellTabs` with grouped suggestions.\n   - Testing implications:\n     - High impact: New UI grouping logic requires extensive testing of tab creation, filtering, fallback behavior, and sorting.\n     - Need to test feature flag toggling and its effect on tabbed vs non-tabbed responses.\n     - Test item availability impact on tab visibility.\n     - Test overlap with popular items and its effect on tab removal.\n     - Test sorting of items within tabs by score.\n     - Test fallback scenarios when tabs are removed or have insufficient items.\n     - Test response structure includes new `crossSellTabs` field.\n     - Validate behavior with different combinations of item categories and availability.\n     - Verify integration with DSP and CMS services with respect to categories.\n     - Verify that existing non-tabbed behavior remains intact when tabbed is disabled or fallback occurs.\n\n2. Minor Data Model Change:\n   - Category: Data model change\n   - Addition of CrossSellTab and SuggestedItemCategory imports and usage.\n   - The response model now includes a new field `crossSellTabs` (list of CrossSellTab).\n   - The mockEmptyResponse method initializes this new field.\n   - This is a backward-compatible addition but requires validation in tests to ensure the new field is handled correctly.\n\n3. Updates to Mock Data for DSP Responses:\n   - Category: Business logic update\n   - SuggestedItem now includes a category field (e.g., "Dessert", "Beverage", "Side").\n   - This affects how suggestions are grouped and displayed in tabs.\n   - Tests now mock DSP responses with categorized SuggestedItems to verify tabbed behavior.\n\nSummary of test impact:\n- The major change is the introduction of a tabbed cross-sell suggestion UI and logic, controlled via feature flags.\n- The service behavior changes significantly when the tabbed variant is enabled, including grouping, filtering, and fallback logic.\n- The response structure includes a new field for tabs.\n- Tests must cover multiple scenarios around tab visibility, item availability, category overlaps, sorting, and fallback.\n- Existing tests for non-tabbed behavior remain relevant but should be verified to coexist with new tabbed logic.\n\nFunctions called by changed code (downstream dependencies):\n- dspService.getItemSuggestion()\n- cmsService.getMenuItems()\n- availabilityService.getRegularItemAvailability()\n- crossSellSuggestionRepository.get()\n- crossSellSuggestionRepository.save()\n- xpPlatformUtil.getAttributes()\n- xpPlatformUtil.getXPDataForConfig()\n- sessionService.getSessionData()\n\nModified inputs/parameters:\n- GetCrossSellSuggestionRequest now includes multiple items with categories.\n- Feature flags: isPilledCrossSellWidgetXpSkipped(), getPilledCrossSellWidgetDefaultVariant() influence behavior.\n\nModified outputs/return values:\n- GetCrossSellSuggestionResponse now includes `crossSellTabs` (List<CrossSellTab>) in addition to existing crossSellSuggestions.\n- crossSellTabs contain grouped suggestions by category with filtering and sorting applied.\n\nContext needed for further analysis:\n- CrossSellSuggestionService class implementation (especially getCrossSellSuggestion method)\n- CrossSellTab and SuggestedItemCategory class definitions\n- FeatureFlagService methods related to pilledCrossSellWidget\n- DSP service response handling and mapping logic\n- CMS service menu items structure and filtering\n- AvailabilityService item availability logic\n\nSuggested test scenarios:\n- Happy path: Tabbed cross-sell suggestions returned with multiple tabs, each with multiple available items.\n- Tabs with all items unavailable should not be shown; fallback to non-tabbed suggestions.\n- Tabs with only one or zero items should not be shown; fallback to non-tabbed suggestions.\n- Tabs with items overlapping popular items should be removed.\n- If removal of tabs due to overlap results in fewer than minimum tabs, fallback to non-tabbed suggestions.\n- Sorting of items within tabs by score.\n- Feature flag off: no tabs, only non-tabbed suggestions.\n- DSP service failures or empty responses with tabbed variant enabled.\n- CMS service returns empty or null menu items.\n- Availability service marks some items unavailable.\n- Verify response includes crossSellTabs field correctly populated or empty.\n- Verify fallback logic triggers correctly in edge cases.\n- Verify addon groups and other item details remain intact in tabbed suggestions.\n- Verify that existing non-tabbed tests still pass when tabbed variant is disabled.\n\n</change_analysis>\n\nFile: CrossSellSuggestionServiceTest.java  \nChange Type: Business logic update, UI change, Data model change  \nSummary: Introduces tabbed cross-sell suggestion grouping by item category with conditional tab visibility and fallback to non-tabbed suggestions.  \nEndpoint/Function: CrossSellSuggestionService.getCrossSellSuggestion() and its test class CrossSellSuggestionServiceTest  \nNew Behavior: Cross-sell suggestions are grouped into tabs based on item categories when enabled via feature flags; tabs are conditionally shown or hidden based on item availability, count, and overlap with popular items; fallback to non-tabbed suggestions occurs when tabs are insufficient or removed.  \nTest Impact: High  \nDownstream Dependencies: dspService.getItemSuggestion(), cmsService.getMenuItems(), availabilityService.getRegularItemAvailability(), crossSellSuggestionRepository.get(), crossSellSuggestionRepository.save(), xpPlatformUtil.getAttributes(), xpPlatformUtil.getXPDataForConfig(), sessionService.getSessionData()  \nModified Inputs: GetCrossSellSuggestionRequest.items with categories, feature flags (isPilledCrossSellWidgetXpSkipped, getPilledCrossSellWidgetDefaultVariant)  \nModified Outputs: GetCrossSellSuggestionResponse.crossSellTabs (List<CrossSellTab>) added alongside existing crossSellSuggestions  \nContext Needed: CrossSellSuggestionService class (getCrossSellSuggestion method), CrossSellTab and SuggestedItemCategory classes, FeatureFlagService methods for pilledCrossSellWidget, DSP service response processing, CMSService menu item handling, AvailabilityService availability logic  \nSuggested Test Scenarios:  \n- Verify tabbed suggestions with multiple tabs and items (happy path)  \n- Verify tabs hidden when items unavailable or count ≤ 1, fallback to non-tabbed  \n- Verify tabs removed if items overlap popular items, and fallback if tabs < min count  \n- Verify sorting of items within tabs by score  \n- Verify behavior when feature flags disable tabbed variant  \n- Verify handling of DSP, CMS, and availability service failures or empty responses  \n- Verify response structure includes crossSellTabs correctly  \n- Verify addon groups and item details in tabbed suggestions  \n- Verify fallback logic correctness in edge cases\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of CrossSellSuggestionService mock and initialization in CartServiceImplTest:\n   - Category: Business logic update\n   - Description: The test class CartServiceImplTest now includes a mock for CrossSellSuggestionService and sets up a default behavior for its getCrossSellSuggestionsAsync method to return a completed future with null.\n   - Implications for testing:\n     - Tests involving cart updates or creations might now implicitly depend on cross-sell suggestion logic being invoked asynchronously.\n     - Although the current tests do not explicitly verify interactions with crossSellSuggestionService, the presence of this mock suggests that the production code (CartServiceImpl) may have been updated to call this service.\n     - Test coverage should be extended to verify that cross-sell suggestions are requested appropriately during cart operations.\n     - Error handling and fallback scenarios for the cross-sell service should be tested if applicable.\n   - Downstream dependencies:\n     - crossSellSuggestionService.getCrossSellSuggestionsAsync()\n   - Modified inputs/outputs:\n     - No direct changes in method signatures or return types in the test code.\n     - Indirectly, cart update/create flows may now trigger cross-sell suggestion fetching.\n   - Context needed:\n     - CartServiceImpl class implementation to confirm where and how crossSellSuggestionService is used.\n     - CrossSellSuggestionService interface and implementation details.\n     - Any changes in CartServiceImpl methods related to cart update/create flows.\n\nSummary:\nOnly one change is present in the diff: the addition of a mock for CrossSellSuggestionService and its initialization in the test setup. This indicates a new asynchronous call to fetch cross-sell suggestions integrated into the cart service logic, which requires corresponding test coverage to ensure proper invocation and handling.\n\nNo API, data model, UI, error handling, or security changes are evident from the diff. The change is purely a business logic integration of a new service call within the cart update/create flows.\n\n</change_analysis>\n\nFile: CartServiceImplTest.java  \nChange Type: Business logic update  \nSummary: Added integration of CrossSellSuggestionService to asynchronously fetch cross-sell suggestions during cart operations.  \nEndpoint/Function: CartServiceImpl (specifically cart update and create operations tested in CartServiceImplTest)  \nNew Behavior: Cart service now triggers asynchronous fetching of cross-sell suggestions via CrossSellSuggestionService during cart updates or creations.  \nTest Impact: Medium (new service integration requires validation of invocation, handling of async results, and error scenarios)  \nDownstream Dependencies: crossSellSuggestionService.getCrossSellSuggestionsAsync()  \nModified Inputs: None directly in test methods; possible implicit addition in cart update/create flows  \nModified Outputs: None directly in test methods; possible side effects in cart response or state  \nContext Needed:  \n- CartServiceImpl class source code to identify where crossSellSuggestionService is invoked  \n- CrossSellSuggestionService interface and implementation  \n- Any changes in cart update/create methods related to cross-sell suggestions  \nSuggested Test Scenarios:  \n- Verify that cross-sell suggestions are requested asynchronously during cart creation and update flows  \n- Validate behavior when cross-sell suggestions are successfully fetched (e.g., response enrichment or state update)  \n- Test fallback or error handling when cross-sell suggestion fetching fails or times out  \n- Confirm no regression in existing cart update/create functionality without cross-sell suggestions  \n- Performance impact of asynchronous cross-sell fetching on cart operations\n--------------------------------------------------------------------------------\n<change_analysis>\n1. Addition of the test method `moneyToFloatTest()`:\n   - Category: Business logic update\n   - Function/component affected: `MoneyUtils.moneyToFloat(com.swiggy.api.pojo.Money)`\n   - New behavior: Validates that the utility method `moneyToFloat` correctly converts a `Money` object (with currency code, units as string, and nanos) into a float representation of the monetary value.\n   - Inputs: A `Money` object with currency code "INR", units "12", and nanos 340000000.\n   - Outputs: A float value 12.34f.\n   - Test implications: Requires verification that the conversion logic from the `Money` POJO to float is accurate and consistent with the expected monetary value. This is a new functionality test, so new test coverage is needed.\n\n2. Addition of the test method `moneyToFloatExceptionTest()`:\n   - Category: Error handling change\n   - Function/component affected: `MoneyUtils.moneyToFloat(com.swiggy.api.pojo.Money)`\n   - New behavior: Tests that the `moneyToFloat` method throws a `NumberFormatException` when the `units` field of the `Money` object is an empty string.\n   - Inputs: A `Money` object with currency code "INR", units as an empty string `""`, and nanos 0.\n   - Outputs: Exception thrown (`NumberFormatException`).\n   - Test implications: This introduces an error handling scenario that must be tested to ensure the method correctly handles invalid input and throws the expected exception. This is a negative test case that complements the positive conversion test.\n\nSummary of changes:\n- New test coverage added for the `moneyToFloat` method in `MoneyUtils`.\n- Tests both normal conversion and exception scenarios.\n- No changes to existing methods or APIs; purely additive test cases.\n- The `moneyToFloat` method is assumed to be a new or existing utility method converting a POJO `Money` object to a float monetary value.\n\nDownstream dependencies:\n- `MoneyUtils.moneyToFloat(com.swiggy.api.pojo.Money)` is the key function under test.\n- The `Money` POJO class from `com.swiggy.api.pojo` is used as input.\n\nModified inputs:\n- New test inputs involving `Money` objects with specific fields (currency code, units string, nanos).\n\nModified outputs:\n- Return float value representing money.\n- Exception thrown on invalid input.\n\nTest implications:\n- New test scenarios must be added for both successful conversion and exception handling.\n- Validation of correct float conversion precision.\n- Validation of exception type and message on invalid input.\n\nContext needed:\n- Implementation details of `MoneyUtils.moneyToFloat`.\n- Definition of `com.swiggy.api.pojo.Money` class.\n- Existing tests or usages of `moneyToFloat` if any.\n\nSuggested test scenarios:\n- Happy path: Convert valid `Money` object with non-empty units and nanos to float.\n- Edge case: Convert `Money` object with zero nanos.\n- Error case: Convert `Money` object with empty or invalid units string, expect `NumberFormatException`.\n- Boundary case: Very large or very small monetary values.\n- Currency code variations (if applicable).\n\n</change_analysis>\n\nFile: MoneyUtilsTest.java  \nChange Type: Business logic update, Error handling change  \nSummary: Added tests for the `moneyToFloat` method to verify correct float conversion from a `Money` POJO and to ensure proper exception handling when invalid input is provided.  \nEndpoint/Function: `MoneyUtils.moneyToFloat(com.swiggy.api.pojo.Money)`  \nNew Behavior: Converts a `Money` object to a float value representing the monetary amount; throws `NumberFormatException` if the units string is invalid.  \nTest Impact: Medium  \nDownstream Dependencies: `MoneyUtils.moneyToFloat`  \nModified Inputs: `Money` object with fields `currencyCode`, `units` (string), `nanos` (int)  \nModified Outputs: float monetary value, or exception (`NumberFormatException`)  \nContext Needed: `MoneyUtils.moneyToFloat` implementation, `com.swiggy.api.pojo.Money` class definition  \nSuggested Test Scenarios:  \n- Valid conversion from `Money` to float (e.g., "12" units and 340000000 nanos → 12.34f)  \n- Exception thrown when units string is empty or non-numeric  \n- Conversion with zero nanos  \n- Conversion with boundary values (large/small amounts)  \n- Handling of different currency codes (if relevant)\n--------------------------------------------------------------------------------\n'
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
            model_settings={"parallel_tool_calls": True, "max_tokens": 8000},
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
