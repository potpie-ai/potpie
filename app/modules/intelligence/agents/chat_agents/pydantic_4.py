import re
import asyncio
from typing import List, AsyncGenerator, Dict, Any, Optional, Callable, Set, Tuple

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

# Core pydantic_ai imports
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

logger = setup_logger(__name__)

# Constants
MAX_FIX_ATTEMPTS = 5
STREAM_CHANGE_THRESHOLD = 50  # Number of characters changed to trigger a stream update


class GraphNode:
    """A node in the execution graph that can call tools"""

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str = "",
        required_params: List[str] | None = None,
    ):
        self.name = name
        self.func = func
        self.description = description
        self.required_params = required_params or []
        self.input_connections: Dict[str, Tuple[str, str]] = (
            {}
        )  # {input_param: (from_node, output_field)}
        self.outputs: Set[str] = set()  # Fields this node returns

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node function with appropriate inputs from state"""
        # Extract required inputs from state based on connections
        inputs = {}

        # Check for missing required parameters
        missing_params = []

        # First, try to get inputs from connections
        for param, (node_name, field) in self.input_connections.items():
            key = f"{node_name}.{field}"
            if key in state:
                inputs[param] = state[key]
            elif field in state:  # Also check for direct field
                inputs[param] = state[field]

        # Check for any direct parameters in state that match required params
        for param in self.required_params:
            if param not in inputs and param in state:
                inputs[param] = state[param]

        # Verify all required parameters are present
        for param in self.required_params:
            if param not in inputs:
                missing_params.append(param)

        if missing_params:
            param_list = ", ".join(missing_params)
            raise ValueError(
                f"Node {self.name} missing required parameters: {param_list}"
            )

        # Pass any additional context in state
        if "context" in state and "context" not in inputs:
            inputs["context"] = state["context"]

        # Execute function with gathered inputs
        result = await self.func(**inputs)

        # Update state with prefixed keys
        for key, value in result.items():
            self.outputs.add(key)
            state[f"{self.name}.{key}"] = value
            state[key] = value  # Also store direct key for convenience

        return result


class CustomGraph:
    """Execution graph for tool-enabled agent workflow"""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[Tuple[str, str]] = []  # [(from_node, to_node)]
        self.execution_order: List[str] = (
            []
        )  # Topological sort (filled in when finalized)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.name] = node

    def connect(
        self, from_node: str, output_field: str, to_node: str, input_param: str
    ) -> None:
        """Connect an output from one node to an input of another"""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(
                f"Cannot connect: nodes {from_node} or {to_node} not found"
            )

        self.edges.append((from_node, to_node))
        self.nodes[to_node].input_connections[input_param] = (from_node, output_field)

    def finalize(self) -> None:
        """Finalize the graph and compute execution order (topological sort)"""
        # Simple topological sort
        visited = set()
        temp = set()
        order = []

        def visit(node: str) -> None:
            if node in temp:
                raise ValueError(f"Graph has a cycle involving node {node}")
            if node in visited:
                return

            temp.add(node)

            # Visit all destinations of edges from this node
            for src, dst in self.edges:
                if src == node:
                    visit(dst)

            temp.remove(node)
            visited.add(node)
            order.append(node)

        # Visit all nodes
        for node in self.nodes:
            if node not in visited:
                visit(node)

        # Reverse to get correct execution order
        self.execution_order = order[::-1]

    async def stream_execute_with_loop(
        self,
        initial_state: Dict[str, Any],
        loop_condition: Callable[[Dict[str, Any], str], bool],
        loop_from: str,
        loop_to: str,
        max_iterations: int = 5,
        stream_handler: Optional[Callable[[str, Dict[str, Any], int], None]] = None,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any], int], None]:
        """
        Stream execution with looping, yielding node name, results, and iteration count.
        Optionally calls stream_handler for each result for real-time processing.
        """
        if not self.execution_order:
            self.finalize()

        state = initial_state.copy()
        iteration = 0
        skip_to_output = False  # Flag to skip directly to output if needed

        try:
            # Execute up to loop_from first
            start_idx = 0
            to_idx = self.execution_order.index(loop_from)

            for node_name in self.execution_order[start_idx:to_idx]:
                node = self.nodes[node_name]
                try:
                    node_result = await node.execute(state)
                    if stream_handler:
                        await stream_handler(node_name, node_result, 0)
                    yield node_name, node_result, 0
                except Exception as e:
                    error_result = {
                        "error": str(e),
                        "node": node_name,
                        "success": False,
                    }
                    if stream_handler:
                        await stream_handler(node_name, error_result, 0)
                    yield node_name, error_result, 0
                    logger.error(f"Error in node {node_name}: {str(e)}", exc_info=True)

            # Now handle the loop section
            while iteration < max_iterations and not skip_to_output:
                iteration += 1
                loop_from_idx = self.execution_order.index(loop_from)

                for node_idx, node_name in enumerate(
                    self.execution_order[loop_from_idx:]
                ):
                    node = self.nodes[node_name]
                    try:
                        node_result = await node.execute(state)
                        if stream_handler:
                            await stream_handler(node_name, node_result, iteration)
                        yield node_name, node_result, iteration

                        # Check verification result - if verified is True,
                        # we need to skip to output directly
                        if node_name == "verify" and node_result.get("verified", False):
                            skip_to_output = True
                            break

                        # Check loop condition after the loop_to node executes
                        if node_name == loop_to and loop_condition(state, node_name):
                            # Continue looping
                            break

                        # If we've reached the end without breaking, stop looping
                        if node_name == self.execution_order[-1]:
                            iteration = max_iterations  # Force exit

                    except Exception as e:
                        error_result = {
                            "error": str(e),
                            "node": node_name,
                            "success": False,
                        }
                        if stream_handler:
                            await stream_handler(node_name, error_result, iteration)
                        yield node_name, error_result, iteration
                        skip_to_output = True  # On error, skip to output
                        logger.error(
                            f"Error in node {node_name}: {str(e)}", exc_info=True
                        )
                        break

                # Stop if we've hit max iterations
                if iteration >= max_iterations:
                    break

            # ALWAYS run the output node at the end if we have one
            output_idx = (
                self.execution_order.index("output")
                if "output" in self.execution_order
                else -1
            )
            if output_idx >= 0:
                output_node = self.nodes["output"]
                try:
                    output_result = await output_node.execute(state)
                    if stream_handler:
                        await stream_handler("output", output_result, iteration)
                    yield "output", output_result, iteration
                except Exception as e:
                    error_result = {"error": str(e), "success": False}
                    if stream_handler:
                        await stream_handler("output", error_result, iteration)
                    yield "output", error_result, iteration
                    logger.error(f"Error in output node: {str(e)}", exc_info=True)

        except Exception as e:
            error_result = {"error": str(e), "success": False}
            if stream_handler:
                await stream_handler("error", error_result, 0)
            yield "error", error_result, 0
            logger.error(f"Error in graph execution: {str(e)}", exc_info=True)


class ToolCallingGraphAgent:
    """Agent that uses a custom graph for execution flow with tool calling support"""

    def __init__(
        self, llm: Any, graph: CustomGraph, tools: List[Tool], system_prompt: str
    ):
        self.llm = llm
        self.graph = graph
        self.tools = tools
        self.system_prompt = system_prompt

        # Create a pydantic_ai agent specifically for tool calls
        self.agent = Agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            result_type=str,
            retries=3,
            defer_model_check=True,
            end_strategy="exhaustive",
            model_settings={"parallel_tool_calls": True, "max_tokens": 8000},
        )

    async def execute_with_tools(
        self, prompt: str, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Execute the agent with the given prompt and gather tool calls and responses"""
        if context is None:
            context = {}

        tool_calls = []
        tool_results = []
        output_text = ""

        # Use an async context manager for the pydantic_ai agent to stream the execution
        try:
            async with self.agent.iter(user_prompt=prompt) as stream:
                async for node in stream:
                    # Handle model output (text generation)
                    if self.agent.is_model_request_node(node):
                        async with node.stream(stream.ctx) as req_stream:
                            async for evt in req_stream:
                                if isinstance(evt, PartStartEvent) and isinstance(
                                    evt.part, TextPart
                                ):
                                    output_text += evt.part.content
                                elif isinstance(evt, PartDeltaEvent) and isinstance(
                                    evt.delta, TextPartDelta
                                ):
                                    output_text += evt.delta.content_delta

                    # Handle tool calls
                    elif self.agent.is_call_tools_node(node):
                        async with node.stream(stream.ctx) as handle_stream:
                            async for evt in handle_stream:
                                # Capture tool call
                                if isinstance(evt, FunctionToolCallEvent):
                                    tool_call = {
                                        "call_id": evt.part.tool_call_id or "",
                                        "tool_name": evt.part.tool_name,
                                        "args": evt.part.args_as_dict(),
                                        "summary": get_tool_call_info_content(
                                            evt.part.tool_name,
                                            evt.part.args_as_dict(),
                                        ),
                                    }
                                    tool_calls.append(tool_call)

                                # Capture tool result
                                if isinstance(evt, FunctionToolResultEvent):
                                    tool_result = {
                                        "call_id": evt.result.tool_call_id or "",
                                        "tool_name": evt.result.tool_name
                                        or "unknown tool",
                                        "content": evt.result.content,
                                        "summary": get_tool_result_info_content(
                                            evt.result.tool_name or "unknown tool",
                                            evt.result.content,
                                        ),
                                    }
                                    tool_results.append(tool_result)
        except Exception as e:
            logger.error(f"Error during tool execution: {str(e)}", exc_info=True)
            return {
                "output": f"Error during execution: {str(e)}",
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "error": str(e),
            }

        return {
            "output": output_text,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    async def run(self, **initial_state) -> Dict[str, Any]:
        """
        Run the graph execution end-to-end with tool calling support.

        Args:
            **initial_state: Initial state for the graph execution

        Returns:
            Dict containing the final results and tool information
        """
        logger.info("Running ToolCallingGraphAgent with graph execution")

        try:
            # Define loop condition - should return True to continue looping
            def loop_condition(state: Dict[str, Any], node_name: str) -> bool:
                return (
                    state.get("continue_loop", False)
                    and state.get("attempt_count", 0) < MAX_FIX_ATTEMPTS
                )

            # Track accumulated tool data
            all_tool_calls = []
            all_tool_results = []

            # Execute the graph with looping
            last_node_result = {}

            # Run graph with looping from verify back to fix if needed
            async for (
                node_name,
                node_result,
                iteration,
            ) in self.graph.stream_execute_with_loop(
                initial_state=initial_state,
                loop_condition=loop_condition,
                loop_from="verify",
                loop_to="fix",
                max_iterations=MAX_FIX_ATTEMPTS,
            ):
                # Store the last node result
                last_node_result = node_result

                # Accumulate tool calls and results
                if "tool_calls" in node_result:
                    all_tool_calls.extend(node_result["tool_calls"])
                if "tool_results" in node_result:
                    all_tool_results.extend(node_result["tool_results"])

                # For verification failure, log the issues
                if node_name == "verify" and not node_result.get("verified", True):
                    logger.info(
                        f"Verification failed: {node_result.get('issue_report', 'No details')}"
                    )

                # For fix attempts, log the attempt count
                if node_name == "fix":
                    logger.info(
                        f"Fix attempt {node_result.get('attempt_count', 0)}/{MAX_FIX_ATTEMPTS}"
                    )

            # Build the final result
            final_result = {}

            # Extract the output from the last node result (should be output node)
            if "output" in last_node_result:
                final_result["output"] = last_node_result["output"]
            elif "fixed_solution" in last_node_result:
                final_result["output"] = last_node_result["fixed_solution"]
            elif "solution" in last_node_result:
                final_result["output"] = last_node_result["solution"]
            else:
                final_result["output"] = "Unable to generate a proper solution."

            # Add all accumulated tool calls and results
            final_result["tool_calls"] = all_tool_calls
            final_result["tool_results"] = all_tool_results

            # Add execution metadata
            final_result["execution_completed"] = True

            logger.info("ToolCallingGraphAgent execution completed successfully")
            return final_result

        except Exception as e:
            logger.error(
                f"Error in ToolCallingGraphAgent run method: {str(e)}", exc_info=True
            )
            return {
                "output": f"Error during execution: {str(e)}",
                "tool_calls": [],
                "tool_results": [],
                "execution_completed": False,
                "error": str(e),
            }

    async def run_stream(
        self, **initial_state
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Stream the graph execution with tool calls"""

        # Define loop condition - should return True to continue looping
        def loop_condition(state: Dict[str, Any], node_name: str) -> bool:
            return (
                state.get("continue_loop", False)
                and state.get("attempt_count", 0) < MAX_FIX_ATTEMPTS
            )

        # Stream graph with looping
        async for (
            node_name,
            node_result,
            iteration,
        ) in self.graph.stream_execute_with_loop(
            initial_state=initial_state,
            loop_condition=loop_condition,
            loop_from="verify",
            loop_to="fix",
            max_iterations=MAX_FIX_ATTEMPTS,
        ):
            # Add any additional derived data to the node result
            # For example, for the output node, ensure tool information is complete
            if node_name == "output" and "tool_calls" in initial_state:
                # Make sure all tool calls are included in the final output
                if "tool_calls" not in node_result:
                    node_result["tool_calls"] = []
                node_result["tool_calls"].extend(initial_state.get("tool_calls", []))

                # Same for tool results
                if "tool_results" not in node_result:
                    node_result["tool_results"] = []
                node_result["tool_results"].extend(
                    initial_state.get("tool_results", [])
                )

            # Stream the node result
            yield node_name, node_result


class PydanticToolGraphAgent(ChatAgent):
    """
    A graph-based agent that uses tools to solve complex tasks with planning,
    implementation, verification, and fixing steps.

    This agent supports streaming output from each step of the graph execution,
    allowing for real-time feedback during the entire process.
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ):
        """
        Initialize a tool-enabled graph agent with planning, implementation,
        verification, and fix steps.

        This agent uses a custom graph to:
        1. Plan a strategy to solve the user's query with tools
        2. Implement a solution (using tools)
        3. Verify the solution (using tools if needed)
        4. Fix any issues (with looping back to verify again if needed)
        5. Generate final output

        Args:
            llm_provider: Provider for language model
            config: Configuration for the agent
            tools: List of structured tools available to the agent
        """
        self.config = config
        self.tasks = config.tasks
        self.tools = tools
        self.llm_provider = llm_provider

        # Build the execution graph
        self.graph = self._build_graph()

        # Create the GraphAgent with pydantic_ai tools
        self.agent = ToolCallingGraphAgent(
            llm=llm_provider.get_pydantic_model(),
            graph=self.graph,
            tools=[
                Tool(
                    name=re.sub(r" ", "", tool.name),
                    description=tool.description,
                    function=tool.func,  # type: ignore
                )
                for tool in tools
            ],
            system_prompt=f"Role: {config.role}\nGoal: {config.goal}\nBackstory: {config.backstory}. Respond to the user query",
        )

        # Initialize streaming state
        self.stream_state = {
            "current_node": None,
            "last_content": {},
            "tool_calls_seen": set(),
            "phase_labels": {
                "plan": "📝 Planning",
                "implement": "⚙️ Implementing",
                "verify": "🔍 Verification",
                "fix": "🔄 Fixing",
                "output": "🏁 Final Result",
            },
        }

    async def _plan_step(
        self, task: str, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """
        PLAN NODE: Create a plan for solving the task using relevant tools.

        Args:
            task: The task description
            context: Additional context information

        Returns:
            Dict containing plan, tool calls, and other metadata
        """
        if not context:
            context = {}

        # Create a prompt that encourages tool usage for planning
        plan_prompt = f"""
        PLANNING PHASE:
        Given the following task, create a detailed plan for how to approach it.
        Use tools to gather information that will help you create a better plan.
        
        TASK: {task}
        
        Your response should include:
        1. A step-by-step plan
        2. Which tools you'll need to use and why
        3. Any specific information you need to extract
        
        Begin by using tools to explore available information, then formulate your plan.
        """

        # Call agent with tools enabled
        result = await self.agent.execute_with_tools(
            prompt=plan_prompt, context=context
        )

        # Extract tool calls and results for history
        tool_calls = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])

        # Extract chat context for history
        chat_context = []
        if tool_calls and tool_results:
            # Create a list of tuples (call, result) for each tool call
            for i, call in enumerate(tool_calls):
                if i < len(tool_results):
                    chat_context.append(
                        (
                            get_tool_call_info_content(call["tool_name"], call["args"]),
                            get_tool_result_info_content(
                                call["tool_name"], tool_results[i]["content"]
                            ),
                        )
                    )

        return {
            "plan": result["output"],
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "chat_context": chat_context,
            "task": task,
            "context": context,
            "attempt_count": 0,
        }

    async def _implement_step(
        self,
        plan: str,
        task: str,
        context: Dict[str, Any],
        attempt_count: int,
        tool_calls: List[Dict] | None = None,
        tool_results: List[Dict] | None = None,
        chat_context: List | None = None,
    ) -> Dict[str, Any]:
        """
        IMPLEMENT NODE: Implement a solution based on the plan using tools.

        Args:
            plan: The planning output
            task: The task description
            context: Additional context information
            attempt_count: Current attempt number
            tool_calls: Previous tool calls
            tool_results: Previous tool results
            chat_context: Previous chat context

        Returns:
            Dict containing solution, tool calls, and other metadata
        """
        if tool_calls is None:
            tool_calls = []
        if tool_results is None:
            tool_results = []
        if chat_context is None:
            chat_context = []

        # Build history from chat_context for continuity
        history_text = ""
        for call_info, result_info in chat_context:
            history_text += f"Tool Call: {call_info}\nTool Result: {result_info}\n\n"

        # Create a prompt that encourages tool usage for implementation
        implement_prompt = f"""
        IMPLEMENTATION PHASE:
        Implement a solution for this task following your plan. Use tools as needed to retrieve information or perform operations.
        
        TASK: {task}
        
        YOUR PLAN:
        {plan}
        
        TOOL USAGE HISTORY:
        {history_text if history_text else "No previous tool usage."}
        
        Start implementing your solution by calling necessary tools and clearly documenting what you're doing.
        Focus on translating your plan into concrete actions and a clear solution.
        """

        # Call agent with tools enabled and history context
        result = await self.agent.execute_with_tools(
            prompt=implement_prompt, context=context
        )

        # Extract new tool calls and results
        new_tool_calls = result.get("tool_calls", [])
        new_tool_results = result.get("tool_results", [])

        # Update chat context with new tool interactions
        new_chat_context = list(chat_context)  # Copy existing context
        if new_tool_calls and new_tool_results:
            for i, call in enumerate(new_tool_calls):
                if i < len(new_tool_results):
                    new_chat_context.append(
                        (
                            get_tool_call_info_content(call["tool_name"], call["args"]),
                            get_tool_result_info_content(
                                call["tool_name"], new_tool_results[i]["content"]
                            ),
                        )
                    )

        # Combine tool calls and results for history
        all_tool_calls = tool_calls + new_tool_calls
        all_tool_results = tool_results + new_tool_results

        return {
            "solution": result["output"],
            "plan": plan,
            "task": task,
            "context": context,
            "attempt_count": attempt_count,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "chat_context": new_chat_context,
        }

    async def _verify_step(
        self,
        solution: str,
        plan: str,
        task: str,
        context: Dict[str, Any],
        attempt_count: int,
        tool_calls: List[Dict] | None = None,
        tool_results: List[Dict] | None = None,
        chat_context: List | None = None,
    ) -> Dict[str, Any]:
        """
        VERIFY NODE: Check the solution for errors or issues using tools if needed.

        Args:
            solution: The implemented solution
            plan: The planning output
            task: The task description
            context: Additional context information
            attempt_count: Current attempt number
            tool_calls: Previous tool calls
            tool_results: Previous tool results
            chat_context: Previous chat context

        Returns:
            Dict containing verification results, tool calls, and other metadata
        """
        if tool_calls is None:
            tool_calls = []
        if tool_results is None:
            tool_results = []
        if chat_context is None:
            chat_context = []

        # Build history from chat_context for continuity
        history_text = ""
        for call_info, result_info in chat_context:
            history_text += f"Tool Call: {call_info}\nTool Result: {result_info}\n\n"

        # Create a prompt that enables tool usage for verification
        verification_prompt = f"""
        VERIFICATION PHASE:
        Carefully review this solution and identify any issues or errors.
        Use tools if needed to verify correctness or gather additional information.
        
        TASK: {task}
        
        PLAN: {plan}
        
        SOLUTION TO VERIFY:
        {solution}
        
        TOOL USAGE HISTORY:
        {history_text if history_text else "No tool usage history."}
        
        Check for:
        1. Completeness - Does it fully address the task?
        2. Correctness - Are there any logical errors or bugs?
        3. Style - Is it well-formatted and clear?
        
        First, state whether the solution is VERIFIED (no issues) or NEEDS FIXING.
        If NEEDS FIXING, provide a detailed explanation of each issue.
        You may use tools to help with verification if needed.
        """

        # Call agent with tools enabled and history context
        result = await self.agent.execute_with_tools(
            prompt=verification_prompt, context=context
        )

        # Extract new tool calls and results
        new_tool_calls = result.get("tool_calls", [])
        new_tool_results = result.get("tool_results", [])

        # Update chat context with new tool interactions
        new_chat_context = list(chat_context)  # Copy existing context
        if new_tool_calls and new_tool_results:
            for i, call in enumerate(new_tool_calls):
                if i < len(new_tool_results):
                    new_chat_context.append(
                        (
                            get_tool_call_info_content(call["tool_name"], call["args"]),
                            get_tool_result_info_content(
                                call["tool_name"], new_tool_results[i]["content"]
                            ),
                        )
                    )

        # Combine tool calls and results for history
        all_tool_calls = tool_calls + new_tool_calls
        all_tool_results = tool_results + new_tool_results

        # Determine if verified by checking for key phrases
        needs_fixing = any(
            phrase in result["output"].lower()
            for phrase in [
                "needs fixing",
                "issues found",
                "needs improvement",
                "error",
                "bug",
                "fix needed",
            ]
        )

        return {
            "solution": solution,
            "plan": plan,
            "task": task,
            "context": context,
            "issue_report": result["output"],
            "verified": not needs_fixing,
            "attempt_count": attempt_count,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "chat_context": new_chat_context,
        }

    async def _fix_step(
        self,
        solution: str,
        issue_report: str,
        plan: str,
        task: str,
        context: Dict[str, Any],
        attempt_count: int,
        verified: bool,
        tool_calls: List[Dict] | None = None,
        tool_results: List[Dict] | None = None,
        chat_context: List | None = None,
    ) -> Dict[str, Any]:
        """
        FIX NODE: Address issues identified during verification with tool support.

        Args:
            solution: The implemented solution
            issue_report: The verification findings
            plan: The planning output
            task: The task description
            context: Additional context information
            attempt_count: Current attempt number
            verified: Whether the solution is verified
            tool_calls: Previous tool calls
            tool_results: Previous tool results
            chat_context: Previous chat context

        Returns:
            Dict containing fixed solution, tool calls, and other metadata
        """
        if tool_calls is None:
            tool_calls = []
        if tool_results is None:
            tool_results = []
        if chat_context is None:
            chat_context = []

        # Short-circuit if already verified or max attempts reached
        if verified:
            return {
                "fixed_solution": solution,
                "continue_loop": False,
                "attempt_count": attempt_count,
                "final": True,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "chat_context": chat_context,
            }

        if attempt_count >= MAX_FIX_ATTEMPTS:
            return {
                "fixed_solution": f"After {MAX_FIX_ATTEMPTS} attempts, here's the best solution:\n\n{solution}",
                "continue_loop": False,
                "attempt_count": attempt_count,
                "final": True,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "chat_context": chat_context,
            }

        # Increment attempt count
        new_attempt_count = attempt_count + 1

        # Build history from chat_context for continuity
        history_text = ""
        for call_info, result_info in chat_context:
            history_text += f"Tool Call: {call_info}\nTool Result: {result_info}\n\n"

        # Create a prompt that enables tool usage for fixes
        fix_prompt = f"""
        FIX PHASE (Attempt {new_attempt_count}/{MAX_FIX_ATTEMPTS}):
        Fix the following solution based on the issues identified.
        Use tools as needed to improve the solution or verify corrections.
        
        TASK: {task}
        
        ORIGINAL PLAN: {plan}
        
        CURRENT SOLUTION:
        {solution}
        
        ISSUES IDENTIFIED:
        {issue_report}
        
        TOOL USAGE HISTORY:
        {history_text if history_text else "No tool usage history."}
        
        Please provide a completely fixed version that addresses ALL identified issues.
        Don't just describe the fixes - implement them and provide the complete corrected solution.
        Use tools as needed to verify your corrections or gather additional information.
        """

        # Call agent with tools enabled and history context
        result = await self.agent.execute_with_tools(prompt=fix_prompt, context=context)

        # Extract new tool calls and results
        new_tool_calls = result.get("tool_calls", [])
        new_tool_results = result.get("tool_results", [])

        # Update chat context with new tool interactions
        new_chat_context = list(chat_context)  # Copy existing context
        if new_tool_calls and new_tool_results:
            for i, call in enumerate(new_tool_calls):
                if i < len(new_tool_results):
                    new_chat_context.append(
                        (
                            get_tool_call_info_content(call["tool_name"], call["args"]),
                            get_tool_result_info_content(
                                call["tool_name"], new_tool_results[i]["content"]
                            ),
                        )
                    )

        # Combine tool calls and results for history
        all_tool_calls = tool_calls + new_tool_calls
        all_tool_results = tool_results + new_tool_results

        return {
            "fixed_solution": result["output"],
            "continue_loop": True,  # Signal to continue the loop
            "attempt_count": new_attempt_count,
            "final": False,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "chat_context": new_chat_context,
        }

    async def _output_step(
        self,
        solution: Optional[str] = None,
        fixed_solution: Optional[str] = None,
        final: bool = True,
        context: Any = None,
        tool_calls: List[Dict] | None = None,
        tool_results: List[Dict] | None = None,
        chat_context: List | None = None,
    ) -> Dict[str, Any]:
        """
        OUTPUT NODE: Prepare the final response with tool calls.

        Args:
            solution: The implemented solution
            fixed_solution: The fixed solution (if any)
            final: Whether this is the final output
            context: Additional context information
            tool_calls: Previous tool calls
            tool_results: Previous tool results
            chat_context: Previous chat context

        Returns:
            Dict containing the final output and tool calls
        """
        # Choose the best solution
        if tool_calls is None:
            tool_calls = []
        if tool_results is None:
            tool_results = []
        if chat_context is None:
            chat_context = []

        final_output = fixed_solution if fixed_solution else solution
        if not final_output:
            final_output = "Unable to generate a proper solution."

        # Return the output and tool calls
        return {
            "output": final_output,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def _build_graph(self) -> CustomGraph:
        """
        Build the execution graph with five nodes and proper connections.

        Returns:
            CustomGraph: The execution graph with all nodes and connections
        """
        # Create the graph
        graph = CustomGraph()

        # Create nodes with async callables and required parameters
        plan_node = GraphNode(
            name="plan",
            func=self._plan_step,
            description="Planning phase - create a strategy for solving the task",
            required_params=["task"],
        )

        implement_node = GraphNode(
            name="implement",
            func=self._implement_step,
            description="Implementation phase - execute the plan to solve the task",
            required_params=["plan", "task", "context", "attempt_count"],
        )

        verify_node = GraphNode(
            name="verify",
            func=self._verify_step,
            description="Verification phase - check for issues in the solution",
            required_params=["solution", "plan", "task", "context", "attempt_count"],
        )

        fix_node = GraphNode(
            name="fix",
            func=self._fix_step,
            description="Fix phase - address any issues found during verification",
            required_params=[
                "solution",
                "issue_report",
                "plan",
                "task",
                "context",
                "attempt_count",
                "verified",
            ],
        )

        output_node = GraphNode(
            name="output",
            func=self._output_step,
            description="Output phase - prepare the final response",
        )

        # Add nodes to graph
        graph.add_node(plan_node)
        graph.add_node(implement_node)
        graph.add_node(verify_node)
        graph.add_node(fix_node)
        graph.add_node(output_node)

        # Connect nodes - with all required fields
        # Plan -> Implement connections
        graph.connect("plan", "plan", "implement", "plan")
        graph.connect("plan", "task", "implement", "task")
        graph.connect("plan", "context", "implement", "context")
        graph.connect("plan", "attempt_count", "implement", "attempt_count")
        graph.connect("plan", "tool_calls", "implement", "tool_calls")
        graph.connect("plan", "tool_results", "implement", "tool_results")
        graph.connect("plan", "chat_context", "implement", "chat_context")

        # Implement -> Verify connections
        graph.connect("implement", "solution", "verify", "solution")
        graph.connect("implement", "plan", "verify", "plan")
        graph.connect("implement", "task", "verify", "task")
        graph.connect("implement", "context", "verify", "context")
        graph.connect("implement", "attempt_count", "verify", "attempt_count")
        graph.connect("implement", "tool_calls", "verify", "tool_calls")
        graph.connect("implement", "tool_results", "verify", "tool_results")
        graph.connect("implement", "chat_context", "verify", "chat_context")

        # Verify -> Fix connections
        graph.connect("verify", "solution", "fix", "solution")
        graph.connect("verify", "issue_report", "fix", "issue_report")
        graph.connect("verify", "plan", "fix", "plan")
        graph.connect("verify", "task", "fix", "task")
        graph.connect("verify", "context", "fix", "context")
        graph.connect("verify", "attempt_count", "fix", "attempt_count")
        graph.connect("verify", "verified", "fix", "verified")
        graph.connect("verify", "tool_calls", "fix", "tool_calls")
        graph.connect("verify", "tool_results", "fix", "tool_results")
        graph.connect("verify", "chat_context", "fix", "chat_context")

        # Fix -> Output and Verify -> Output connections
        graph.connect("fix", "fixed_solution", "output", "fixed_solution")
        graph.connect("fix", "final", "output", "final")
        graph.connect("fix", "tool_calls", "output", "tool_calls")
        graph.connect("fix", "tool_results", "output", "tool_results")
        graph.connect("fix", "chat_context", "output", "chat_context")

        # Direct connection from verify to output (for when solution is verified on first try)
        graph.connect("verify", "solution", "output", "solution")
        graph.connect("verify", "verified", "output", "final")
        graph.connect("verify", "tool_calls", "output", "tool_calls")
        graph.connect("verify", "tool_results", "output", "tool_results")
        graph.connect("verify", "chat_context", "output", "chat_context")

        # Finalize the graph
        graph.finalize()

        return graph

    def _create_task_input(
        self, task_config: TaskConfig, ctx: ChatContext
    ) -> Dict[str, Any]:
        """
        Create the task input for the graph agent with proper context.

        Args:
            task_config: The task configuration
            ctx: The chat context

        Returns:
            Dict containing the task description and context
        """
        if ctx.node_ids is None:
            ctx.node_ids = []
        if isinstance(ctx.node_ids, str):
            ctx.node_ids = [ctx.node_ids]

        # Create detailed context
        context = {
            "query": ctx.query,
            "project_id": ctx.project_id,
            "project_name": ctx.project_name,
            "node_ids": ctx.node_ids,
            "additional_context": (
                ctx.additional_context
                if ctx.additional_context != ""
                else "no additional context"
            ),
            "task_description": task_config.description,
            "expected_output": task_config.expected_output,
            "history": ctx.history,
        }

        task_description = f"""
        CONTEXT:
        User Query: {ctx.query}
        Project ID: {ctx.project_id}
        Node IDs: {" ,".join(ctx.node_ids)}
        Project Name (this is name from github. i.e. owner/repo): {ctx.project_name}
        
        Additional Context:
        {ctx.additional_context if ctx.additional_context != "" else "no additional context"}
        
        TASK:
        {task_config.description}
        
        Expected Output:
        {task_config.expected_output}
        
        INSTRUCTIONS:
        1. Use the available tools to gather information
        2. Process and synthesize the gathered information
        3. Format your response in markdown, make sure it's well formatted
        4. Include relevant code snippets and file references
        5. Provide clear explanations
        6. Verify your output before submitting
        
        IMPORTANT:
        - Use tools efficiently and avoid unnecessary API calls
        - Only use the tools listed below
        
        With above information answer the user query: {ctx.query}
        """

        return {"task": task_description, "context": context}

    def _process_tool_calls_for_response(
        self, tool_calls_data: List[Dict]
    ) -> List[ToolCallResponse]:
        """
        Process tool calls for ChatAgentResponse format.

        Args:
            tool_calls_data: List of tool call dictionaries

        Returns:
            List of ToolCallResponse objects
        """
        tool_calls = []
        for call in tool_calls_data:
            if "tool_name" in call and "args" in call:
                tc = ToolCallResponse(
                    call_id=call.get("call_id", ""),
                    event_type=ToolCallEventType.CALL,
                    tool_name=call["tool_name"],
                    tool_response=get_tool_run_message(call["tool_name"]),
                    tool_call_details={
                        "summary": get_tool_call_info_content(
                            call["tool_name"], call["args"]
                        )
                    },
                )
                tool_calls.append(tc)
        return tool_calls

    def _should_stream_update(self, node_name: str, content: str) -> bool:
        """
        Determine if a stream update should be sent based on content changes.

        Args:
            node_name: The current node name
            content: The current content

        Returns:
            bool: True if update should be streamed
        """
        # Always stream if node changes
        if node_name != self.stream_state["current_node"]:
            return True

        # Always stream output node
        if node_name == "output":
            return True

        # Get previous content
        last_content = self.stream_state["last_content"].get(node_name, "")

        # Stream if content has changed significantly
        content_change = abs(len(content) - len(last_content))
        return content_change > STREAM_CHANGE_THRESHOLD

    def _get_content_key_for_node(self, node_name: str) -> Optional[str]:
        """
        Get the appropriate content key for a node.

        Args:
            node_name: The node name

        Returns:
            Optional[str]: The content key or None
        """
        content_key_map = {
            "plan": "plan",
            "implement": "solution",
            "verify": "issue_report",
            "fix": "fixed_solution",
            "output": "output",
        }
        return content_key_map.get(node_name)

    async def _process_node_for_streaming(
        self, node_name: str, node_result: Dict[str, Any], iteration: int
    ) -> Optional[ChatAgentResponse]:
        """
        Process a node result and prepare a streaming response if needed.

        Args:
            node_name: The node name
            node_result: The node result
            iteration: The current iteration

        Returns:
            Optional[ChatAgentResponse]: Response to stream or None
        """
        # Check if this node has streamable content
        content_key = self._get_content_key_for_node(node_name)
        if not content_key or content_key not in node_result:
            return None

        # Get current content
        content = node_result.get(content_key, "")

        # Check if we should stream an update
        if not self._should_stream_update(node_name, content):
            return None

        # Update stream state
        self.stream_state["current_node"] = node_name
        self.stream_state["last_content"][node_name] = content

        # Process any new tool calls and their results
        tool_call_responses = []

        # First, process any tool calls
        if "tool_calls" in node_result and "tool_results" in node_result:
            tool_calls = node_result["tool_calls"]
            tool_results = node_result["tool_results"]

            # Process each tool call and find its matching result
            for call in tool_calls:
                call_id = call.get("call_id", "")

                # Skip if we've already seen this call
                if call_id in self.stream_state["tool_calls_seen"]:
                    continue

                # Mark as seen
                self.stream_state["tool_calls_seen"].add(call_id)

                # Add the tool call to the response
                if "tool_name" in call and "args" in call:
                    # First add the tool call
                    tool_call_responses.append(
                        ToolCallResponse(
                            call_id=call_id,
                            event_type=ToolCallEventType.CALL,
                            tool_name=call["tool_name"],
                            tool_response=get_tool_run_message(call["tool_name"]),
                            tool_call_details={
                                "summary": get_tool_call_info_content(
                                    call["tool_name"], call["args"]
                                )
                            },
                        )
                    )

                    # Then find and add the matching result
                    for result in tool_results:
                        if result.get("call_id", "") == call_id:
                            tool_call_responses.append(
                                ToolCallResponse(
                                    call_id=call_id,
                                    event_type=ToolCallEventType.RESULT,
                                    tool_name=call["tool_name"],
                                    tool_response=get_tool_response_message(
                                        call["tool_name"]
                                    ),
                                    tool_call_details={
                                        "summary": get_tool_result_info_content(
                                            call["tool_name"], result.get("content", "")
                                        )
                                    },
                                )
                            )
                            break

        # Format the response based on node type
        phase_label = self.stream_state["phase_labels"].get(
            node_name, f"Processing {node_name.capitalize()}"
        )

        # Add verification status for verify node
        if node_name == "verify":
            if node_result.get("verified", False):
                phase_label = "✅ Verification (Passed)"
            else:
                phase_label = "❌ Verification (Issues Found)"

        # Add attempt count for fix node
        if node_name == "fix" and "attempt_count" in node_result:
            phase_label = f"{phase_label} (Attempt {node_result['attempt_count']}/{MAX_FIX_ATTEMPTS})"

        # Format response text
        if node_name == "output":
            response_text = content
        else:
            response_text = f"{phase_label}...\n\n{content}"

        return ChatAgentResponse(
            response=response_text,
            tool_calls=tool_call_responses,
            citations=[],  # No citations as per requirements
        )

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """
        Run the graph agent with planning, implementation, verification and fix steps.

        Args:
            ctx: The chat context

        Returns:
            ChatAgentResponse: The final response
        """
        logger.info("Running pydantic tool graph agent")
        try:
            # Create input for the graph agent
            task_input = self._create_task_input(self.tasks[0], ctx)

            # Run the graph agent
            result = await self.agent.run(**task_input)

            # Extract the final output
            final_output = result.get("output", "Unable to generate response")

            # Process tool calls for response
            tool_calls = self._process_tool_calls_for_response(
                result.get("tool_calls", [])
            )

            return ChatAgentResponse(
                response=final_output,
                tool_calls=tool_calls,
                citations=[],  # No citations as per requirements
            )
        except Exception as e:
            logger.error(
                f"Error in pydantic tool graph agent run method: {str(e)}",
                exc_info=True,
            )
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """
        Stream the graph agent execution, showing each step in real-time with tool calls.

        Args:
            ctx: The chat context

        Yields:
            ChatAgentResponse: Streamed responses from each step
        """
        logger.info("Running pydantic tool graph agent stream")

        # Reset streaming state
        self.stream_state = {
            "current_node": None,
            "last_content": {},
            "tool_calls_seen": set(),
            "phase_labels": {
                "plan": "📝 Planning",
                "implement": "⚙️ Implementing",
                "verify": "🔍 Verification",
                "fix": "🔄 Fixing",
                "output": "🏁 Final Result",
            },
        }

        # Create input for the graph agent
        task_input = self._create_task_input(self.tasks[0], ctx)

        try:
            # Stream execution
            async for node_name, node_result in self.agent.run_stream(**task_input):
                # Process tool calls and results directly for immediate streaming
                if "tool_calls" in node_result and len(node_result["tool_calls"]) > 0:
                    # Process each tool call that hasn't been seen yet
                    for tool_call in node_result["tool_calls"]:
                        call_id = tool_call.get("call_id", "")

                        # Skip tool calls we've already seen
                        if call_id in self.stream_state["tool_calls_seen"]:
                            continue

                        # Mark this call as seen
                        self.stream_state["tool_calls_seen"].add(call_id)

                        # Create a tool call response and yield it immediately
                        if "tool_name" in tool_call and "args" in tool_call:
                            tool_call_response = ChatAgentResponse(
                                response="",  # Empty response as we're just streaming a tool call
                                tool_calls=[
                                    ToolCallResponse(
                                        call_id=call_id,
                                        event_type=ToolCallEventType.CALL,
                                        tool_name=tool_call["tool_name"],
                                        tool_response=get_tool_run_message(
                                            tool_call["tool_name"]
                                        ),
                                        tool_call_details={
                                            "summary": get_tool_call_info_content(
                                                tool_call["tool_name"],
                                                tool_call["args"],
                                            )
                                        },
                                    )
                                ],
                                citations=[],
                            )
                            # Yield the tool call immediately
                            yield tool_call_response

                            # Find and yield the corresponding tool result if available
                            if "tool_results" in node_result:
                                for tool_result in node_result["tool_results"]:
                                    if tool_result.get("call_id", "") == call_id:
                                        tool_result_response = ChatAgentResponse(
                                            response="",  # Empty response as we're just streaming a tool result
                                            tool_calls=[
                                                ToolCallResponse(
                                                    call_id=call_id,
                                                    event_type=ToolCallEventType.RESULT,
                                                    tool_name=tool_call["tool_name"],
                                                    tool_response=get_tool_response_message(
                                                        tool_call["tool_name"]
                                                    ),
                                                    tool_call_details={
                                                        "summary": get_tool_result_info_content(
                                                            tool_call["tool_name"],
                                                            tool_result.get(
                                                                "content", ""
                                                            ),
                                                        )
                                                    },
                                                )
                                            ],
                                            citations=[],
                                        )
                                        # Yield the tool result immediately
                                        yield tool_result_response
                                        break

                # Process the node content for streaming (as before)
                content_response = await self._process_node_for_streaming(
                    node_name, node_result, 0
                )

                # Yield the content response if it exists
                if content_response:
                    yield content_response

            logger.info("Pydantic tool graph agent stream completed successfully")

        except Exception as e:
            logger.error(
                f"Error in pydantic tool graph agent stream method: {str(e)}",
                exc_info=True,
            )
            # Yield error message to avoid silent failures
            yield ChatAgentResponse(
                response=f"Error during execution: {str(e)}",
                tool_calls=[],
                citations=[],
            )
            raise Exception from e
