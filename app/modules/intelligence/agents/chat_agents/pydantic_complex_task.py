import re
from typing import List, AsyncGenerator, Dict, Any, Optional, Callable, Set, Tuple
import asyncio

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


class GraphNode:
    """A simple node in our execution graph"""

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

        # Execute function
        result = await self.func(**inputs)

        # Update state with prefixed keys
        for key, value in result.items():
            self.outputs.add(key)
            state[f"{self.name}.{key}"] = value
            state[key] = value  # Also store direct key for convenience

        return result


class CustomGraph:
    """Simple execution graph for agent workflow"""

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

    def get_execution_path(self, start_node: Optional[str] = None) -> List[str]:
        """Get the execution path from a starting node"""
        if not self.execution_order:
            self.finalize()

        if not start_node:
            return self.execution_order

        # Get subset of execution order starting from start_node
        start_idx = self.execution_order.index(start_node)
        return self.execution_order[start_idx:]

    async def execute(
        self, initial_state: Dict[str, Any], custom_path: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute the graph with the given initial state"""
        if not self.execution_order and not custom_path:
            self.finalize()

        path = custom_path if custom_path else self.execution_order
        state = initial_state.copy()

        results = {}
        node_results = {}

        for node_name in path:
            node = self.nodes[node_name]
            node_result = await node.execute(state)
            node_results[node_name] = node_result

            # Update "direct" results dictionary with latest outputs
            for key, value in node_result.items():
                results[key] = value

        return {"state": state, "results": results, "node_results": node_results}

    async def execute_with_loop(
        self,
        initial_state: Dict[str, Any],
        loop_condition: Callable[[Dict[str, Any], str], bool],
        loop_from: str,
        loop_to: str,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Execute the graph with conditional looping between nodes"""
        if not self.execution_order:
            self.finalize()

        state = initial_state.copy()
        results = {}
        node_results = {}
        iteration = 0

        # Execute up to loop_from first
        start_idx = 0
        to_idx = self.execution_order.index(loop_from)

        for node_name in self.execution_order[start_idx:to_idx]:
            node = self.nodes[node_name]
            node_result = await node.execute(state)
            node_results[node_name] = node_result

        # Now handle the loop section
        while iteration < max_iterations:
            iteration += 1

            # Execute from loop_from to end
            loop_from_idx = self.execution_order.index(loop_from)

            for node_name in self.execution_order[loop_from_idx:]:
                node = self.nodes[node_name]
                node_result = await node.execute(state)
                node_results[node_name] = node_result

                # Update results
                for key, value in node_result.items():
                    results[key] = value

                # Check loop condition after the loop_to node executes
                if node_name == loop_to and loop_condition(state, node_name):
                    # Condition true, continue looping
                    break

                # If we've reached the end without breaking, stop looping
                if node_name == self.execution_order[-1]:
                    iteration = max_iterations  # Force exit

            # If we've hit max iterations, add that information to results
            if iteration >= max_iterations:
                results["max_iterations_reached"] = True
                break

        return {
            "state": state,
            "results": results,
            "node_results": node_results,
            "iterations": iteration,
        }

    async def stream_execute(
        self, initial_state: Dict[str, Any], custom_path: Optional[List[str]] = None
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Stream execution, yielding each node's name and results as they complete"""
        if not self.execution_order and not custom_path:
            self.finalize()

        path = custom_path if custom_path else self.execution_order
        state = initial_state.copy()

        for node_name in path:
            node = self.nodes[node_name]
            node_result = await node.execute(state)
            yield node_name, node_result

    async def stream_execute_with_loop(
        self,
        initial_state: Dict[str, Any],
        loop_condition: Callable[[Dict[str, Any], str], bool],
        loop_from: str,
        loop_to: str,
        max_iterations: int = 5,
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any], int], None]:
        """Stream execution with looping, yielding node name, results, and iteration count"""
        if not self.execution_order:
            self.finalize()

        state = initial_state.copy()
        iteration = 0
        has_error = False
        skip_to_output = False  # Add this flag to skip directly to output if needed

        try:
            # Execute up to loop_from first
            start_idx = 0
            to_idx = self.execution_order.index(loop_from)

            for node_name in self.execution_order[start_idx:to_idx]:
                node = self.nodes[node_name]
                try:
                    node_result = await node.execute(state)
                    yield node_name, node_result, 0
                except Exception as e:
                    error_result = {
                        "error": str(e),
                        "node": node_name,
                        "success": False,
                    }
                    has_error = True
                    yield node_name, error_result, 0

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
                        yield node_name, node_result, iteration

                        # IMPORTANT: Check verification result - if verified is True,
                        # we need to skip to output directly
                        if node_name == "verify" and node_result.get("verified", False):
                            skip_to_output = True
                            # Skip to output node directly since verification passed
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
                        yield node_name, error_result, iteration
                        skip_to_output = True  # On error, skip to output
                        break

                # Stop if we've hit max iterations
                if iteration >= max_iterations:
                    break

            # ALWAYS run the output node at the end
            output_idx = (
                self.execution_order.index("output")
                if "output" in self.execution_order
                else -1
            )
            if output_idx >= 0:
                output_node = self.nodes["output"]
                try:
                    output_result = await output_node.execute(state)
                    yield "output", output_result, iteration
                except Exception as e:
                    yield "output", {"error": str(e)}, iteration

        except Exception as e:
            error_result = {"error": str(e), "success": False}
            yield "error", error_result, 0


class CustomGraphAgent:
    """Agent that uses a custom graph for execution flow"""

    def __init__(
        self, llm: Any, graph: CustomGraph, tools: List[Tool], system_prompt: str
    ):
        self.llm = llm
        self.graph = graph
        self.tools = tools
        self.system_prompt = system_prompt

        # Create a regular pydantic_ai agent for tool usage
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

    async def run(self, **initial_state) -> Dict[str, Any]:
        """Run the graph execution end-to-end"""

        # Define loop condition - should return True to continue looping
        def loop_condition(state: Dict[str, Any], node_name: str) -> bool:
            return (
                state.get("continue_loop", False)
                and state.get("attempt_count", 0) < MAX_FIX_ATTEMPTS
            )

        # Run graph with looping from fix back to verify if needed
        result = await self.graph.execute_with_loop(
            initial_state=initial_state,
            loop_condition=loop_condition,
            loop_from="verify",
            loop_to="fix",
            max_iterations=MAX_FIX_ATTEMPTS,
        )

        # Return the final results
        return result["results"]

    async def run_stream(
        self, **initial_state
    ) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
        """Stream the graph execution"""

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
            yield node_name, node_result


class PydanticGraphAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        config: AgentConfig,
        tools: List[StructuredTool],
    ):
        """
        Initialize a graph agent with planning, implementation, verification and fix steps.

        This agent uses a custom graph to:
        1. Plan a strategy to solve the user's query
        2. Implement a solution
        3. Verify the solution
        4. Fix any issues (with looping back to verify again if needed)

        It supports both standard and streaming output.
        """
        self.config = config
        self.tasks = config.tasks
        self.tools = tools
        self.llm_provider = llm_provider

        # Build the graph
        self.graph = self._build_graph()

        # Create the GraphAgent
        self.agent = CustomGraphAgent(
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

    async def _plan_step(
        self, task: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a plan for solving the task"""
        if not context:
            context = {}

        async with self.agent.agent.iter(
            user_prompt=f"PLANNING PHASE: {task}"
        ) as stream:
            plan_text = ""
            async for node in stream:
                if self.agent.agent.is_model_request_node(node):
                    # Only capture model output
                    async with node.stream(stream.ctx) as req_stream:
                        async for evt in req_stream:
                            if (
                                isinstance(evt, PartStartEvent)
                                and hasattr(evt, "part")
                                and hasattr(evt.part, "content")
                            ):
                                plan_text += evt.part.content
                            elif (
                                isinstance(evt, PartDeltaEvent)
                                and hasattr(evt, "delta")
                                and hasattr(evt.delta, "content_delta")
                            ):
                                plan_text += evt.delta.content_delta

        return {"plan": plan_text, "task": task, "context": context, "attempt_count": 0}

    async def _implement_step(
        self, plan: str, task: str, context: Dict[str, Any], attempt_count: int
    ) -> Dict[str, Any]:
        """Implement the solution based on the plan"""
        async with self.agent.agent.iter(
            user_prompt=f"IMPLEMENTATION PHASE: Implement a solution for: {task}\n\nFollowing this plan: {plan}"
        ) as stream:
            solution_text = ""
            async for node in stream:
                if self.agent.agent.is_model_request_node(node):
                    # Only capture model output
                    async with node.stream(stream.ctx) as req_stream:
                        async for evt in req_stream:
                            if (
                                isinstance(evt, PartStartEvent)
                                and hasattr(evt, "part")
                                and hasattr(evt.part, "content")
                            ):
                                solution_text += evt.part.content
                            elif (
                                isinstance(evt, PartDeltaEvent)
                                and hasattr(evt, "delta")
                                and hasattr(evt.delta, "content_delta")
                            ):
                                solution_text += evt.delta.content_delta

        return {
            "solution": solution_text,
            "plan": plan,
            "task": task,
            "context": context,
            "attempt_count": attempt_count,
        }

    async def _verify_step(
        self,
        solution: str,
        plan: str,
        task: str,
        context: Dict[str, Any],
        attempt_count: int,
    ) -> Dict[str, Any]:
        """Verify the solution for errors or issues"""
        verification_prompt = f"""
        VERIFICATION PHASE: Carefully review this solution and identify any issues or errors.
        
        TASK: {task}
        
        PLAN: {plan}
        
        SOLUTION TO VERIFY:
        {solution}
        
        Check for:
        1. Completeness - Does it fully address the task?
        2. Correctness - Are there any logical errors or bugs?
        3. Style - Is it well-formatted and clear?
        
        First, state whether the solution is VERIFIED (no issues) or NEEDS FIXING.
        If NEEDS FIXING, provide a detailed explanation of each issue.
        """

        result = await self.agent.agent.run(user_prompt=verification_prompt)
        verification_text = result.data

        # Determine if verified by checking for key phrases
        needs_fixing = any(
            phrase in verification_text.lower()
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
            "issue_report": verification_text,
            "verified": not needs_fixing,
            "attempt_count": attempt_count,
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
    ) -> Dict[str, Any]:
        """Fix issues in the solution if verification failed"""
        # Short-circuit if already verified or max attempts reached
        if verified:
            return {
                "fixed_solution": solution,
                "continue_loop": False,
                "attempt_count": attempt_count,
                "final": True,
            }

        if attempt_count >= MAX_FIX_ATTEMPTS:
            return {
                "fixed_solution": f"After {MAX_FIX_ATTEMPTS} attempts, here's the best solution:\n\n{solution}",
                "continue_loop": False,
                "attempt_count": attempt_count,
                "final": True,
            }

        # Increment attempt count
        new_attempt_count = attempt_count + 1

        fix_prompt = f"""
        FIX PHASE (Attempt {new_attempt_count}/{MAX_FIX_ATTEMPTS}): Fix the following solution based on the issues identified.
        
        TASK: {task}
        
        ORIGINAL PLAN: {plan}
        
        CURRENT SOLUTION:
        {solution}
        
        ISSUES IDENTIFIED:
        {issue_report}
        
        Please provide a completely fixed version that addresses ALL identified issues.
        Don't just describe the fixes - implement them and provide the complete corrected solution.
        """

        result = await self.agent.agent.run(user_prompt=fix_prompt)
        fixed_solution = result.data

        return {
            "fixed_solution": fixed_solution,
            "continue_loop": True,  # Signal to continue the loop
            "attempt_count": new_attempt_count,
            "final": False,
        }

    async def _output_step(
        self,
        solution: Optional[str] = None,
        fixed_solution: Optional[str] = None,
        final: bool = True,
        context: Any = None,
    ) -> Dict[str, Any]:
        """Final output node that selects the best solution"""
        # Choose the best solution
        final_output = fixed_solution if fixed_solution else solution
        if not final_output:
            final_output = "Unable to generate a proper solution."

        return {"output": final_output}

    def _build_graph(self) -> CustomGraph:
        """Build the execution graph with planning, implementation, verification and fix phases"""
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

        # Connect nodes - standard path
        graph.connect("plan", "plan", "implement", "plan")
        graph.connect("plan", "task", "implement", "task")
        graph.connect("plan", "context", "implement", "context")
        graph.connect("plan", "attempt_count", "implement", "attempt_count")

        graph.connect("implement", "solution", "verify", "solution")
        graph.connect("implement", "plan", "verify", "plan")
        graph.connect("implement", "task", "verify", "task")
        graph.connect("implement", "context", "verify", "context")
        graph.connect("implement", "attempt_count", "verify", "attempt_count")

        # Connect verify to either fix or output based on verification result
        graph.connect("verify", "solution", "fix", "solution")
        graph.connect("verify", "issue_report", "fix", "issue_report")
        graph.connect("verify", "plan", "fix", "plan")
        graph.connect("verify", "task", "fix", "task")
        graph.connect("verify", "context", "fix", "context")
        graph.connect("verify", "attempt_count", "fix", "attempt_count")
        graph.connect("verify", "verified", "fix", "verified")

        # Connect fix to output
        graph.connect("fix", "fixed_solution", "output", "fixed_solution")
        graph.connect("fix", "final", "output", "final")

        # IMPORTANT: Direct connection from verify to output
        graph.connect("verify", "solution", "output", "solution")
        graph.connect(
            "verify", "verified", "output", "final"
        )  # Pass verified status to output

        # Finalize the graph
        graph.finalize()

        return graph

    def _create_task_input(
        self, task_config: TaskConfig, ctx: ChatContext
    ) -> Dict[str, Any]:
        """Create the task input for the graph agent"""
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

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run the graph agent with planning, implementation, verification and fix steps"""
        logger.info("Running custom graph agent")
        try:
            # Create input for the graph agent
            task_input = self._create_task_input(self.tasks[0], ctx)

            # Run the graph agent
            result = await self.agent.run(**task_input)

            # Extract the final output
            final_output = result.get("output", "Unable to generate response")

            return ChatAgentResponse(
                response=final_output,
                tool_calls=[],
                citations=[],
            )
        except Exception as e:
            logger.error(f"Error in graph agent run method: {str(e)}", exc_info=True)
            raise Exception from e

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Stream the graph agent execution, showing each step in real-time"""
        logger.info("Running custom graph agent stream")

        # Create input for the graph agent
        task_input = self._create_task_input(self.tasks[0], ctx)

        try:
            # Stream execution
            async for node_name, node_result in self.agent.run_stream(**task_input):
                # Node-specific responses
                if node_name == "plan":
                    yield ChatAgentResponse(
                        # response=f"📝 Planning approach...\n\n",
                        response=f"📝 Planning approach...\n\n{node_result.get('plan', '')}",
                        tool_calls=[],
                        citations=[],
                    )
                elif node_name == "implement":
                    yield ChatAgentResponse(
                        # response=f"⚙️ Implementing solution...\n\n...",
                        response=f"⚙️ Implementing solution...\n\n{node_result.get('solution', '')[:100]}...",
                        tool_calls=[],
                        citations=[],
                    )
                elif node_name == "verify":
                    if node_result.get("verified", False):
                        yield ChatAgentResponse(
                            # response=f"✅ Verification passed!\n\n",
                            response=f"✅ Verification passed!\n\n{node_result.get('issue_report', '')}",
                            tool_calls=[],
                            citations=[],
                        )
                    else:
                        yield ChatAgentResponse(
                            # response=f"❌ Verification found issues:\n\n",
                            response=f"❌ Verification found issues:\n\n{node_result.get('issue_report', '')}",
                            tool_calls=[],
                            citations=[],
                        )
                elif node_name == "fix":
                    attempt = node_result.get("attempt_count", 0)
                    if node_result.get("continue_loop", False):
                        yield ChatAgentResponse(
                            response=f"🔄 Fix attempt {attempt}/{MAX_FIX_ATTEMPTS}: Making improvements and reverifying...",
                            tool_calls=[],
                            citations=[],
                        )
                    else:
                        yield ChatAgentResponse(
                            response=f"🏁 Fix complete after {attempt} attempts.",
                            tool_calls=[],
                            citations=[],
                        )
                elif node_name == "output":
                    # Final output
                    yield ChatAgentResponse(
                        response=node_result.get(
                            "output",
                            f"Final response unavailable {node_result.get('error')}",
                        ),
                        tool_calls=[],
                        citations=[],
                    )

            logger.info("Graph agent stream completed successfully")

        except Exception as e:
            logger.error(f"Error in graph agent stream method: {str(e)}", exc_info=True)
            # Yield error message to avoid silent failures
            yield ChatAgentResponse(
                response=f"Error during execution: {str(e)}",
                tool_calls=[],
                citations=[],
            )
            raise Exception from e
