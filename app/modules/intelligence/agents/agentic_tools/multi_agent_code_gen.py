import os
from typing import Any, Dict, List
from dataclasses import dataclass

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel, Field

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    get_node_neighbours_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    GetCodeFromMultipleNodeIdsTool,
    get_code_from_multiple_node_ids_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    get_code_from_node_id_tool,
)


class CodeGenerationPlan(BaseModel):
    affected_files: List[str] = Field(..., description="Files that need modification")
    changes_required: Dict[str, str] = Field(..., description="Required changes per file")
    dependencies: List[str] = Field(..., description="Related dependencies to consider")
    implementation_steps: List[str] = Field(..., description="Ordered steps for implementation")


class CodeModification(BaseModel):
    file_path: str
    original_code: str
    updated_code: str
    changes_description: str


class CodeGenerationResult(BaseModel):
    plan: CodeGenerationPlan
    modifications: List[CodeModification]
    validation_notes: List[str]


@dataclass
class AgentContext:
    query: str
    project_id: str
    node_ids: List[NodeContext]
    code_results: List[Dict[str, Any]]
    max_iter: int


class MultiAgentCodeGenerator:
    def __init__(self, sql_db, llm, mini_llm, user_id):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_iter = os.getenv("MAX_ITER", 5)
        self.sql_db = sql_db
        self.get_code_from_node_id = get_code_from_node_id_tool(sql_db, user_id)
        self.get_code_from_multiple_node_ids = get_code_from_multiple_node_ids_tool(
            sql_db, user_id
        )
        self.get_node_neighbours = get_node_neighbours_from_node_id_tool(sql_db)
        self.llm = llm
        self.mini_llm = mini_llm
        self.user_id = user_id

    def create_analyzer_agent(self) -> Agent:
        return Agent(
            role="Code Context Analyzer",
            goal="Analyze existing codebase and create comprehensive context for modifications",
            backstory="""
                You are an expert code analyst specialized in understanding codebases and their structures.
                You excel at:
                1. Analyzing code patterns and architecture
                2. Identifying dependencies and relationships
                3. Understanding the scope and impact of changes
                4. Creating detailed context for other agents
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_multiple_node_ids,
                self.get_node_neighbours,
            ],
            allow_delegation=True,
            verbose=True,
            llm=self.llm,
        )

    def create_planner_agent(self) -> Agent:
        return Agent(
            role="Implementation Planner",
            goal="Create detailed implementation plan based on analysis",
            backstory="""
                You are a strategic code planner who excels at:
                1. Breaking down complex changes into manageable steps
                2. Identifying potential risks and dependencies
                3. Creating clear implementation roadmaps
                4. Ensuring minimal impact on existing functionality
            """,
            tools=[],  # Planner uses context from analyzer
            allow_delegation=True,
            verbose=True,
            llm=self.llm,
        )

    def create_implementer_agent(self) -> Agent:
        return Agent(
            role="Code Implementer",
            goal="Generate and modify code according to the implementation plan",
            backstory="""
                You are an expert code implementer who:
                1. Writes clean, efficient code
                2. Follows existing patterns and conventions
                3. Implements changes with precision
                4. Maintains code quality and consistency
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_multiple_node_ids,
            ],
            allow_delegation=True,
            verbose=True,
            llm=self.llm,
        )

    def create_validator_agent(self) -> Agent:
        return Agent(
            role="Code Validator",
            goal="Validate generated code for correctness and consistency",
            backstory="""
                You are a thorough code validator who:
                1. Reviews code changes for correctness
                2. Ensures consistency with existing codebase
                3. Identifies potential issues or edge cases
                4. Provides validation feedback
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_node_neighbours,
            ],
            allow_delegation=True,
            verbose=True,
            llm=self.llm,
        )

    def create_analysis_task(self, context: AgentContext, analyzer: Agent) -> Task:
        return Task(
            description=f"""
            Analyze the codebase context within {context.max_iter} iterations:
            - Query: {context.query}
            - Project ID: {context.project_id}
            - Target Nodes: {[node.model_dump() for node in context.node_ids]}
            - Existing Code: {context.code_results}

            Provide:
            1. Detailed analysis of affected code sections
            2. Identification of patterns and conventions
            3. List of dependencies and relationships
            4. Potential impact areas
            
            Return analysis in structured format focusing on technical details.
            """,
            agent=analyzer,
            expected_output="Return analysis in structured format focusing on technical details.",
        )

    def create_planning_task(self, context: AgentContext, planner: Agent) -> Task:
        return Task(
            description=f"""
            Create implementation plan based on analysis:
            - Break down the query: {context.query}
            - Consider provided analysis
            - Design step-by-step implementation approach

            Provide CodeGenerationPlan with:
            1. List of affected files
            2. Specific changes needed per file
            3. Dependencies to consider
            4. Ordered implementation steps
            
            Focus on technical accuracy and implementation details.
            """,
            agent=planner,
            expected_output=f"{CodeGenerationPlan.model_json_schema()}",
        )

    def create_implementation_task(self, context: AgentContext, implementer: Agent) -> Task:
        return Task(
            description=f"""
            Implement code changes following the provided plan:
            - Follow implementation steps precisely
            - Maintain consistent code style
            - Generate/modify code as needed
            
            For each file:
            1. Generate complete function/class if new
            2. Show only modified sections if updating
            3. Include necessary imports and dependencies
            4. Add/update docstrings as needed
            
            Return CodeModification objects for each change.
            Use markdown code blocks with language specification.
            """,
            agent=implementer,
            expected_output=f"{CodeModification.model_json_schema()}",
        )

    def create_validation_task(self, context: AgentContext, validator: Agent) -> Task:
        return Task(
            description=f"""
            Validate implemented code changes:
            - Review against original requirements
            - Check consistency with existing code
            - Verify proper handling of edge cases
            
            Provide validation notes including:
            1. Correctness of implementation
            2. Consistency with codebase
            3. Potential issues or concerns
            4. Suggestions for improvements
            
            Return list of validation notes.
            """,
            agent=validator,
            expected_output=f"Return list of validation notes.",
        )

    async def run(
        self,
        query: str,
        project_id: str,
        node_ids: List[NodeContext],
    ) -> CodeGenerationResult:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])

        context = AgentContext(
            query=query,
            project_id=project_id,
            node_ids=node_ids,
            code_results=code_results,
            max_iter=self.max_iter,
        )

        # Create agents
        analyzer = self.create_analyzer_agent()
        planner = self.create_planner_agent()
        implementer = self.create_implementer_agent()
        validator = self.create_validator_agent()

        # Create tasks
        analysis_task = self.create_analysis_task(context, analyzer)
        planning_task = self.create_planning_task(context, planner)
        implementation_task = self.create_implementation_task(context, implementer)
        validation_task = self.create_validation_task(context, validator)

        # Create and run crew
        crew = Crew(
            agents=[analyzer, planner, implementer, validator],
            tasks=[analysis_task, planning_task, implementation_task, validation_task],
            process=Process.sequential,
            verbose=False,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_multi_agent_code_generation(
    query: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> CodeGenerationResult:
    generator = MultiAgentCodeGenerator(sql_db, llm, mini_llm, user_id)
    result = await generator.run(query, project_id, node_ids)
    return result