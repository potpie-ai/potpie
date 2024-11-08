import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task

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


class CodeGenerationAgent:
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


    async def create_agents(self):
        code_generator = Agent(
            role="Code Generation Agent",
            goal="Generate precise, copy-paste ready code modifications that maintain project consistency",
            backstory=f"""
                You are an expert code generation agent specialized in creating production-ready, 
                immediately usable code modifications. Your primary responsibilities include:
                1. Analyzing existing codebase context and understanding dependencies
                2. Planning code changes that maintain exact project patterns and style
                3. Implementing changes with copy-paste ready output
                4. Following existing code conventions exactly as shown in the input files
                5. Never modifying string literals, escape characters, or formatting unless specifically requested

                Key principles:
                - Provide required new imports in a separate code block
                - Output only the specific functions/classes being modified
                - Never change existing string formats or escape characters
                - Maintain exact indentation and spacing patterns from original code
                - Include clear section markers for where code should be inserted/modified

                Maintain efficiency within {self.max_iter} iterations while ensuring production-ready output.
            """,
            tools=[
                self.get_code_from_node_id,
                self.get_code_from_multiple_node_ids,
                self.get_node_neighbours,
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
            max_iter=self.max_iter,
        )

        return code_generator

    async def create_tasks(
        self,
        query: str,
        project_id: str,
        node_ids: List[NodeContext],
        code_results: List[Dict[str, Any]],
        code_generator,
    ):
        code_generation_task = Task(
            description=f"""
            Work within {self.max_iter} iterations to generate copy-paste ready code based on:
            - Query: {query}
            - Project ID: {project_id}
            - Target Node IDs: {[node.model_dump() for node in node_ids]}
            - Existing Code Context: {code_results}

            Follow this structured approach:

            1. Context Analysis:
               - Review existing code style precisely
               - Note exact indentation patterns
               - Identify string literal formats
               - Review import organization patterns

            2. Implementation Planning:
               - Plan changes that maintain exact formatting
               - Never modify existing patterns unless requested
               - Identify required new imports

            3. Code Generation Format:
               Required imports (if any):
               ```python
               from new.module import NewClass
               ```

               Modified/New Code:
               ```python
               def modified_function():
                   # Your code here
                   pass
               ```

               Location Guide:
               ```
               File: path/to/file.py
               Replace function/class: name_of_function
               - or -
               Insert after line: X
               ```

            Important Output Rules:
            1. Always separate new imports from main code changes
            2. Never modify existing string literals or escape characters
            3. Follow existing indentation exactly
            4. Include only the specific functions/classes being modified
            5. Provide clear location markers for each code block
            6. Maintain exact spacing patterns from original code
            7. For new files, provide complete file content with imports at top
            8. Format code blocks with language specification for syntax highlighting

            Your response format should be:

            Brief explanation of changes (2-3 lines max)

            Required Imports:
            [code block with imports]

            Code Changes:
            [code block with changes]

            Location:
            [location details]
            
            Keep responses focused and formatted for easy copy-paste into an IDE.
            No conversation or unnecessary explanation - just clear, actionable code blocks.
            """,
            expected_output="Copy-paste ready code changes with clear location markers and required imports",
            agent=code_generator,
        )

        return code_generation_task


    async def run(
        self,
        query: str,
        project_id: str,
        node_ids: List[NodeContext],
    ) -> str:
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        code_results = []
        if len(node_ids) > 0:
            code_results = await GetCodeFromMultipleNodeIdsTool(
                self.sql_db, self.user_id
            ).run_multiple(project_id, [node.node_id for node in node_ids])

        code_generator = await self.create_agents()
        generation_task = await self.create_tasks(
            query,
            project_id,
            node_ids,
            code_results,
            code_generator,
        )

        crew = Crew(
            agents=[code_generator],
            tasks=[generation_task],
            process=Process.sequential,
            verbose=False,
        )

        result = await crew.kickoff_async()
        return result


async def kickoff_code_generation_crew(
    query: str,
    project_id: str,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> str:
    code_gen_agent = CodeGenerationAgent(sql_db, llm, mini_llm, user_id)
    result = await code_gen_agent.run(query, project_id, node_ids)
    return result