import logging
import asyncio
from typing import Dict, Optional, List
from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.modules.intelligence.tools.change_detection.patch_extraction_tool import get_patch_extraction_tool
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService


class ProcessLargePRInput(BaseModel):
    project_id: str = Field(
        ..., description="The ID of the project being evaluated, this is a UUID."
    )
    base_branch: Optional[str] = Field(
        None, description="The base branch to compare against, this is a branch name. If none is provided, the default branch will be used."
    )


class ProcessLargePRTool:
    name = "Process large PR"
    description = """Processes a large PR by extracting and displaying patches for all changed files.
        :param project_id: string, the ID of the project being evaluated (UUID).
        :param base_branch: string, (optional) the base branch to compare against, this is a branch name. If none is provided, the default branch will be used.

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns a summary of all changed files and their patches.
        """

    def __init__(self, user_id: str, db_session=None):
        self.user_id = user_id
        self.db_session = db_session or next(get_db())
        self.patch_extraction_tool = get_patch_extraction_tool(user_id)
        self.provider_service = ProviderService.create(self.db_session, user_id)
        self.code_service = CodeProviderService(self.db_session)
        self.project_service = ProjectService(self.db_session)

    async def _get_file_content(self, project_name: str, file_path: str, branch_name: str, project_id: str) -> str:
        try:
            return self.code_service.get_file_content(
                project_name,
                file_path,
                0,
                0,
                branch_name,
                project_id,
            )
        except Exception as e:
            logging.error(f"Error getting content for {file_path}: {str(e)}")
            return ""

    async def _analyze_file_changes(self, file_path: str, file_content: str, diff: str) -> str:
        prompt = f"""You are an expert software analyst tasked with reviewing changes in a pull request (PR) for a software project. Your analysis will be used as input for another language model to generate a comprehensive test plan. Your goal is to extract and summarize key business logic changes that would require test updates or new tests.

Here is the content of the file being analyzed:

<file_content>
{file_content}
</file_content>

And here are the changes made to the file (diff):

<diff>
{diff}
</diff>

Please analyze the file content and its diff to identify and summarize important functional changes. Focus on changes that would impact testing requirements.

Instructions:
1. Carefully review the file content and diff.
2. Identify the key changes in business logic, functionality, or behavior.
3. Determine the type of change based on the following categories:
   - API change (parameters, return types, endpoints)
   - Data model change (schema, validation, relationships)
   - Business logic update (algorithms, rules, calculations)
   - UI change (user interface, interactions)
   - Error handling change (exceptions, fallbacks)
   - Performance improvement (optimizations)
   - Security change (authentication, authorization, validation)
   - Other (if none of the above apply)
4. Identify the specific endpoint, function, or component affected by the change.
5. Describe the new behavior or functionality resulting from the change.
6. Identify potential test implications of this change.

Before providing your final output, wrap your analysis in <change_analysis> tags. In this analysis:
1. List out each specific change found in the diff, numbering them.
2. Categorize each change according to the provided categories.
3. Consider the implications of each change for testing.
4. Identify functions that are called by the changed code (downstream dependencies).
5. Identify modified inputs/parameters and outputs/return values.
It's OK for this section to be quite long.

After your analysis, provide your final output in the following format:

File: [filename]
Change Type: [one of the categories listed above]
Summary: [One concise sentence describing what the change does]
Endpoint/Function: [Specific endpoint, function, or component affected]
New Behavior: [Concise description of the new behavior or functionality]
Test Impact: [High/Medium/Low]
Downstream Dependencies: [Functions called by this code - visible in the file]
Modified Inputs: [List of changed input parameters]
Modified Outputs: [List of changed return values/responses]
Context Needed: [List specific functions, classes, or files the second step should fetch]
Suggested Test Scenarios: [Brief list of key scenarios to test, including happy path and edge cases]"""

        try:
            response = await self.provider_service.call_llm(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                config_type="inference"
            )
            return response
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {str(e)}")
            return f"Error analyzing file {file_path}: {str(e)}"

    async def _process_file(self, file_path: str, patch: str, project_name: str, branch_name: str, project_id: str) -> str:
        try:
            # Get file content and analyze changes in parallel
            file_content = await self._get_file_content(project_name, file_path, branch_name, project_id)
            if not file_content:
                return f"Error: Could not get content for file {file_path}"
            
            return await self._analyze_file_changes(file_path, file_content, patch)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return f"Error processing file {file_path}: {str(e)}"

    async def process_large_pr(self, project_id: str, base_branch: Optional[str] = None) -> str:
        try:
            # Get patches using the patch extraction tool

            patches = await self.patch_extraction_tool.arun({"project_id": project_id, "base_branch": base_branch})
            
            if not patches:
                return "No changes found in the PR."
            
            # Get project details
            project = await ProjectService(self.db_session).get_project_from_db_by_id(project_id)
            
            # Create tasks for processing each file in parallel
            process_tasks = []
            if patches.patches:
                for file_path, patch in patches.patches.items():
                    process_tasks.append(
                        self._process_file(
                            file_path,
                        patch,
                        project["project_name"],
                        base_branch,
                        project_id
                    )
                )
            
            # Wait for all analyses to complete
            analyses = await asyncio.gather(*process_tasks)
            
            # Build final summary
            summary = "PR Analysis Summary:\n\n"
            for analysis in analyses:
                summary += analysis + "\n" + "-" * 80 + "\n"
            
            return summary
            
        except Exception as e:
            logging.error(f"Error processing large PR: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing large PR: {str(e)}"
            )

    async def arun(self, project_id: str, base_branch: Optional[str] = None) -> str:
        return await self.process_large_pr(project_id, base_branch)

    def run(self, project_id: str, base_branch: Optional[str] = None) -> str:
        return asyncio.run(self.process_large_pr(project_id, base_branch))


def get_process_large_pr_tool(user_id: str, db_session) -> StructuredTool:
    """
    Get a LangChain Tool object for processing large PRs.
    """
    process_large_pr_tool = ProcessLargePRTool(user_id, db_session)
    return StructuredTool.from_function(
        coroutine=process_large_pr_tool.arun,
        func=process_large_pr_tool.run,
        name="Process large PR",
        description="""
            Process a large PR by extracting and displaying patches for all changed files.
            This tool helps review large PRs by showing all file changes in a structured format.
            Inputs:
            - project_id (str): The ID of the project being evaluated, this is a UUID.
            - base_branch (str): (optional) The base branch to compare against, this is a branch name. If none is provided, the default branch will be used.
            The output includes a summary of all changed files and their patches.
            """,
        args_schema=ProcessLargePRInput,
    ) 