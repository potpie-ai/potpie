import asyncio
import logging
import os
import time
from typing import List

import aiofiles
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from sqlalchemy.orm import Session

from app.core.celery_worker.celery_worker import celery_worker_instance
from app.core.database import get_db
from app.modules.parsing.graph_construction.parsing_helper import ParseHelper
from app.modules.parsing.graph_construction.parsing_schema import RepoDetails
from app.modules.parsing.knowledge_graph.kg_models import FileAnalysis
from app.modules.projects.projects_service import ProjectService
from app.modules.tasks.task_model import TaskType
from app.modules.tasks.task_service import TaskService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodebaseInferenceService:
    def __init__(self, db: Session):
        self.db = db
        self.task_service = TaskService(db)
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.celery_app = celery_worker_instance.celery_instance

    async def process_repository(self, repo_name, user_id: str, project_id):
        db = next(get_db())
        service = CodebaseInferenceService(db)
        try:
            logger.info(
                f"Processing repository: {repo_name} for user: {user_id} and project: {project_id}"
            )
            return await service._process_repository(repo_name, user_id, project_id)
        finally:
            db.close()

    async def _process_repository(self, repo_name, user_id: str, project_id):
        codebase_task = self.task_service.create_task(
            TaskType.CODEBASE_PROCESSING, "submitted", project_id
        )
        logger.info(
            f"Created task for codebase processing: {codebase_task.id} for project: {project_id}"
        )

        try:
            repo, extracted_dir = await self.clone_repository(repo_name, user_id)
            self.task_service.update_task(codebase_task.id, custom_status="cloned")
            logger.info(
                f"Cloned repository: {repo_name} to {extracted_dir} for project: {project_id}"
            )
            self.task_service.update_task(codebase_task.id, custom_status="processing")
            logger.info(
                f"Started processing files in {extracted_dir} for project: {project_id}"
            )

            await self.process_files_in_parallel_batches(
                project_id, extracted_dir, codebase_task.id
            )

            return extracted_dir
        except Exception as e:
            logger.error(
                f"Error processing repository: {str(e)} for project: {project_id}"
            )
            self.task_service.update_task(codebase_task.id, custom_status="errored")
            raise e

    async def clone_repository(self, repo_details: RepoDetails, user_id: str):
        repo, owner, auth = await self.parse_helper.clone_or_copy_repository(
            repo_details, self.db, user_id
        )
        extracted_dir = await self.parse_helper.download_and_extract_tarball(
            repo, repo_details.branch_name, "/tmp", auth, repo, user_id
        )
        logger.info(
            f"Cloned repository {repo_details} with owner {owner} to {extracted_dir}"
        )
        return repo, extracted_dir

    async def process_files_in_parallel_batches(
        self,
        project_id: str,
        directory: str,
        codebase_task_id: int,
        batch_size: int = 2,
        num_parallel_tasks: int = 5,
    ):
        files = self.get_text_files(directory)
        tasks = []
        for i, file_path in enumerate(files):
            task = self.task_service.create_task(
                TaskType.FILE_INFERENCE, "submitted", project_id
            )
            tasks.append((file_path, task.id))
            logger.info(
                f"Created task for file inference: {file_path} for project task: {codebase_task_id}"
            )

            if len(tasks) == batch_size * num_parallel_tasks or i == len(files) - 1:
                batch_tasks = [
                    tasks[j : j + batch_size] for j in range(0, len(tasks), batch_size)
                ]
                await asyncio.gather(
                    *(process_file_batch(batch) for batch in batch_tasks)
                )
                logger.info(
                    f"Processing {len(tasks)} files in {len(batch_tasks)} batches for project task: {codebase_task_id}"
                )
                tasks = []

    def get_text_files(self, directory: str) -> List[str]:
        text_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if self.is_text_file(file):
                    text_files.append(os.path.join(root, file))
        logger.info(f"Found {len(text_files)} text files in directory: {directory}")
        return text_files

    def is_text_file(self, filename: str) -> bool:
        extensions = [
            ".cs",
            ".c",
            ".cpp",
            ".cxx",
            ".cc",
            ".el",
            ".ex",
            ".exs",
            ".elm",
            ".go",
            ".java",
            ".js",
            ".jsx",
            ".ml",
            ".mli",
            ".php",
            ".py",
            ".ql",
            ".rb",
            ".rs",
            ".ts",
            ".tsx",
            ".txt",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".md",
        ]
        if "." not in filename:
            return False
        is_text = any(
            filename.lower().endswith(ext) for ext in extensions
        ) or self.is_text_content(filename)
        logger.debug(f"File {filename} is text file: {is_text}")
        return is_text

    def is_text_content(self, file_path: str) -> bool:
        if not os.path.isfile(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False

        try:
            with open(file_path, "rb") as file:
                is_text = not bool(file.read(1024).translate(None, bytes([0])))
                logger.debug(f"File {file_path} is text content: {is_text}")
                return is_text
        except OSError as e:
            logger.error(
                f"Error checking if file is text content: {file_path}, error: {str(e)}"
            )
            return False

    async def process_single_file(self, file_path: str, task_id: int):
        try:
            if not os.path.isfile(file_path):
                if os.path.isdir(file_path):
                    return  # Pass if it's a directory
                logger.warning(f"File does not exist: {file_path}")
                return

            logger.info(f"Processing single file: {file_path} for task: {task_id}")
            async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
                content = await file.read()

            output_parser = PydanticOutputParser(pydantic_object=FileAnalysis)

            prompt = ChatPromptTemplate.from_template(
                template="""You are an expert code analyzer with deep understanding of software engineering principles, design patterns, and best practices. Your task is to analyze the given code and provide a detailed breakdown of its structure, functionality, and relationships. Please follow these instructions carefully:

                1. Analyze the provided code snippet or file content.
                2. Provide your analysis in a JSON format that matches the structure defined below.
                3. Be as detailed and accurate as possible in your explanations.
                4. If certain information is not applicable or cannot be determined, use null or an empty list/object as appropriate.
                \n{format_instructions}\n
                ## Guidelines for Analysis
                1. File Purpose: Provide a concise yet comprehensive description of the file's role in the project.
                2. APIs: Identify any API endpoints defined in the file, including their HTTP methods and routes.
                3. Imports: List all project-specific files and external libraries imported.
                4. Classes: For each class, explain its purpose, list its methods and attributes, and identify any inheritance.
                5. Functions: For each function (including methods), explain its purpose in a consise yet comprehensive manner. Retaining some technical detail, parameters, return value, and any notable implementation details.
                6. Complexity: Estimate the time complexity of functions where applicable (e.g., O(n), O(log n)).
                7. Dependencies: Identify any files or modules that this code depends on.
                8. Relationships: Note any important relationships between functions, classes, or modules.

                Now, analyze the following code and provide a detailed breakdown:

                {content}

                Respond in the specified JSON format.
                """,
                partial_variables={
                    "format_instructions": output_parser.get_format_instructions()
                },
            )

            chain = prompt | self.llm | output_parser
            print(f"Processing file: {file_path} at {time.time()}")
            start_time = time.time()
            result = await chain.ainvoke({"content": content})
            end_time = time.time()
            print(
                f"Time taken to process file: {file_path} with task: {task_id} is {end_time - start_time} seconds"
            )
            self.task_service.update_task(
                task_id, custom_status="ready", result=str(result)
            )
            logger.info(
                f"Successfully processed file: {file_path} and updated task: {task_id} for project: {task_id}"
            )
        except Exception as e:
            error_message = f"Error processing file: {file_path}, error: {str(e)}"
            self.task_service.update_task(
                task_id, custom_status="errored", result=error_message
            )
            logger.error(error_message)


async def process_file_batch(tasks: List[tuple]):
    logger.info(f"Processing file batch for: {tasks}")
    db = next(get_db())
    inference_service = CodebaseInferenceService(db)
    for task in tasks:
        try:
            logger.info(f"Processing file batch for: {task[0]} with task ID: {task[1]}")
            await inference_service.process_single_file(task[0], task[1])
        except Exception as e:
            error_message = f"Error processing file: {task[0]} with task ID: {task[1]} error: {str(e)}"
            inference_service.task_service.update_task(
                task[1], custom_status="errored", result=error_message
            )
            logger.error(error_message)
    db.close()
