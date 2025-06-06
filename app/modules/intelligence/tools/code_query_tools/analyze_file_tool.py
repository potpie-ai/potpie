import ast
import logging
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy.orm import Session
from langchain_core.tools import StructuredTool
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService
from app.core.config_provider import config_provider


class AnalyzeCodeToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    file_path: str = Field(..., description="Path to the Python file within the repo")
    include_methods: bool = Field(
        True, description="Whether to include class methods in the analysis"
    )
    include_private: bool = Field(
        False,
        description="Whether to include private functions/methods (starting with _)",
    )


class CodeElement(BaseModel):
    name: str
    type: str  # 'function', 'class', 'method'
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None  # For methods
    signature: str  # Function/method signature
    is_private: bool = False
    is_async: bool = False


class AnalyzeCodeTool:
    name: str = "analyze_code_structure"
    description: str = (
        """Analyze Python file structure to extract all functions, classes, and methods with their docstrings and line numbers.
        This tool is particularly useful for large files (6000+ lines) to understand the code structure without loading the entire file.
        
        Returns detailed information about:
        - All classes with their docstrings and line ranges
        - All functions (both standalone and class methods) with signatures and docstrings
        - Exact start and end line numbers for each code element
        - Function/method signatures for better understanding
        
        param project_id: string, the repository ID (UUID) to analyze
        param file_path: string, the path to the Python file in the repository
        param include_methods: bool, whether to include class methods (default: True)
        param include_private: bool, whether to include private functions/methods starting with _ (default: False)
        
        example:
        {
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "file_path": "src/main.py",
            "include_methods": true,
            "include_private": false
        }
        
        Returns a structured analysis of the code with all extractable elements.
        """
    )
    args_schema: Type[BaseModel] = AnalyzeCodeToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.cp_service = CodeProviderService(self.sql_db)
        self.project_service = ProjectService(self.sql_db)
        self.redis = Redis.from_url(config_provider.get_redis_url())

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a function or class node."""
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    arg_str += ": ..."
            args.append(arg_str)

        # *args
        if node.args.vararg:
            vararg = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                try:
                    vararg += f": {ast.unparse(node.args.vararg.annotation)}"
                except:
                    vararg += ": ..."
            args.append(vararg)

        # **kwargs
        if node.args.kwarg:
            kwarg = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                try:
                    kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
                except:
                    kwarg += ": ..."
            args.append(kwarg)

        signature = f"{node.name}({', '.join(args)})"

        # Return type annotation
        if node.returns:
            try:
                signature += f" -> {ast.unparse(node.returns)}"
            except:
                signature += " -> ..."

        return signature

    def _analyze_ast(
        self, tree: ast.AST, include_methods: bool, include_private: bool
    ) -> List[CodeElement]:
        """Analyze AST and extract code elements."""
        elements = []

        class CodeVisitor(ast.NodeVisitor):
            def __init__(self, parent_tool):
                self.current_class = None
                self.class_stack = []
                self.parent_tool = parent_tool

            def visit_ClassDef(self, node):
                is_private = node.name.startswith("_")
                if include_private or not is_private:
                    element = CodeElement(
                        name=node.name,
                        type="class",
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=self.parent_tool._extract_docstring(node),
                        signature=f"class {node.name}",
                        is_private=is_private,
                        is_async=False,
                    )
                    elements.append(element)

                # Visit class methods if requested
                if include_methods:
                    self.class_stack.append(node.name)
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.class_stack.pop()
                    self.current_class = (
                        self.class_stack[-1] if self.class_stack else None
                    )

            def visit_FunctionDef(self, node):
                self._visit_function(node, is_async=False)

            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node, is_async=True)

            def _visit_function(self, node, is_async=False):
                is_private = node.name.startswith("_")

                # Skip if private functions are not included
                if not include_private and is_private:
                    return

                # Skip if it's a method and methods are not included
                if self.current_class and not include_methods:
                    return

                element_type = "method" if self.current_class else "function"

                element = CodeElement(
                    name=node.name,
                    type=element_type,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    docstring=self.parent_tool._extract_docstring(node),
                    parent_class=self.current_class,
                    signature=self.parent_tool._get_function_signature(node),
                    is_private=is_private,
                    is_async=is_async,
                )
                elements.append(element)

        visitor = CodeVisitor(self)
        visitor.visit(tree)
        return elements

    def _run(
        self,
        project_id: str,
        file_path: str,
        include_methods: bool = True,
        include_private: bool = False,
    ) -> Dict[str, Any]:
        try:
            # Check cache first
            cache_key = f"code_analysis:{project_id}:{file_path}:{include_methods}:{include_private}"
            cached_result = self.redis.get(cache_key)
            if cached_result:
                import json

                return json.loads(cached_result.decode("utf-8"))

            details = self._get_project_details(project_id)

            # Get file content
            content = self.cp_service.get_file_content(
                repo_name=details["project_name"],
                file_path=file_path,
                start_line=None,
                end_line=None,
                branch_name=details["branch_name"],
                project_id=project_id,
                commit_id=details["commit_id"],
            )

            # Parse the Python file
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Syntax error in Python file: {str(e)}",
                    "elements": [],
                }

            # Analyze the AST
            elements = self._analyze_ast(tree, include_methods, include_private)

            # Sort elements by line number
            elements.sort(key=lambda x: x.start_line)

            # Convert to dict for JSON serialization
            elements_dict = [element.dict() for element in elements]

            result = {
                "success": True,
                "file_path": file_path,
                "total_elements": len(elements_dict),
                "elements": elements_dict,
                "summary": {
                    "classes": len([e for e in elements_dict if e["type"] == "class"]),
                    "functions": len(
                        [e for e in elements_dict if e["type"] == "function"]
                    ),
                    "methods": len([e for e in elements_dict if e["type"] == "method"]),
                    "private_elements": len(
                        [e for e in elements_dict if e["is_private"]]
                    ),
                    "async_functions": len([e for e in elements_dict if e["is_async"]]),
                },
            }

            # Cache the result for 30 minutes
            import json

            self.redis.setex(cache_key, 1800, json.dumps(result))

            return result

        except Exception as e:
            logging.exception(
                f"Failed to analyze code structure for {file_path}: {str(e)}"
            )
            return {"success": False, "error": str(e), "elements": []}

    async def _arun(
        self,
        project_id: str,
        file_path: str,
        include_methods: bool = True,
        include_private: bool = False,
    ) -> Dict[str, Any]:
        return self._run(project_id, file_path, include_methods, include_private)


def analyze_code_tool(sql_db: Session, user_id: str):
    tool_instance = AnalyzeCodeTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="analyze_code_structure",
        description=tool_instance.description,
        args_schema=AnalyzeCodeToolInput,
    )
