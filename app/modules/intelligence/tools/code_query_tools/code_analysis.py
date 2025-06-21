import logging
import os
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Optional, Type, Dict, Any, List

from pydantic import BaseModel, Field
from redis import Redis
from sqlalchemy.orm import Session
from grep_ast import filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter_languages import get_language, get_parser

from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.projects.projects_service import ProjectService
from app.core.config_provider import config_provider

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)

Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())


class UniversalAnalyzeCodeToolInput(BaseModel):
    project_id: str = Field(
        ..., description="Project ID that references the repository"
    )
    file_path: str = Field(..., description="Path to the file within the repo")
    include_methods: bool = Field(
        True, description="Whether to include class methods in the analysis"
    )
    include_private: bool = Field(
        False,
        description="Whether to include private functions/methods (starting with _ or private keyword)",
    )
    language: Optional[str] = Field(
        None,
        description="Programming language (auto-detected from file extension if not provided)",
    )


class CodeElement(BaseModel):
    name: str
    type: str  # 'function', 'class', 'method', 'interface', 'struct', 'enum', 'variable', 'constant'
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None  # For methods
    signature: str  # Function/method signature
    is_private: bool = False
    is_async: bool = False
    is_static: bool = False
    visibility: Optional[str] = None  # public, private, protected, internal
    language: str
    parameters: List[str] = []
    return_type: Optional[str] = None


class UniversalCodeAnalyzer:
    """Universal code analyzer using Tree-sitter for multiple programming languages."""

    def __init__(self):
        self.warned_files = set()

    def get_scm_fname(self, lang):
        # Get the repository root (adjust the number of parent() calls as needed)
        repo_root = Path(
            __file__
        ).parent.parent.parent.parent.parent  # or however many levels up
        query_file = (
            repo_root
            / "modules/parsing/graph_construction/queries"
            / f"tree-sitter-{lang}-tags.scm"
        )

        return query_file if query_file.exists() else None

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension using grep_ast."""
        return filename_to_lang(file_path)

    def get_tags_raw(self, fname, rel_fname, code=None):
        """Extract tags from source code using tree-sitter queries."""
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as e:
            logging.warning(f"Could not get language/parser for {lang}: {e}")
            return

        query_scm = self.get_scm_fname(lang)
        if not query_scm or not query_scm.exists():
            return

        query_scm_content = query_scm.read_text()

        if code is None:
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                logging.warning(f"Could not read file {fname}: {e}")
                return

        if not code:
            return

        try:
            tree = parser.parse(bytes(code, "utf-8"))
        except Exception as e:
            logging.warning(f"Could not parse code for {fname}: {e}")
            return

        # Run the tags queries
        try:
            query = language.query(query_scm_content)
            captures = query.captures(tree.root_node)
            captures = list(captures)
        except Exception as e:
            logging.warning(f"Could not run query for {fname}: {e}")
            return

        saw = set()

        for node, tag in captures:
            try:
                node_text = node.text.decode("utf-8")
            except UnicodeDecodeError:
                continue

            if tag.startswith("name.definition."):
                kind = "def"
                type_name = tag.split(".")[-1]
            elif tag.startswith("name.reference."):
                kind = "ref"
                type_name = tag.split(".")[-1]
            else:
                continue

            saw.add(kind)

            # Enhanced node text extraction for Java methods
            if lang == "java" and type_name == "method":
                # Handle method calls with object references (e.g., productService.listAllProducts())
                parent = node.parent
                if parent and parent.type == "method_invocation":
                    object_node = parent.child_by_field_name("object")
                    if object_node:
                        try:
                            object_text = object_node.text.decode("utf-8")
                            node_text = f"{object_text}.{node_text}"
                        except UnicodeDecodeError:
                            pass

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node_text,
                kind=kind,
                line=node.start_point[0],
                end_line=node.end_point[0],
                type=type_name,
            )

            yield result

        # If we only saw definitions but no references, use pygments to backfill
        if "ref" in saw:
            return
        if "def" not in saw:
            return

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                end_line=-1,
                type="unknown",
            )

    def get_tags(self, fname, rel_fname, code=None):
        """Get tags for a file, handling file existence checks."""
        if not os.path.isfile(fname):
            if fname not in self.warned_files:
                if os.path.exists(fname):
                    logging.warning(f"Can't include {fname}, it is not a normal file")
                else:
                    logging.warning(f"Can't include {fname}, it no longer exists")
                self.warned_files.add(fname)
            return []

        try:
            return list(self.get_tags_raw(fname, rel_fname, code))
        except Exception as e:
            logging.warning(f"Error getting tags for {fname}: {e}")
            return []

    def _extract_docstring_comment(
        self, node_text: str, language: str, start_line: int, code_lines: List[str]
    ) -> Optional[str]:
        """Extract docstring or comment for a code element."""
        if language == "python":
            # Look for triple-quoted strings at the beginning of function/class
            lines = node_text.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Find the end of the docstring
                    quote_type = '"""' if stripped.startswith('"""') else "'''"
                    if stripped.count(quote_type) >= 2:
                        # Single line docstring
                        return stripped.strip(quote_type).strip()
                    else:
                        # Multi-line docstring
                        docstring_lines = [stripped[3:]]  # Remove opening quotes
                        for j in range(i + 1, len(lines)):
                            if quote_type in lines[j]:
                                docstring_lines.append(
                                    lines[j][: lines[j].find(quote_type)]
                                )
                                break
                            docstring_lines.append(lines[j])
                        return "\n".join(docstring_lines).strip()

        elif language in ["javascript", "typescript", "java", "cpp", "c", "php"]:
            # Look for comments before the function/class
            if start_line > 0:
                prev_line = (
                    code_lines[start_line - 1].strip()
                    if start_line - 1 < len(code_lines)
                    else ""
                )
                if prev_line.startswith("/*") and prev_line.endswith("*/"):
                    return prev_line[2:-2].strip()
                elif prev_line.startswith("//"):
                    return prev_line[2:].strip()

        return None

    def _extract_signature(self, node_text: str, language: str) -> str:
        """Extract function/method signature."""
        lines = node_text.split("\n")

        if language == "python":
            # Extract until the colon
            signature_parts = []
            for line in lines:
                signature_parts.append(line.strip())
                if line.strip().endswith(":"):
                    break
            return " ".join(signature_parts).replace(":", "").strip()

        elif language in ["javascript", "typescript"]:
            # Extract until opening brace
            if "{" in node_text:
                return node_text.split("{")[0].strip()
            return lines[0].strip()

        elif language in ["java", "cpp", "c"]:
            # Extract method/function declaration
            if "{" in node_text:
                return node_text.split("{")[0].strip()
            elif ";" in node_text:
                return node_text.split(";")[0].strip()
            return lines[0].strip()

        else:
            # Default: return first line or up to opening brace
            first_line = lines[0].strip()
            if "{" in first_line:
                return first_line.split("{")[0].strip()
            return first_line

    def _is_private(self, name: str, signature: str, language: str) -> bool:
        """Check if a code element is private based on naming conventions or keywords."""
        if language == "python":
            return name.startswith("_")
        elif language in ["javascript", "typescript"]:
            return name.startswith("_") or name.startswith("#")
        elif language == "java":
            return "private" in signature.lower()
        elif language in ["cpp", "c"]:
            return "private" in signature.lower()
        elif language == "php":
            return name.startswith("_") or "private" in signature.lower()
        elif language == "ruby":
            return name.startswith("_")
        return False

    def _get_element_type(self, tag_type: str, parent_class: Optional[str]) -> str:
        """Map tag type to our element type."""
        type_mapping = {
            "function": "method" if parent_class else "function",
            "method": "method",
            "class": "class",
            "interface": "interface",
            "struct": "struct",
            "enum": "enum",
            "type": "type",
            "namespace": "namespace",
            "module": "module",
            "trait": "trait",
            "impl": "impl",
        }
        return type_mapping.get(tag_type, tag_type)

    def analyze(
        self,
        source_code: str,
        language: str,
        file_path: str = "unknown",
        include_methods: bool = True,
        include_private: bool = False,
    ) -> List[CodeElement]:
        """Analyze source code and extract all code elements."""
        if not language:
            language = self.detect_language(file_path)
            if not language:
                raise ValueError(f"Could not detect language for file: {file_path}")

        # Get tags using the tree-sitter approach
        tags = list(self.get_tags_raw(file_path, file_path, source_code))

        if not tags:
            return []

        elements = []
        code_lines = source_code.split("\n")

        # Group tags by type and track class context
        definitions = [tag for tag in tags if tag.kind == "def"]
        current_class = None

        for tag in definitions:
            # Determine if this is inside a class
            parent_class = None
            if tag.type in ["method"] or (tag.type == "function" and current_class):
                parent_class = current_class
            elif tag.type == "class":
                current_class = tag.name

            element_type = self._get_element_type(tag.type, parent_class)

            # Skip methods if not included
            if element_type == "method" and not include_methods:
                continue

            # Extract signature from the code
            try:
                start_line = max(0, tag.line)
                end_line = min(len(code_lines) - 1, tag.end_line)
                node_text = "\n".join(code_lines[start_line : end_line + 1])
            except (IndexError, ValueError):
                node_text = tag.name

            signature = self._extract_signature(node_text, language)
            is_private = self._is_private(tag.name, signature, language)

            # Skip private elements if not included
            if not include_private and is_private:
                continue

            # Extract docstring
            docstring = self._extract_docstring_comment(
                node_text, language, tag.line, code_lines
            )

            # Detect async and static
            is_async = "async" in signature.lower()
            is_static = "static" in signature.lower()

            element = CodeElement(
                name=tag.name,
                type=element_type,
                start_line=tag.line + 1,  # Convert to 1-indexed
                end_line=tag.end_line + 1,
                start_column=0,  # Tree-sitter queries don't provide column info easily
                end_column=0,
                docstring=docstring,
                parent_class=parent_class,
                signature=signature,
                is_private=is_private,
                is_async=is_async,
                is_static=is_static,
                language=language,
            )

            elements.append(element)

        # Sort by line number
        elements.sort(key=lambda x: x.start_line)
        return elements


class UniversalAnalyzeCodeTool:
    name: str = "analyze_code_structure_universal"
    description: str = (
        """Universal code structure analyzer that works with multiple programming languages using Tree-sitter.
        Supports Python, JavaScript, TypeScript, Java, C++, C, Rust, Go, PHP, Ruby, and more.

        Extracts detailed information about:
        - All classes, structs, interfaces, enums with their docstrings and line ranges
        - All functions and methods with signatures and docstrings
        - Exact start and end line numbers for each code element
        - Language-specific elements (traits in Rust, namespaces in C++, modules in Ruby, etc.)
        - Visibility and access modifiers
        - Async/static function detection

        param project_id: string, the repository ID (UUID) to analyze
        param file_path: string, the path to the file in the repository
        param include_methods: bool, whether to include class methods (default: True)
        param include_private: bool, whether to include private functions/methods (default: False)
        param language: string, programming language (auto-detected from file extension if not provided)

        Returns a structured analysis of the code with all extractable elements.
        """
    )
    args_schema: Type[BaseModel] = UniversalAnalyzeCodeToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.cp_service = CodeProviderService(self.sql_db)
        self.project_service = ProjectService(self.sql_db)
        self.redis = Redis.from_url(config_provider.get_redis_url())
        self.analyzer = UniversalCodeAnalyzer()

    def _get_project_details(self, project_id: str) -> Dict[str, str]:
        details = self.project_service.get_project_from_db_by_id_sync(project_id)
        if not details or "project_name" not in details:
            raise ValueError(f"Cannot find repo details for project_id: {project_id}")
        if details["user_id"] != self.user_id:
            raise ValueError(
                f"Cannot find repo details for project_id: {project_id} for current user"
            )
        return details

    def _run(
        self,
        project_id: str,
        file_path: str,
        include_methods: bool = True,
        include_private: bool = False,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            # Detect language if not provided
            if not language:
                language = self.analyzer.detect_language(file_path)
                if not language:
                    return {
                        "success": False,
                        "error": f"Could not detect programming language for file: {file_path}",
                        "elements": [],
                    }

            # Check cache first
            cache_key = f"universal_code_analysis:{project_id}:{file_path}:{include_methods}:{include_private}:{language}"
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

            # Analyze the code
            try:
                elements = self.analyzer.analyze(
                    content, language, file_path, include_methods, include_private
                )
            except Exception as e:
                logging.exception(f"Failed to analyze {language} file: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to parse {language} file: {str(e)}",
                    "elements": [],
                }

            # Convert to dict for JSON serialization
            elements_dict = [element.dict() for element in elements]

            result = {
                "success": True,
                "file_path": file_path,
                "language": language,
                "total_elements": len(elements_dict),
                "elements": elements_dict,
                "summary": {
                    "classes": len([e for e in elements_dict if e["type"] == "class"]),
                    "functions": len(
                        [e for e in elements_dict if e["type"] == "function"]
                    ),
                    "methods": len([e for e in elements_dict if e["type"] == "method"]),
                    "interfaces": len(
                        [e for e in elements_dict if e["type"] == "interface"]
                    ),
                    "structs": len([e for e in elements_dict if e["type"] == "struct"]),
                    "enums": len([e for e in elements_dict if e["type"] == "enum"]),
                    "private_elements": len(
                        [e for e in elements_dict if e["is_private"]]
                    ),
                    "async_functions": len([e for e in elements_dict if e["is_async"]]),
                    "static_functions": len(
                        [e for e in elements_dict if e["is_static"]]
                    ),
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
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._run(
            project_id, file_path, include_methods, include_private, language
        )


def universal_analyze_code_tool(sql_db: Session, user_id: str):
    """Factory function to create the universal code analysis tool."""
    from langchain.tools import StructuredTool

    tool_instance = UniversalAnalyzeCodeTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance._arun,
        func=tool_instance._run,
        name="analyze_code_structure",
        description=tool_instance.description,
        args_schema=UniversalAnalyzeCodeToolInput,
    )
