from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FunctionAnalysis(BaseModel):
    explanation: str = Field(
        ...,
        description="Detailed explanation of the function's purpose and implementation",
    )
    functions_called: List[str] = Field(
        default_factory=list,
        description="List of other functions called within this function",
    )
    returns: Optional[str] = Field(
        None, description="Description of what the function returns"
    )
    parameters: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of parameters with their types and descriptions",
    )
    complexity: Optional[str] = Field(
        None, description="Estimated time complexity of the function"
    )
    libraries_used: List[str] = Field(
        default_factory=list,
        description="Libraries or modules used within this function",
    )


class ClassAnalysis(BaseModel):
    explanation: str = Field(
        ...,
        description="Detailed explanation of the class's purpose and implementation",
    )
    functions_defined: List[str] = Field(
        default_factory=list, description="List of functions defined in this class"
    )
    inherits_extends: List[str] = Field(
        default_factory=list,
        description="List of classes this class inherits from or extends",
    )
    attributes: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of class attributes with their types and descriptions",
    )
    methods: Dict[str, FunctionAnalysis] = Field(
        default_factory=dict,
        description="Detailed analysis of each method in the class",
    )


class FileAnalysis(BaseModel):
    apis_defined: List[str] = Field(
        default_factory=list, description="List of APIs defined in this file"
    )
    imported_files: List[str] = Field(
        default_factory=list, description="List of project files imported in this file"
    )
    libraries_used: List[str] = Field(
        default_factory=list, description="List of external libraries used in this file"
    )
    classes_defined: Dict[str, ClassAnalysis] = Field(
        default_factory=dict,
        description="Detailed analysis of each class defined in the file",
    )
    functions_defined: Dict[str, FunctionAnalysis] = Field(
        default_factory=dict,
        description="Detailed analysis of each function defined in the file",
    )
    file_purpose: str = Field(
        ..., description="Overall purpose and role of this file in the project"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of other files or modules this file depends on",
    )


class CodebaseAnalysis(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_description: str = Field(
        ..., description="Overall description of the project"
    )
    files: Dict[str, FileAnalysis] = Field(
        ..., description="Detailed analysis of each file in the project"
    )
    main_entry_points: List[str] = Field(
        default_factory=list,
        description="List of main entry points or important files in the project",
    )
    project_structure: Dict[str, List[str]] = Field(
        default_factory=dict, description="Directory structure of the project"
    )
    global_dependencies: List[str] = Field(
        default_factory=list,
        description="List of global project dependencies or requirements",
    )
