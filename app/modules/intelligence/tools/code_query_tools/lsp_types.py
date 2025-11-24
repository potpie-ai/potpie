"""
Shared types and enumerations for LSP-based code query tooling.

These definitions provide a common schema for modeling the core LSP
capabilities we intend to expose to agents (definitions, references,
hover, and symbol search). Concrete language server integrations can
extend or refine these types as needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LspMethod(str, Enum):
    """Supported LSP request types."""

    DEFINITION = "textDocument/definition"
    REFERENCES = "textDocument/references"
    HOVER = "textDocument/hover"
    DOCUMENT_SYMBOL = "textDocument/documentSymbol"
    WORKSPACE_SYMBOL = "workspace/symbol"


class Position(BaseModel):
    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> "Position":
        return cls(line=data["line"], character=data["character"])

    """Represents a zero-based line/character location in a text document."""

    line: int = Field(..., ge=0, description="Zero-based line index.")
    character: int = Field(..., ge=0, description="Zero-based character offset.")


class TextDocumentIdentifier(BaseModel):
    """Identifies a text document by its URI."""

    uri: str = Field(..., description="file:// URI pointing to the document.")


class Location(BaseModel):
    """Represents a location inside a resource, such as a line inside a text file."""

    uri: str = Field(..., description="file:// URI where the symbol is located.")
    start: Position = Field(..., description="Start position of the range.")
    end: Position = Field(..., description="End position of the range.")

    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> "Location":
        range_data = data.get("range") or {}
        start = Position.from_lsp(range_data.get("start", {"line": 0, "character": 0}))
        end = Position.from_lsp(range_data.get("end", {"line": 0, "character": 0}))
        return cls(uri=data["uri"], start=start, end=end)


class HoverContents(BaseModel):
    """Minimal hover payload wrapper."""

    language: Optional[str] = Field(
        None, description="Optional language hint for the hovered content."
    )
    value: str = Field(..., description="Hover text content.")


class HoverResult(BaseModel):
    """Hover response structure."""

    contents: List[HoverContents] = Field(
        default_factory=list, description="Ordered hover content segments."
    )
    start: Optional[Position] = Field(
        None, description="Optional start of the hover range."
    )
    end: Optional[Position] = Field(
        None, description="Optional end of the hover range."
    )

    @classmethod
    def from_lsp(cls, data: Any) -> "HoverResult":
        contents_field = data.get("contents") if isinstance(data, dict) else data

        def _normalize_content(content: Any) -> HoverContents:
            if isinstance(content, str):
                return HoverContents(language=None, value=content)
            if isinstance(content, dict):
                language = content.get("language")
                value = content.get("value") or content.get("contents") or ""
                return HoverContents(language=language, value=value)
            return HoverContents(language=None, value=str(content))

        if isinstance(contents_field, list):
            contents = [_normalize_content(item) for item in contents_field]
        else:
            contents = [_normalize_content(contents_field)]

        range_data = data.get("range") if isinstance(data, dict) else None
        start_pos = (
            Position.from_lsp(range_data.get("start"))
            if range_data and range_data.get("start")
            else None
        )
        end_pos = (
            Position.from_lsp(range_data.get("end"))
            if range_data and range_data.get("end")
            else None
        )

        return cls(contents=contents, start=start_pos, end=end_pos)


class SymbolInformation(BaseModel):
    @classmethod
    def from_lsp(cls, data: dict[str, Any]) -> "SymbolInformation":
        location_data = data.get("location", {})
        location = (
            Location.from_lsp(location_data)
            if "range" in location_data and "uri" in location_data
            else Location(
                uri=location_data.get("uri", ""),
                start=Position(line=0, character=0),
                end=Position(line=0, character=0),
            )
        )

        return cls(
            name=data.get("name", ""),
            kind=data.get("kind", 0),
            location=location,
            container_name=data.get("containerName"),
        )

    """Workspace or document symbol metadata."""

    name: str = Field(..., description="Symbol name.")
    kind: int = Field(..., description="Symbol kind (LSP SymbolKind enum value).")
    location: Location = Field(..., description="Where the symbol is located.")
    container_name: Optional[str] = Field(
        None, description="Optional enclosing symbol name."
    )


class LspQueryRequest(BaseModel):
    """Input payload for an LSP query."""

    project_id: str = Field(..., description="Project associated with the request.")
    language: str = Field(
        ..., description="Language identifier (e.g., 'python', 'typescript')."
    )
    method: LspMethod = Field(..., description="LSP method to invoke.")
    text_document: Optional[TextDocumentIdentifier] = Field(
        None, description="Target document for position-based queries."
    )
    position: Optional[Position] = Field(
        None, description="Cursor position for definition/references/hover."
    )
    query: Optional[str] = Field(
        None, description="Search query used with symbol requests."
    )


class LspQueryResponse(BaseModel):
    """Normalized response payload sent back to the agent."""

    success: bool = Field(..., description="Indicates call success.")
    method: LspMethod = Field(..., description="Method that produced this response.")
    status_messages: List[str] = Field(
        default_factory=list,
        description="Human-readable status updates about language server startup and execution.",
    )
    locations: List[Location] = Field(
        default_factory=list,
        description="Resolved locations (definitions/references/symbol hits).",
    )
    hover: Optional[HoverResult] = Field(
        None,
        description="Hover information when available.",
    )
    symbols: List[SymbolInformation] = Field(
        default_factory=list,
        description="Symbol search payload for document/workspace symbol requests.",
    )
    error: Optional[str] = Field(
        None, description="Error message if the operation failed."
    )
