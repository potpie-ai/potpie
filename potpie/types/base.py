from enum import Enum


class EdgeType(str, Enum):
    """Types of relationships between code nodes."""

    DEPENDENCY = "dependency"
    REFERENCE = "reference"
    INHERITANCE = "inheritance"
    IMPLIES = "implies"
    ANNOTATED_WITH = "annotated_with"
    DEFINED_IN = "defined_in"


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
