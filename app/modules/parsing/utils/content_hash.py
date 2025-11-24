import hashlib
import re
from typing import Optional

# Cache version for invalidation - bump when prompts/schemas change
CACHE_VERSION = "v1"


def generate_content_hash(node_text: str, node_type: Optional[str] = None) -> str:
    """
    Generate SHA256 hash of normalized node content with cache versioning.

    Args:
        node_text: The code content to hash
        node_type: Optional node type for differentiation

    Returns:
        SHA256 hash as hexadecimal string
    """
    # Normalize content: collapse whitespace, strip leading/trailing space
    normalized_content = re.sub(r"\s+", " ", node_text.strip())

    # Normalize node_type: uppercase and strip whitespace
    normalized_node_type = node_type.upper().strip() if node_type else None

    # Include version and node type for differentiation
    if normalized_node_type:
        hash_input = f"{CACHE_VERSION}:{normalized_node_type}:{normalized_content}"
    else:
        hash_input = f"{CACHE_VERSION}:{normalized_content}"

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def has_unresolved_references(node_text: str) -> bool:
    """
    Check if content contains unresolved reference placeholders.

    Args:
        node_text: The code content to check

    Returns:
        True if content contains unresolved references
    """
    return "Code replaced for brevity. See node_id" in node_text


def is_content_cacheable(node_text: str, min_length: int = 100) -> bool:
    """
    Determine if content is worth caching based on size and complexity.

    Args:
        node_text: The code content to evaluate
        min_length: Minimum character length to consider for caching

    Returns:
        True if content should be cached
    """
    # Cache only substantial content to avoid overhead
    if len(node_text.strip()) < min_length:
        return False

    # Skip very repetitive content (likely generated code)
    lines = node_text.strip().split("\n")
    if len(set(lines)) < len(lines) * 0.3:  # <30% unique lines
        return False

    return True
