import hashlib
import re
from typing import Optional


def generate_content_hash(node_text: str, node_type: Optional[str] = None) -> str:
    """
    Generate SHA256 hash of normalized node content.

    Args:
        node_text: The code content to hash
        node_type: Optional node type for differentiation

    Returns:
        SHA256 hash as hexadecimal string
    """
    # Normalize content: collapse whitespace, strip leading/trailing space
    normalized_content = re.sub(r"\s+", " ", node_text.strip())

    # Include node type for differentiation if provided
    if node_type:
        hash_input = f"{node_type}:{normalized_content}"
    else:
        hash_input = normalized_content

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


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
