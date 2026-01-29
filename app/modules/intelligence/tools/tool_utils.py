"""
Utility functions for tools to handle common operations like response truncation.
"""

import logging

logger = logging.getLogger(__name__)

# Character limit for tool responses to prevent sending insanely large content to LLM
MAX_RESPONSE_LENGTH = 80000  # 80k characters


def truncate_response(content: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    """Truncate response content if it exceeds the maximum length and add truncation notice.

    Args:
        content: The content to truncate
        max_length: Maximum length in characters (default: 80,000)

    Returns:
        Truncated content with notice if truncation occurred, original content otherwise
    """
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]
    notice = f"\n\n⚠️ [TRUNCATED] Response truncated: showing first {max_length:,} characters of {len(content):,} total characters. The response may be incomplete."
    return truncated + notice


def truncate_dict_response(
    response: dict, max_length: int = MAX_RESPONSE_LENGTH
) -> dict:
    """Truncate string values in a dictionary response if they exceed the maximum length.

    This is useful for tools that return dictionaries with potentially large string values.
    The function will truncate string values and add truncation notices.

    Args:
        response: Dictionary response that may contain large string values
        max_length: Maximum length in characters for each string value (default: 80,000)

    Returns:
        Dictionary with truncated string values and notices if truncation occurred
    """
    truncated_response = {}
    for key, value in response.items():
        if isinstance(value, str):
            original_length = len(value)
            truncated_value = truncate_response(value, max_length)
            if len(truncated_value) > max_length:
                # Truncation occurred
                logger.warning(
                    f"Truncated '{key}' field from {original_length} to {max_length} characters"
                )
            truncated_response[key] = truncated_value
        elif isinstance(value, dict):
            # Recursively truncate nested dictionaries
            truncated_response[key] = truncate_dict_response(value, max_length)
        elif isinstance(value, list):
            # Truncate list items if they are strings or dicts
            truncated_list = []
            for item in value:
                if isinstance(item, str):
                    truncated_list.append(truncate_response(item, max_length))
                elif isinstance(item, dict):
                    truncated_list.append(truncate_dict_response(item, max_length))
                else:
                    truncated_list.append(item)
            truncated_response[key] = truncated_list
        else:
            truncated_response[key] = value

    return truncated_response
