from __future__ import annotations

import re
from typing import Dict, Any


class ContextCompressor:
    """
    Lightweight code/context compressor to reduce low-signal tokens before
    sending code snippets to the LLM.

    Current strategy (intentionally conservative):
    - Strip full-line comments for common languages (//, #).
    - Strip block comments / docstrings (/* */, /** */, triple quotes).
    - Normalize consecutive blank lines.
    """

    _single_line_comment_re = re.compile(
        r"^\s*(//|#).*$"
    )  # matches // comment or # comment

    _block_comment_re = re.compile(
        r"(/\*[\s\S]*?\*/|\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')", re.MULTILINE
    )

    @classmethod
    def compress_code(cls, code: str) -> str:
        if not code:
            return code

        # Remove block comments and docstrings
        without_blocks = re.sub(cls._block_comment_re, "", code)

        # Remove full-line comments
        lines = []
        for line in without_blocks.splitlines():
            if cls._single_line_comment_re.match(line):
                continue
            lines.append(line)

        # Collapse multiple blank lines
        compressed_lines = []
        blank_streak = 0
        for line in lines:
            if not line.strip():
                blank_streak += 1
                if blank_streak > 1:
                    continue
            else:
                blank_streak = 0
            compressed_lines.append(line)

        return "\n".join(compressed_lines)

    @classmethod
    def compress_tool_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience helper for tools that return dicts with 'code_content' keys.
        Returns a shallow copy with compressed content.
        """
        if not isinstance(result, dict):
            return result

        new_result = dict(result)
        code_key_candidates = ["code_content", "code", "content"]
        for key in code_key_candidates:
            if key in new_result and isinstance(new_result[key], str):
                new_result[key] = cls.compress_code(new_result[key])

        return new_result

