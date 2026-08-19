"""Shared diff-splitting helper for source-connector resolvers.

Both the GitHub and GitLab pull/merge-request resolvers hand the agent
``snippets`` carved out of a unified diff, and both need the same rule:
at most ``max_chunks`` per-file hunks, each clamped to the caller's
per-item budget and labelled by its file path so the agent can cite the
location. Keeping one implementation here means a fix to the chunking
rule lands for every forge at once.
"""

from __future__ import annotations

from potpie_context_engine.domain.source_resolution import clamp_text


def split_diff_chunks(
    text: str,
    *,
    per_item: int,
    max_chunks: int,
) -> list[tuple[str, str | None]]:
    """Split a unified diff into at most ``max_chunks`` per-file hunks.

    Falls back to a single clamped chunk when the input is not a recognizable
    diff. Each chunk is labelled by its ``+++`` file path when available so
    agents can cite the location.
    """
    if not text or max_chunks <= 0:
        return []
    if "diff --git" not in text and "+++ " not in text:
        return [(clamp_text(text, per_item), None)]
    chunks: list[tuple[str, str | None]] = []
    current_lines: list[str] = []
    current_path: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git"):
            if current_lines:
                chunks.append(
                    (clamp_text("\n".join(current_lines), per_item), current_path)
                )
                if len(chunks) >= max_chunks:
                    return chunks
            current_lines = [line]
            current_path = None
            continue
        if line.startswith("+++ "):
            current_path = line[4:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append((clamp_text("\n".join(current_lines), per_item), current_path))
    return chunks[:max_chunks]


__all__ = ["split_diff_chunks"]
