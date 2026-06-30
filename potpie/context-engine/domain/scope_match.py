"""Hierarchical scope matching for retrieval (R4).

Scope is a hierarchy — ``repo › service › path-prefix › symbol`` (plus
``environment`` for infra). A repo-wide preference must surface for a file in
that repo; a ``src/payments/**`` rule must surface for ``src/payments/client.py``.
Flat exact-string equality (the old ``_scope_overlap``) misses all of that.

This is the single shared scope matcher, used by the readers (and, later, the
unified retriever) so scope logic lives in one place. It scores how well a
*rule's* scope applies to a *task's* scope in [0, 1]:

- a rule dimension that conflicts with the task (``service=ledger`` vs
  ``service=payments``) drives the score down — the rule does not apply;
- a rule dimension the task does not constrain is neutral (the rule *may* apply);
- ``file_path`` matches by prefix/containment, not equality;
- a rule with no scope at all is global (neutral 0.5).
"""

from __future__ import annotations

from typing import Any, Mapping

# ``path``/``file_path`` and ``symbol`` use containment; every other dimension
# (language / framework / repo / service / audience / environment) is exact.
_PATH_KEYS = ("file_path", "path")
_SYMBOL_KEYS = ("symbol", "function_name")


def _norm(scope: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in scope.items():
        if isinstance(value, str) and value.strip():
            out[key] = value.strip().lower()
    return out


def _strip_glob(path: str) -> str:
    p = path.strip().rstrip("/")
    for suffix in ("/**", "/*", "**", "*"):
        if p.endswith(suffix):
            p = p[: -len(suffix)].rstrip("/")
    return p


def path_contains(rule_path: str, task_path: str) -> bool:
    """True iff ``rule_path`` (a dir / prefix / glob) contains ``task_path``."""
    rule = _strip_glob(rule_path)
    task = task_path.strip().rstrip("/")
    if not rule:
        return True  # a bare glob == repo-wide
    return task == rule or task.startswith(rule + "/")


def hierarchical_scope_overlap(
    task_scope: Mapping[str, Any], rule_scope: Mapping[str, Any]
) -> float:
    """Score how well ``rule_scope`` applies to ``task_scope`` in [0, 1].

    A broader rule (fewer / higher dimensions) that is consistent with the task
    scores high; a rule that conflicts on any concrete dimension scores low.
    """
    rule = _norm(rule_scope)
    task = _norm(task_scope)
    if not rule:
        return 0.5  # global rule — applies neutrally everywhere

    matched = 0.0
    total = 0.0
    for key, rule_val in rule.items():
        total += 1.0
        task_val = task.get(key)
        if key in _PATH_KEYS:
            task_path = task.get("file_path") or task.get("path")
            if task_path is None:
                matched += 0.5  # task didn't name a file — rule may still apply
            elif path_contains(rule_val, task_path):
                matched += 1.0
            # else 0: the rule governs a different subtree.
        elif key in _SYMBOL_KEYS:
            task_sym = task.get("symbol") or task.get("function_name")
            if task_sym is None:
                matched += 0.5
            elif task_sym == rule_val:
                matched += 1.0
        else:  # exact-match dimensions
            if task_val is None:
                matched += 0.5
            elif task_val == rule_val:
                matched += 1.0
            # else 0: a conflicting concrete dimension.
    return matched / total if total else 0.5


__all__ = ["hierarchical_scope_overlap", "path_contains"]
