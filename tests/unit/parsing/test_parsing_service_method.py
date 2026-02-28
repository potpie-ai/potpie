"""
Regression test for duplicate_graph indentation bug.

duplicate_graph was defined at module level (0 indentation) instead of
inside the ParsingService class (4-space indent), making it an unreachable
standalone function rather than an instance method. Any caller doing
`service.duplicate_graph(...)` would get AttributeError at runtime.
"""

from app.modules.parsing.graph_construction.parsing_service import ParsingService


def test_duplicate_graph_is_a_method_of_parsing_service():
    assert hasattr(ParsingService, "duplicate_graph"), (
        "duplicate_graph is not a method of ParsingService. "
        "It is defined at module level due to missing indentation."
    )
