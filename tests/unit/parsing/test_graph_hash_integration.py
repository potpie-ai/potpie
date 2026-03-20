"""Tests for hash-based caching integration in graph construction and inference."""

import pytest
from app.modules.parsing.utils.content_hash import generate_content_hash, is_content_cacheable


pytestmark = pytest.mark.unit


class TestGraphNodeHashGeneration:
    """Test that content hashes are correctly computed for graph nodes."""

    def test_function_node_gets_hash(self):
        """A function node with substantial code should get a content_hash."""
        node_text = (
            "def calculate_total(items):\n"
            "    total = 0\n"
            "    for item in items:\n"
            "        total += item.price * item.quantity\n"
            "    return total\n"
        )
        assert is_content_cacheable(node_text)
        content_hash = generate_content_hash(node_text, "FUNCTION")
        assert len(content_hash) == 64

    def test_small_node_skipped(self):
        """A trivially small node should not be cached."""
        node_text = "x = 1"
        assert not is_content_cacheable(node_text)

    def test_class_and_function_same_code_different_hash(self):
        """Same code classified as CLASS vs FUNCTION should have different hashes."""
        code = (
            "class Handler:\n"
            "    def __init__(self):\n"
            "        self.data = []\n"
            "    def process(self):\n"
            "        return len(self.data)\n"
        )
        hash_class = generate_content_hash(code, "CLASS")
        hash_func = generate_content_hash(code, "FUNCTION")
        assert hash_class != hash_func


class TestCrossBranchCacheReuse:
    """
    Test the core caching scenario: when the same code appears in
    different branches, the hash should match and inference should be reused.
    """

    def test_identical_code_across_branches_same_hash(self):
        """Same code parsed from two different branches should produce the same hash."""
        code = (
            "async def fetch_user(user_id: str):\n"
            "    response = await http_client.get(f'/users/{user_id}')\n"
            "    if response.status_code != 200:\n"
            "        raise HTTPException(status_code=404)\n"
            "    return response.json()\n"
        )
        hash_branch_a = generate_content_hash(code, "FUNCTION")
        hash_branch_b = generate_content_hash(code, "FUNCTION")
        assert hash_branch_a == hash_branch_b

    def test_modified_code_different_hash(self):
        """Modified code in a new branch should produce a different hash."""
        code_v1 = (
            "def process(data):\n"
            "    return [x * 2 for x in data]\n"
        )
        code_v2 = (
            "def process(data):\n"
            "    return [x * 3 for x in data]  # Changed multiplier\n"
        )
        hash_v1 = generate_content_hash(code_v1, "FUNCTION")
        hash_v2 = generate_content_hash(code_v2, "FUNCTION")
        assert hash_v1 != hash_v2

    def test_cache_lookup_simulation(self):
        """Simulate the full cache flow: store from branch A, lookup from branch B."""
        code = (
            "def validate_input(data: dict) -> bool:\n"
            "    required = ['name', 'email', 'age']\n"
            "    return all(k in data for k in required)\n"
        )
        # Branch A: generate hash and simulate storing inference
        hash_a = generate_content_hash(code, "FUNCTION")
        cached_inference = {
            "docstring": "Validates that input dict contains required keys.",
            "tags": ["validation", "input"],
        }
        cache_store = {hash_a: cached_inference}

        # Branch B: generate hash for same code and look up
        hash_b = generate_content_hash(code, "FUNCTION")
        assert hash_b in cache_store
        assert cache_store[hash_b]["docstring"] == cached_inference["docstring"]

    def test_partial_branch_changes(self):
        """
        When a branch modifies only some files, unchanged nodes should cache-hit
        while modified nodes should cache-miss.
        """
        # Unchanged function
        unchanged_code = (
            "def helper(x):\n"
            "    return x.strip().lower()\n"
            "    # normalize input string\n"
        )
        # Modified function
        original_code = "def main():\n    print('hello')\n    return 0\n"
        modified_code = "def main():\n    print('goodbye')\n    return 1\n"

        # Build cache from original branch
        cache = {}
        cache[generate_content_hash(unchanged_code, "FUNCTION")] = {"docstring": "Helper normalizes strings"}
        cache[generate_content_hash(original_code, "FUNCTION")] = {"docstring": "Main prints hello"}

        # Check from new branch
        unchanged_hash = generate_content_hash(unchanged_code, "FUNCTION")
        modified_hash = generate_content_hash(modified_code, "FUNCTION")

        assert unchanged_hash in cache  # Cache hit
        assert modified_hash not in cache  # Cache miss
