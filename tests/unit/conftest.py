"""
Minimal conftest for unit tests.

Unit tests should test individual functions/classes in isolation without
requiring external dependencies like databases, Redis, Neo4j, etc.

For tests that need the full app context, use tests/integration/ instead.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
