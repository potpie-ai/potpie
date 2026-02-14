"""
Minimal conftest for unit tests.

Unit tests should test individual functions/classes in isolation without
requiring external dependencies like databases, Redis, Neo4j, etc.

For tests that need the full app context, use tests/integration/ instead.
"""

import sys
from pathlib import Path

# Add project root to path for imports (idempotent)
project_root = Path(__file__).parent.parent.parent.resolve()
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
