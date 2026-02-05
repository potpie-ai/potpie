#!/usr/bin/env python3
"""
Standalone script to install gVisor.

This script can be run directly to install gVisor runsc binary.
It's designed to be run as part of project setup.

Usage:
    python scripts/install_gvisor.py
    or
    python -m scripts.install_gvisor
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.modules.utils.install_gvisor import main  # noqa: E402

if __name__ == "__main__":
    main()
