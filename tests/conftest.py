"""
Pytest configuration and fixtures for Enterprise Vision AI tests.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add parent directory for enterprise_vision_ai
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))
