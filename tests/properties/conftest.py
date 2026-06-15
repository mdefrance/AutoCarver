"""Makes the shared ``strategies`` module importable from subfolder tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
