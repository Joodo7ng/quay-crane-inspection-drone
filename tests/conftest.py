"""Allow tests to run from either the repository or an external runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
