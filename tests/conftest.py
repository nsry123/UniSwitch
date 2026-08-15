"""Shared test environment configuration."""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp/universal-switch-numba-tests")))
