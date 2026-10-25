"""
    Lifecycle tests.

    The remembered GPU attach outcome (src.lifecycle.client) goes to a temporary directory for the
    whole test run, also for the CLI runs in a subprocess: tests never read or change the user's cache.
"""

import atexit
import os
import shutil
import tempfile

_cache = tempfile.mkdtemp(prefix="tml-test-cache-")
os.environ["TML_CACHE_DIR"] = _cache
atexit.register(shutil.rmtree, _cache, ignore_errors=True)
