from __future__ import annotations

import pytest

from contentblitz.tools.cache import clear_cache


@pytest.fixture(autouse=True)
def _isolate_process_cache() -> None:
    clear_cache()
    yield
    clear_cache()
