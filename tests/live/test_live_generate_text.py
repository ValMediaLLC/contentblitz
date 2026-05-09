import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from contentblitz.tools.generate_text import generate_text

pytestmark = pytest.mark.skipif(
    os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1",
    reason="Live OpenAI tests are disabled by default.",
)


def test_live_generate_text_openai():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    result = generate_text(
        prompt="Reply with exactly: ContentBlitz live test passed.",
        agent_key="query_handler",
        model="gpt-4o-mini",
        max_tokens=20,
    )

    assert result.degraded is False
    assert "ContentBlitz" in result.text
    assert result.provider == "openai"
    assert result.total_tokens >= 0