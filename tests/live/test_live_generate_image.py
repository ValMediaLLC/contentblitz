import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from contentblitz.tools.generate_image import generate_image


pytestmark = pytest.mark.skipif(
    os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1"
    or os.getenv("CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS") != "1",
    reason="Live image tests are disabled by default.",
)


def test_live_generate_image_openai():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    result = generate_image(
        prompt=(
            "A clean futuristic content marketing dashboard, "
            "abstract interface, professional SaaS aesthetic, no text."
        ),
        model="dall-e-3",
        size="1024x1024",
    )

    print("\nIMAGE RESULT")
    print("------------")
    print("Provider:", result.provider)
    print("Model:", result.model)
    print("Degraded:", result.degraded)
    print("Error:", result.error)
    print("Image URL:", result.image_url)
    print("Revised Prompt:", result.revised_prompt)

    assert result.degraded is False
    assert result.provider == "openai"
    assert result.model in {"dall-e-3", "dall-e-2"}
    assert result.image_url
    assert "base64" not in str(result.image_url).lower()
