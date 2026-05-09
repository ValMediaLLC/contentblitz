"""Stateless tool interface scaffolds for ContentBlitz."""

from contentblitz.tools.cache import build_research_cache_key, get_cached_research, set_cached_research
from contentblitz.tools.generate_text import GenerateTextResult, generate_text as generate_text_result
from contentblitz.tools.image import generate_image
from contentblitz.tools.provider_types import SearchResult, SearchWebResult
from contentblitz.tools.search_web import search_web as search_web_result
from contentblitz.tools.text import generate_text
from contentblitz.tools.web_search import search_web

__all__ = [
    "generate_text",
    "generate_text_result",
    "GenerateTextResult",
    "search_web",
    "search_web_result",
    "SearchResult",
    "SearchWebResult",
    "generate_image",
    "build_research_cache_key",
    "get_cached_research",
    "set_cached_research",
]
