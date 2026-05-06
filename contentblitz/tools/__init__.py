"""Stateless tool interface scaffolds for ContentBlitz."""

from contentblitz.tools.image import generate_image
from contentblitz.tools.text import generate_text
from contentblitz.tools.web_search import search_web

__all__ = ["generate_text", "search_web", "generate_image"]
