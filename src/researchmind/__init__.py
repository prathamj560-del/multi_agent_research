"""Public API - keep import-time side effects minimal."""

from researchmind.config import Settings, get_settings
from researchmind.models import Critique, ScrapedPage, SearchHit

__all__ = ["Critique", "ScrapedPage", "SearchHit", "Settings", "get_settings"]
__version__ = "1.0.0"
