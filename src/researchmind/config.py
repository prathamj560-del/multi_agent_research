"""Central application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings.

    Every field can be overridden with an environment variable of the same
    name (case-insensitive). A ``.env`` file in the working directory is
    loaded automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Credentials ------------------------------------------------------
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # -- Model -------------------------------------------------------------
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_request_timeout: float = 45.0
    llm_max_retries: int = 5

    # -- Research pipeline tuning -----------------------------------------
    search_max_results: int = 5
    num_sources: int = 3
    scrape_timeout: float = 10.0
    scrape_max_chars: int = 4000
    scrape_concurrency: int = 3

    # -- Revision loop -----------------------------------------------------
    quality_threshold: int = 8  # critic score (out of 10) at which we stop revising
    max_revisions: int = 2

    # -- Cache -------------------------------------------------------------
    cache_ttl_seconds: int = 3600

    # -- Observability -----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def missing_keys(self) -> list[str]:
        """Names of required credentials that are not configured."""
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing


_load_settings = lru_cache(maxsize=1)(lambda: Settings())

# Test seam: when set (e.g. by pytest fixtures), get_settings() returns this
# instance instead of loading from the environment.
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the cached singleton settings instance (overridable in tests)."""
    if _settings_instance is not None:
        return _settings_instance
    return _load_settings()
