"""Central configuration. Every value can be overridden with an env var or .env file."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agentic Research Engine"
    version: str = "0.1.0"

    # ---- LLM -------------------------------------------------------------
    # auto -> openai if OPENAI_API_KEY is set, else anthropic if ANTHROPIC_API_KEY, else mock
    llm_provider: str = "auto"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = ""
    llm_timeout_s: float = 60.0

    # ---- Search / scraping ----------------------------------------------
    # auto -> serper if SERPER_API_KEY is set, else mock
    search_provider: str = "auto"
    serper_api_key: Optional[str] = None
    http_timeout_s: float = 15.0
    max_page_chars: int = 6000
    max_page_bytes: int = 2_000_000

    # ---- Research loop caps ---------------------------------------------
    max_subtasks: int = 4
    results_per_query: int = 4
    scrape_per_query: int = 2
    max_verify_passes: int = 2          # max follow-up research rounds after the first verification
    max_followup_queries: int = 3       # per follow-up round
    graph_recursion_limit: int = 40     # hard LangGraph super-step ceiling

    # ---- Budget caps (per job) ------------------------------------------
    job_token_budget: int = 60_000
    max_llm_calls: int = 30
    max_search_calls: int = 15
    max_scrape_calls: int = 20
    max_concurrent_jobs: int = 2

    # ---- API / integration ----------------------------------------------
    api_key: Optional[str] = None       # if set, write endpoints require X-API-Key
    webhook_secret: Optional[str] = None
    default_callback_url: Optional[str] = None
    db_path: str = "data/engine.db"

    # ---- Outreach --------------------------------------------------------
    outreach_dry_run: bool = True
    outreach_max_followups_cap: int = 3
    outreach_min_interval_minutes: int = 1
    outreach_token_budget: int = 20_000
    scheduler_tick_seconds: int = 30
    sendgrid_api_key: Optional[str] = None
    email_from: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_sms_from: Optional[str] = None
    twilio_whatsapp_from: Optional[str] = None

    # ---- Keep-alive (Render free tier) --------------------------------------
    keepalive_enabled: bool = True
    keepalive_interval_seconds: int = 300        # floor of 240 s is enforced in code
    keepalive_active_hours: str = ""             # e.g. "7-23"; empty = always
    keepalive_timezone: str = "Asia/Kolkata"
    keepalive_url: Optional[str] = None          # override; otherwise RENDER_EXTERNAL_URL
    render_external_url: Optional[str] = None    # set automatically by Render

    @property
    def resolved_llm_provider(self) -> str:
        p = self.llm_provider.lower()
        if p != "auto":
            return p
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return "mock"

    @property
    def resolved_search_provider(self) -> str:
        p = self.search_provider.lower()
        if p != "auto":
            return p
        return "serper" if self.serper_api_key else "mock"


@lru_cache
def get_settings() -> Settings:
    return Settings()
