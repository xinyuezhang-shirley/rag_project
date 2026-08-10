from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──
    app_name: str = "ai-data-platform"
    app_version: str = "0.1.0"
    environment: str = "dev"  # dev / staging / prod
    debug: bool = False

    # ── Database ──
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_data_platform"

    # ── Redis ──
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── LLM ──
    llm_provider: str = "openai"  # openai / stub
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 2048
    openai_timeout: int = 30
    openai_embedding_model: str = "text-embedding-3-small"

    # ── 作业1：语义相似度缓存 ──
    semantic_cache_enabled: bool = True
    semantic_cache_similarity_threshold: float = 0.95

    # ── 作业2：动态上下文窗口 ──
    llm_context_window_tokens: int = 128000
    history_token_budget_ratio: float = 0.7

    # ── Logging ──
    log_level: str = "INFO"
    log_format: str = "json"  # json / console

    # ── JWT ──
    jwt_secret_key: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 小时

    # ── Encryption ──
    encryption_key: str = "ai-data-platform"  # Fernet key，用于加密数据库密码

    # ── SQL Engine ──
    sql_max_rows: int = 100
    sql_timeout_seconds: int = 15
    sql_sensitive_columns: list[str] = ["password", "ssn", "salary", "credit_card"]

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()