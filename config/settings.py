"""
全局配置
========
从 .env 文件和环境变量中加载配置，统一管理所有常量。

Usage:
    from config.settings import settings
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()



@dataclass(frozen=True)
class Settings:
    """应用配置（不可变）。"""

    # ── OpenAI（Embedding 专用）────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100

    # ── LLM（智谱 GLM）─────────────────────────
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "glm-4-flash")
    )
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # ── Milvus ──────────────────────────────────
    milvus_host: str = field(
        default_factory=lambda: os.getenv("MILVUS_HOST", "localhost")
    )
    milvus_port: str = field(
        default_factory=lambda: os.getenv("MILVUS_PORT", "19530")
    )
    collection_name: str = "rag_knowledge_base"

    # ── PostgreSQL ─────────────────────────────
    pg_host: str = field(
        default_factory=lambda: os.getenv("PG_HOST", "localhost")
    )
    pg_port: str = field(
        default_factory=lambda: os.getenv("PG_PORT", "5432")
    )
    pg_user: str = field(
        default_factory=lambda: os.getenv("PG_USER", "rag_user")
    )
    pg_password: str = field(
        default_factory=lambda: os.getenv("PG_PASSWORD", "rag_password")
    )
    pg_database: str = field(
        default_factory=lambda: os.getenv("PG_DATABASE", "rag_db")
    )

    @property
    def pg_dsn(self) -> str:
        """PostgreSQL 连接串。"""
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    # ── 分块参数 ────────────────────────────────
    parent_chunk_size: int = 1500      # 父块大小
    parent_chunk_overlap: int = 100    # 父块重叠
    chunk_size: int = 500              # 子块大小
    chunk_overlap: int = 50            # 子块重叠

    # ── 检索参数 ────────────────────────────────
    top_k: int = 3


settings = Settings()
