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

    # ── OpenAI ──────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100

    # ── Milvus ──────────────────────────────────
    milvus_host: str = field(
        default_factory=lambda: os.getenv("MILVUS_HOST", "localhost")
    )
    milvus_port: str = field(
        default_factory=lambda: os.getenv("MILVUS_PORT", "19530")
    )
    collection_name: str = "rag_simple_demo"

    # ── 分块参数 ────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── 检索参数 ────────────────────────────────
    top_k: int = 3


settings = Settings()
