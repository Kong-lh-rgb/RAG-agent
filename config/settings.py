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

    # ── OpenAI 兼容配置（主要用于评测）──────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    # ── Embedding（SentenceTransformers）────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )
    eval_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_normalize: bool = (
        os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    )

    # ── LLM─────────────────────────
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "glm-5")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL")
    )
    judge_api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY")
    )
    judge_base_url: str = field(
        default_factory=lambda: os.getenv(
            "JUDGE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
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
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "900"))
    parent_chunk_overlap: int = int(os.getenv("PARENT_CHUNK_OVERLAP", "100"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    semantic_breakpoint_threshold_type: str = field(
        default_factory=lambda: os.getenv(
            "SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile"
        )
    )
    semantic_breakpoint_threshold_amount: int = int(
        os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "85")
    )

    # ── 检索参数 ────────────────────────────────
    top_k: int = 3  # 最终返回结果数（rerank_top_k 的默认值）
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))  # Milvus 搜索 Top-K
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "3"))  # 重排序后返回 Top-K（通常等于 top_k）

    # ── Reranker（CrossEncoder）─────────────────
    enable_reranker: bool = (
        os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
    reranker_use_fp16: bool = (
        os.getenv("RERANKER_USE_FP16", "true").lower() == "true"
    )


settings = Settings()
