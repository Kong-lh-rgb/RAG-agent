"""
Pipeline 编排
=============
组合 core 层各模块，提供 ingest() 和 query() 两个高层接口。

Usage:
    from pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ingest("sample.pdf")
    results = pipeline.query("什么是 RAG")
"""

import time

from core.parser import parse_pdf
from core.chunker import chunk_text
from core.embedder import EmbeddingClient
from core.vector_store import MilvusStore
from core.retriever import MilvusRetriever


class RAGPipeline:
    """RAG Pipeline 编排器。

    将 PDF 解析 → 分块 → 嵌入 → 存储 → 检索 串联为两步操作：
      1. ingest(pdf_path)  — 加载文档到向量库
      2. query(text)       — 执行语义检索
    """

    def __init__(self):
        self._embedder = EmbeddingClient()
        self._store = MilvusStore()
        self._retriever = MilvusRetriever(
            embedding_client=self._embedder,
            store=self._store,
        )

    # ── 公开接口 ────────────────────────────────

    def ingest(
        self,
        pdf_path: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """将 PDF 文档解析、分块、嵌入，并存储到 Milvus。

        Args:
            pdf_path:      PDF 文件路径。
            chunk_size:    分块大小（可选，默认使用配置值）。
            chunk_overlap: 重叠大小（可选，默认使用配置值）。

        Returns:
            插入的文档块数。
        """
        t0 = time.time()

        # Step 1: PDF 解析
        text = parse_pdf(pdf_path)

        # Step 2: 文本分块
        kwargs = {}
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            kwargs["overlap"] = chunk_overlap
        chunks = chunk_text(text, **kwargs)

        # Step 3: 嵌入生成
        embeddings = self._embedder.embed(chunks)

        # Step 4: Milvus 存储
        self._store.connect()
        self._store.recreate_collection()
        count = self._store.insert(chunks, embeddings)
        self._store.build_index()

        elapsed = time.time() - t0
        print(f"\n📥 文档入库完成: {count} 块, 耗时 {elapsed:.2f}s")
        return count

    def query(self, text: str, top_k: int | None = None) -> list[dict]:
        """执行语义检索。

        Args:
            text:  自然语言查询。
            top_k: 返回结果数（可选）。

        Returns:
            检索结果列表 [{"rank", "score", "text"}, ...]。
        """
        t0 = time.time()
        results = self._retriever.search(text, top_k=top_k)
        elapsed = time.time() - t0
        print(f"⏱️  检索耗时: {elapsed:.2f}s")
        return results
