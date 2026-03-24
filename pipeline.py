"""
Pipeline 编排
=============
组合 core 层各模块，提供 ingest() 和 query() 两个高层接口。
支持累积入库、doc_id 过滤检索、父子块策略。

Usage:
    from pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ingest("sample.pdf", filename="sample.pdf")
    results = pipeline.query("什么是 RAG")
"""

import json
import time
import uuid
from collections.abc import Generator

from core.parser import parse_pdf
from core.chunker import split_parent_chunks, split_child_chunks_semantic
from core.embedder import EmbeddingClient
from core.vector_store import MilvusStore
from core.retriever import MilvusRetriever
from core.llm_client import LLMClient
from core.database import get_session
from core.models import Document, ParentChunk
from config.settings import settings


class RAGPipeline:
    """RAG Pipeline 编排器。

    将 PDF 解析 → 分块 → 嵌入 → 存储 → 检索 串联为高层操作：
      1. ingest(pdf_path)  — 加载文档到向量库 + PG（累积入库）
      2. query(text)       — 执行语义检索
      3. query_and_generate_stream(text) — 检索 + 父块替换 + LLM 流式生成
    """

    def __init__(self):
        self._embedder = EmbeddingClient()
        self._store = MilvusStore()
        self._retriever = MilvusRetriever(
            embedding_client=self._embedder,
            store=self._store,
        )
        self._llm: LLMClient | None = None

    @property
    def llm(self) -> LLMClient:
        """懒加载 LLMClient，首次调用时初始化。"""
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    # ── 公开接口 ────────────────────────────────

    def ingest(
        self,
        pdf_path: str,
        filename: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        """将 PDF 文档解析、父子分块、嵌入，并存储到 Milvus + PG。

        Args:
            pdf_path:      PDF 文件路径。
            filename:      原始文件名（用于记录）。
            chunk_size:    兼容保留参数（语义子块模式下不使用）。
            chunk_overlap: 兼容保留参数（语义子块模式下不使用）。

        Returns:
            {"doc_id": ..., "chunk_count": ..., "parent_chunk_count": ...}
        """
        t0 = time.time()
        doc_id = uuid.uuid4().hex[:16]
        _ = (chunk_size, chunk_overlap)

        # Step 1: PDF 解析
        text = parse_pdf(pdf_path)

        # Step 2: 父块分块
        parent_chunks = split_parent_chunks(
            text,
            chunk_size=settings.parent_chunk_size,
            overlap=settings.parent_chunk_overlap,
        )
        print(f"📦 父块分块: {len(parent_chunks)} 个 (大小={settings.parent_chunk_size})")

        # Step 3: 语义子块分块（对每个父块）
        all_child_chunks, all_chunk_indices, all_parent_indices = split_child_chunks_semantic(
            parent_chunks,
            embedding_client=self._embedder,
            breakpoint_threshold_type=settings.semantic_breakpoint_threshold_type,
            breakpoint_threshold_amount=settings.semantic_breakpoint_threshold_amount,
        )
        print(f"✂️  子块总计: {len(all_child_chunks)} 个 (语义分块)")

        # Step 4: 嵌入生成（子块）
        embeddings = self._embedder.embed(all_child_chunks)

        # Step 5: Milvus 存储（累积入库）
        self._store.get_or_create_collection()
        count = self._store.insert(
            doc_id=doc_id,
            chunks=all_child_chunks,
            embeddings=embeddings,
            chunk_indices=all_chunk_indices,
            parent_indices=all_parent_indices,
        )
        self._store.build_index()

        # Step 6: PostgreSQL 存储
        with get_session() as session:
            # 文档元数据
            doc = Document(
                doc_id=doc_id,
                filename=filename or pdf_path,
                chunk_count=len(all_child_chunks),
                parent_chunk_count=len(parent_chunks),
            )
            session.add(doc)

            # 父块文本
            for parent_idx, parent_text in enumerate(parent_chunks):
                pc = ParentChunk(
                    doc_id=doc_id,
                    parent_index=parent_idx,
                    text=parent_text,
                )
                session.add(pc)

        elapsed = time.time() - t0
        print(
            f"\n📥 文档入库完成: doc_id={doc_id}, "
            f"子块={count}, 父块={len(parent_chunks)}, 耗时 {elapsed:.2f}s"
        )
        return {
            "doc_id": doc_id,
            "chunk_count": count,
            "parent_chunk_count": len(parent_chunks),
        }

    def query(
        self,
        text: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[dict]:
        """执行语义检索。

        Args:
            text:   自然语言查询。
            top_k:  返回结果数（可选）。
            doc_id: 可选，仅检索指定文档。

        Returns:
            检索结果列表 [{"rank", "score", "text", "doc_id", "parent_index", ...}]
        """
        t0 = time.time()
        results = self._retriever.search(text, top_k=top_k, doc_id=doc_id)
        elapsed = time.time() - t0
        print(f"⏱️  检索耗时: {elapsed:.2f}s")
        return results

    def _fetch_parent_chunks(self, search_results: list[dict]) -> list[dict]:
        """根据检索结果中的 (doc_id, parent_index) 从 PG 获取父块文本。

        返回去重后的父块列表 [{"doc_id": ..., "parent_index": ..., "text": ...}]
        """
        # 去重：同一 (doc_id, parent_index) 只取一次
        seen: set[tuple[str, int]] = set()
        keys: list[tuple[str, int]] = []
        for r in search_results:
            key = (r["doc_id"], r["parent_index"])
            if key not in seen:
                seen.add(key)
                keys.append(key)

        if not keys:
            return []

        parent_chunks: list[dict] = []
        with get_session() as session:
            for doc_id, parent_index in keys:
                pc = (
                    session.query(ParentChunk)
                    .filter_by(doc_id=doc_id, parent_index=parent_index)
                    .first()
                )
                if pc:
                    parent_chunks.append({
                        "doc_id": doc_id,
                        "parent_index": parent_index,
                        "text": pc.text,
                    })

        print(f"📖 已获取 {len(parent_chunks)} 个父块文本")
        return parent_chunks

    def query_and_generate(
        self,
        text: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> dict:
        """检索 + 父块替换 + LLM 非流式生成。"""
        results = self.query(text, top_k=top_k, doc_id=doc_id)
        parent_chunks = self._fetch_parent_chunks(results)
        answer = self.llm.generate(text, parent_chunks)
        return {"query": text, "results": results, "answer": answer}

    def query_and_generate_stream(
        self,
        text: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> Generator[str, None, None]:
        """检索 + 父块替换 + LLM 流式生成，yield SSE 格式事件。

        事件类型:
          - event: context  — 检索到的子块（含 parent_index）
          - event: token    — LLM 逐步生成的文本
          - event: done     — 生成完毕

        Yields:
            SSE 格式的字符串（每条以 ``\\n\\n`` 结尾）。
        """
        # 1. 检索子块
        results = self.query(text, top_k=top_k, doc_id=doc_id)

        # 2. 发送检索结果（context 事件）
        for r in results:
            data = json.dumps(r, ensure_ascii=False)
            yield f"event: context\ndata: {data}\n\n"

        # 3. 获取父块文本
        parent_chunks = self._fetch_parent_chunks(results)

        # 4. LLM 流式生成（用父块文本作为上下文）
        for token in self.llm.generate_stream(text, parent_chunks):
            data = json.dumps({"content": token}, ensure_ascii=False)
            yield f"event: token\ndata: {data}\n\n"

        # 5. 完成事件
        yield "event: done\ndata: {}\n\n"
