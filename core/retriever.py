"""
Layer 5: 向量检索
=================
在 Milvus 中执行相似度搜索，支持 doc_id 过滤。
支持通过 BGE-Reranker 对结果进行重排序。
"""

from config.settings import settings
from core.embedder import EmbeddingClient
from core.vector_store import MilvusStore
from core.reranker import RerankerClient


class MilvusRetriever:
    """基于 Milvus 的向量检索器 + Reranker。

    组合 EmbeddingClient、MilvusStore 和 RerankerClient，
    完成「查询嵌入 → 相似搜索 (Top-K) → 重排序 (Top-K') → 结果格式化」。
    支持通过 doc_id 过滤只检索特定文档。
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        store: MilvusStore,
        top_k: int = settings.rerank_top_k,
        retrieval_top_k: int = settings.retrieval_top_k,
        enable_reranker: bool = settings.enable_reranker,
    ):
        self._embedder = embedding_client
        self._store = store
        self.top_k = top_k  # 最终返回数
        self.retrieval_top_k = retrieval_top_k  # Milvus 检索数
        self.enable_reranker = enable_reranker
        
        # 初始化 Reranker
        if self.enable_reranker:
            self._reranker = RerankerClient()
        else:
            self._reranker = None

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """执行向量检索 + 可选重排序。

        流程：
        1. 从 Milvus 搜索 Top-retrieval_top_k 结果
        2. 如果启用 Reranker，通过 BGE-Reranker 重排序到 Top-top_k
        3. 否则直接返回 Top-top_k

        Args:
            query:   自然语言查询文本。
            top_k:   最终返回结果数，默认使用初始化时的值。
            doc_ids: 可选，仅在选定的文档列表中检索。

        Returns:
            排序后的检索结果列表，每项包含：
            rank / score / text / doc_id / parent_index / chunk_index / rerank_score (如果启用 reranker)
        """
        final_k = top_k or self.top_k

        print(f"\n🔍 查询: \"{query}\"")
        if doc_ids:
            print(f"   📎 限定文档: {doc_ids}")

        # 1. 生成查询嵌入
        query_embedding = self._embedder.embed([query])

        # 2. 构建搜索参数
        collection = self._store.get_collection()
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}

        # 3. 构建过滤表达式
        expr = f"doc_id in {doc_ids}" if doc_ids else None

        # 4. 执行 Milvus 搜索（搜索 Top-retrieval_top_k）
        retrieval_k = self.retrieval_top_k if self.enable_reranker else final_k
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=retrieval_k,
            expr=expr,
            output_fields=["text", "doc_id", "chunk_index", "parent_index"],
        )

        # 5. 格式化初始结果
        search_results = self._format_results(results[0], retrieval_k, is_reranked=False)

        # 6. 如果启用 Reranker，进行重排序
        if self.enable_reranker and search_results:
            search_results = self._reranker.rerank(query, search_results, top_k=final_k)
            # 更新 rank 字段为重排序后的顺序
            for i, result in enumerate(search_results):
                result["rank"] = i + 1

        return search_results

    # ── 私有方法 ────────────────────────────────

    @staticmethod
    def _format_results(hits, top_k: int, is_reranked: bool = False) -> list[dict]:
        """格式化并打印检索结果。
        
        Args:
            hits: Milvus 搜索结果
            top_k: 结果数量
            is_reranked: 是否已重排序（用于打印消息）
        """
        search_results: list[dict] = []

        title = "重排序前的检索结果" if is_reranked else "初始检索结果"
        print(f"\n{'='*60}")
        print(f"📋 {title} (Top-{top_k})")
        print(f"{'='*60}")

        for i, hit in enumerate(hits):
            text = hit.entity.get("text")
            doc_id = hit.entity.get("doc_id")
            chunk_index = hit.entity.get("chunk_index")
            parent_index = hit.entity.get("parent_index")

            result = {
                "rank": i + 1,
                "score": hit.distance,
                "text": text,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "parent_index": parent_index,
            }
            search_results.append(result)

            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n🏷️  #{i+1}  |  L2: {hit.distance:.4f}  |  doc={doc_id}  parent={parent_index}")
            print(f"   {preview}")

        print(f"\n{'='*60}")
        return search_results
