"""
Layer 5: 向量检索
=================
在 Milvus 中执行相似度搜索并格式化输出结果。
"""

from config.settings import settings
from core.embedder import EmbeddingClient
from core.vector_store import MilvusStore


class MilvusRetriever:
    """基于 Milvus 的向量检索器。

    组合 EmbeddingClient 和 MilvusStore，完成「查询嵌入 → 相似搜索 → 结果格式化」。
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        store: MilvusStore,
        top_k: int = settings.top_k,
    ):
        self._embedder = embedding_client
        self._store = store
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """执行向量检索。

        Args:
            query: 自然语言查询文本。
            top_k: 返回结果数，默认使用初始化时的值。

        Returns:
            排序后的检索结果列表，每项包含 rank / score / text。
        """
        k = top_k or self.top_k

        print(f"\n🔍 查询: \"{query}\"")

        # 1. 生成查询嵌入
        query_embedding = self._embedder.embed([query])

        # 2. 执行 Milvus 搜索
        collection = self._store.get_collection()
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text"],
        )

        # 3. 格式化结果
        search_results = self._format_results(results[0], k)
        return search_results

    # ── 私有方法 ────────────────────────────────

    @staticmethod
    def _format_results(hits, top_k: int) -> list[dict]:
        """格式化并打印检索结果。"""
        search_results: list[dict] = []

        print(f"\n{'='*60}")
        print(f"📋 Top-{top_k} 检索结果")
        print(f"{'='*60}")

        for i, hit in enumerate(hits):
            text = hit.entity.get("text")
            result = {"rank": i + 1, "score": hit.distance, "text": text}
            search_results.append(result)

            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"\n🏷️  #{i+1}  |  L2 距离: {hit.distance:.4f}")
            print(f"   {preview}")

        print(f"\n{'='*60}")
        return search_results
