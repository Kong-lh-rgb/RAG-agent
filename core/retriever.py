"""
Layer 5: 向量检索
=================
在 Milvus 中执行相似度搜索，支持 doc_id 过滤。
"""

from config.settings import settings
from core.embedder import EmbeddingClient
from core.vector_store import MilvusStore


class MilvusRetriever:
    """基于 Milvus 的向量检索器。

    组合 EmbeddingClient 和 MilvusStore，完成「查询嵌入 → 相似搜索 → 结果格式化」。
    支持通过 doc_id 过滤只检索特定文档。
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

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[dict]:
        """执行向量检索。

        Args:
            query:  自然语言查询文本。
            top_k:  返回结果数，默认使用初始化时的值。
            doc_id: 可选，仅检索指定文档的内容。

        Returns:
            排序后的检索结果列表，每项包含
            rank / score / text / doc_id / parent_index / chunk_index。
        """
        k = top_k or self.top_k

        print(f"\n🔍 查询: \"{query}\"")
        if doc_id:
            print(f"   📎 限定文档: {doc_id}")

        # 1. 生成查询嵌入
        query_embedding = self._embedder.embed([query])

        # 2. 构建搜索参数
        collection = self._store.get_collection()
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}

        # 3. 构建过滤表达式
        expr = f'doc_id == "{doc_id}"' if doc_id else None

        # 4. 执行 Milvus 搜索
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["text", "doc_id", "chunk_index", "parent_index"],
        )

        # 5. 格式化结果
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
