"""
Layer 3: 嵌入生成
=================
封装 SentenceTransformer，提供批量文本嵌入能力。
"""
import os
hf_token = os.getenv("HUGGINGFACE_API_KEY")

from sentence_transformers import SentenceTransformer


from config.settings import settings


class EmbeddingClient:
    """SentenceTransformer Embedding 客户端。"""

    def __init__(
        self,
        model: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
        normalize_embeddings: bool = settings.embedding_normalize,
    ):
        self._model = SentenceTransformer(model)
        self.model = model
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入同序的嵌入向量列表。
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(batch_embeddings.tolist())
            print(f"🧠 嵌入生成: {min(i + self.batch_size, len(texts))}/{len(texts)}")

        print(f"🧠 嵌入生成完成: {len(all_embeddings)} 条, 维度={len(all_embeddings[0])}")
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """兼容 SemanticChunker 所需接口。"""
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """兼容 SemanticChunker 所需接口。"""
        result = self.embed([text])
        return result[0] if result else []
