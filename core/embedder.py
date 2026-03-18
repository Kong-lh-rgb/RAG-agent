"""
Layer 3: 嵌入生成
=================
封装 OpenAI Embedding API，提供批量文本嵌入能力。
"""

from openai import OpenAI

from config.settings import settings


class EmbeddingClient:
    """OpenAI Embedding 客户端。

    管理 API client 生命周期，并提供分批嵌入功能。
    """

    def __init__(
        self,
        api_key: str = settings.openai_api_key,
        base_url: str = settings.openai_base_url,
        model: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
    ):
        if not api_key:
            raise EnvironmentError("未设置 OPENAI_API_KEY，请在 .env 文件中配置")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本嵌入向量。

        Args:
            texts: 待嵌入的文本列表。

        Returns:
            与输入同序的嵌入向量列表。
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"🧠 嵌入生成: {min(i + self.batch_size, len(texts))}/{len(texts)}")

        print(f"🧠 嵌入生成完成: {len(all_embeddings)} 条, 维度={len(all_embeddings[0])}")
        return all_embeddings
