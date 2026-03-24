"""
Layer 5.5: 重排序（Reranker）
=============================
使用 CrossEncoder 模型对检索结果进行重排序。
输入：(query, 候选文本列表)
输出：按相关性分数排序的重排序结果

Usage:
    from core.reranker import RerankerClient
    
    reranker = RerankerClient()
    query = "什么是 RAG"
    candidates = [{"text": "文本1", ...}, {"text": "文本2", ...}]
    reranked = reranker.rerank(query, candidates, top_k=3)
"""

import logging
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    logging.warning("sentence_transformers 未安装，请执行: pip install sentence_transformers")

from config.settings import settings

logger = logging.getLogger(__name__)


class RerankerClient:
    """CrossEncoder 重排序客户端。
    
    将 Milvus 搜索的 Top-K 结果通过 CrossEncoder 重排序，
    保留 top_k 个最相关的结果。
    
    模型：sentence-transformers/bge-reranker-v2-m3（与 BGE-Reranker 等价）
    """

    def __init__(
        self,
        model: str = settings.reranker_model,
        batch_size: int = settings.reranker_batch_size,
        use_fp16: bool = settings.reranker_use_fp16,
    ):
        """初始化 Reranker。
        
        Args:
            model: 模型名称（默认：sentence-transformers/bge-reranker-v2-m3）
            batch_size: 批处理大小
            use_fp16: 是否使用 FP16（速度更快，精度略低）
        """
        if CrossEncoder is None:
            raise ImportError(
                "sentence_transformers 未安装。请执行: pip install sentence_transformers"
            )

        self.model = model
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        logger.info(f"🔄 加载 CrossEncoder 模型: {model}")
        # CrossEncoder 自动选择最优设备（GPU 或 CPU）
        self._reranker = CrossEncoder(model)
        logger.info(f"✅ CrossEncoder 模型加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """对候选结果进行重排序。
        
        Args:
            query: 查询文本
            candidates: 候选文本列表，每项为 dict，必须包含 "text" 键
            top_k: 返回结果数（默认使用 settings.rerank_top_k）
        
        Returns:
            按相关性分数从高到低排序的结果列表，
            每项包含原始 dict 内容 + "rerank_score" 字段
        """
        if not candidates:
            return []

        k = top_k or settings.rerank_top_k

        # 准备 (query, text) 对进行重排序
        query_text_pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        logger.info(
            f"🔄 开始重排序: query='{query}', "
            f"候选数={len(candidates)}, top_k={k}"
        )

        # CrossEncoder 计算分数
        scores = self._reranker.predict(query_text_pairs)

        # 将分数附加到候选项
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # 按分数从高到低排序
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 返回 Top-K
        reranked_results = scored_candidates[:k]

        # 打印重排序结果
        self._print_rerank_results(reranked_results, query)

        return reranked_results

    @staticmethod
    def _print_rerank_results(results: list[dict], query: str) -> None:
        """打印重排序结果。"""
        print(f"\n{'='*60}")
        print(f"🔄 重排序结果 (query: {query})")
        print(f"{'='*60}")

        for i, result in enumerate(results):
            score = result.get("rerank_score", 0.0)
            preview = result.get("text", "")
            if len(preview) > 150:
                preview = preview[:150] + "..."

            rank = result.get("rank", i + 1)
            print(f"\n🏷️  #{i+1} (原rank#{rank})  |  score: {score:.4f}")
            print(f"   {preview}")

        print(f"\n{'='*60}")
