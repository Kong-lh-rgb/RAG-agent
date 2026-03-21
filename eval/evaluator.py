"""
RAGAS 评测执行器
=================
封装 RAGAS 评测流程，使用智谱 GLM 作为评测 LLM。

Usage:
    from eval.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    results = evaluator.run(samples)
"""

import json
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from eval.dataset import EvalSample


class RAGEvaluator:
    """RAGAS 评测执行器。

    使用智谱 GLM 作为 judge LLM 对 RAG Pipeline 输出进行评测。
    """

    def __init__(self):
        # 评测用 LLM（智谱 GLM via OpenAI 兼容接口）
        self._llm = LangchainLLMWrapper(ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        ))
        # 评测用 Embedding（复用现有 OpenAI Embedding）
        self._embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ))

    def run(
        self,
        samples: list[EvalSample],
        save_path: str | None = None,
    ) -> dict:
        """执行 RAGAS 评测。

        Args:
            samples:   已填充 answer 和 contexts 的评测样本。
            save_path: 可选，保存评测结果的 JSON 文件路径。

        Returns:
            {"scores": {"faithfulness": 0.85, ...}, "details": [...]}
        """
        # 1. 构建 RAGAS EvaluationDataset
        ragas_samples = []
        for s in samples:
            ragas_samples.append(SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=s.contexts or [],
                reference=s.ground_truth,
            ))
        dataset = EvaluationDataset(samples=ragas_samples)

        # 2. 定义评测指标
        metrics = [
            Faithfulness(llm=self._llm),
            AnswerRelevancy(llm=self._llm, embeddings=self._embeddings),
            LLMContextPrecisionWithReference(llm=self._llm),
            LLMContextRecall(llm=self._llm),
        ]

        # 3. 执行评测
        print("\n🔬 开始 RAGAS 评测...")
        result = evaluate(dataset=dataset, metrics=metrics)

        # 4. 格式化结果
        scores = result.scores.to_dict()
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(samples),
            "scores": scores,
            "details": result.to_pandas().to_dict(orient="records"),
        }

        # 5. 打印概览
        self._print_summary(output)

        # 6. 保存结果
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 评测结果已保存到: {save_path}")

        return output

    @staticmethod
    def _print_summary(output: dict) -> None:
        """打印评测结果概览。"""
        print(f"\n{'='*60}")
        print(f"📊 RAGAS 评测结果 ({output['sample_count']} 条样本)")
        print(f"{'='*60}")

        scores = output["scores"]
        for metric, values in scores.items():
            # scores 通常是 {metric_name: [score1, score2, ...]} 或 {metric_name: avg}
            if isinstance(values, list):
                avg = sum(v for v in values if v is not None) / max(len([v for v in values if v is not None]), 1)
                print(f"  {metric:.<40s} {avg:.4f}")
            else:
                print(f"  {metric:.<40s} {values:.4f}")

        print(f"{'='*60}")
