"""
评测数据集
===========
从 JSON 加载评测集，或通过 Pipeline 自动生成。

评测集 JSON 格式:
[
  {"question": "...", "ground_truth": "..."},
  ...
]
"""

import json
from dataclasses import dataclass

from pipeline import RAGPipeline


@dataclass
class EvalSample:
    """单条评测样本。"""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] | None = None


def load_dataset_from_json(path: str) -> list[EvalSample]:
    """从 JSON 文件加载评测问答对。

    Args:
        path: JSON 文件路径，格式为 [{"question": "...", "ground_truth": "..."}, ...]

    Returns:
        EvalSample 列表（answer 和 contexts 待填充）。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
        ))

    print(f"📂 已加载 {len(samples)} 条评测数据")
    return samples


def build_dataset_from_pipeline(
    pipeline: RAGPipeline,
    samples: list[EvalSample],
    top_k: int = 3,
    doc_id: str | None = None,
) -> list[EvalSample]:
    """使用 Pipeline 运行评测问题，填充 answer 和 contexts。

    Args:
        pipeline: RAG Pipeline 实例。
        samples:  评测样本列表。
        top_k:    检索结果数。
        doc_id:   可选，仅在指定文档中检索。

    Returns:
        填充完 answer 和 contexts 的样本列表。
    """
    print(f"\n{'='*60}")
    print(f"🧪 开始生成评测数据 ({len(samples)} 条)")
    print(f"{'='*60}")

    for i, sample in enumerate(samples):
        print(f"\n--- [{i+1}/{len(samples)}] {sample.question}")

        # 使用非流式生成获取完整回答
        result = pipeline.query_and_generate(
            sample.question, top_k=top_k, doc_id=doc_id
        )

        sample.answer = result["answer"]
        # contexts 取父块文本（已经由 pipeline 从 PG 获取）
        # result["results"] 是子块检索结果，取其文本作为 contexts
        sample.contexts = [r["text"] for r in result["results"]]

        print(f"   ✅ 回答长度: {len(sample.answer)} 字, 上下文: {len(sample.contexts)} 段")

    print(f"\n{'='*60}")
    print(f"🧪 评测数据生成完成")
    print(f"{'='*60}")
    return samples
