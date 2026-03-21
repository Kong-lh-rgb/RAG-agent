"""
RAG Pipeline CLI
================
Usage:
    python main.py ingest    --pdf data/sample.pdf
    python main.py query     --query "什么是RAG"
    python main.py evaluate  --dataset data/eval_sample.json
"""
// # 准备你的评测数据集 JSON（格式见 data/eval_sample.json）
// # 运行评测
"""python main.py evaluate --dataset data/eval_sample.json"""

// # 指定文档 + 自定义输出
"""python main.py evaluate --dataset data/eval_sample.json --doc-id your-doc-id --output results.json"""

import argparse
import time

from config.settings import settings
from pipeline import RAGPipeline


def cmd_ingest(args):
    """子命令: 文档入库。"""
    print("🚀 文档入库启动")
    print(f"   PDF:    {args.pdf}")
    print(f"   Model:  {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"   Milvus: {settings.milvus_host}:{settings.milvus_port}")
    print()

    pipeline = RAGPipeline()
    result = pipeline.ingest(args.pdf, filename=args.pdf)
    print(f"🎉 入库完成! doc_id={result['doc_id']}")


def cmd_query(args):
    """子命令: 语义检索。"""
    print("🚀 语义检索启动")
    print(f"   Query:  {args.query}")
    print(f"   Milvus: {settings.milvus_host}:{settings.milvus_port}")
    if args.doc_id:
        print(f"   Doc ID: {args.doc_id}")
    print()

    pipeline = RAGPipeline()
    pipeline.query(args.query, top_k=args.top_k, doc_id=args.doc_id)
    print("🎉 检索完成!")


def cmd_evaluate(args):
    """子命令: RAGAS 自动化评测。"""
    from eval.dataset import load_dataset_from_json, build_dataset_from_pipeline
    from eval.evaluator import RAGEvaluator

    print("🚀 RAGAS 评测启动")
    print(f"   数据集:  {args.dataset}")
    if args.doc_id:
        print(f"   Doc ID: {args.doc_id}")
    print(f"   Top-K:  {args.top_k}")
    print()

    # 1. 加载评测数据集
    samples = load_dataset_from_json(args.dataset)

    # 2. 调用 Pipeline 生成 answer + contexts
    pipeline = RAGPipeline()
    samples = build_dataset_from_pipeline(
        pipeline, samples, top_k=args.top_k, doc_id=args.doc_id
    )

    # 3. 运行 RAGAS 评测
    evaluator = RAGEvaluator()
    evaluator.run(samples, save_path=args.output)

    print("🎉 评测完成!")


def main():
    parser = argparse.ArgumentParser(
        description="Simple RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python main.py ingest    --pdf data/sample.pdf\n"
               "  python main.py query     --query '什么是RAG'\n"
               "  python main.py evaluate  --dataset data/eval_sample.json\n",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # ── ingest 子命令 ──────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="将 PDF 文档解析并入库到 Milvus")
    p_ingest.add_argument("--pdf", required=True, help="PDF 文件路径")
    p_ingest.add_argument("--chunk-size", type=int, default=None,
                          help=f"子块大小 (默认 {settings.chunk_size})")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None,
                          help=f"重叠大小 (默认 {settings.chunk_overlap})")
    p_ingest.set_defaults(func=cmd_ingest)

    # ── query 子命令 ───────────────────────────
    p_query = subparsers.add_parser("query", help="语义检索已入库的文档")
    p_query.add_argument("--query", required=True, help="查询文本")
    p_query.add_argument("--top-k", type=int, default=None,
                         help=f"返回结果数 (默认 {settings.top_k})")
    p_query.add_argument("--doc-id", default=None, help="可选，仅检索指定文档")
    p_query.set_defaults(func=cmd_query)

    # ── evaluate 子命令 ────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="使用 RAGAS 对 RAG Pipeline 进行评测")
    p_eval.add_argument("--dataset", required=True,
                        help="评测数据集 JSON 文件路径")
    p_eval.add_argument("--doc-id", default=None,
                        help="可选，仅在指定文档中检索")
    p_eval.add_argument("--top-k", type=int, default=3,
                        help="检索结果数 (默认 3)")
    p_eval.add_argument("--output", default="data/eval_results.json",
                        help="评测结果输出路径 (默认 data/eval_results.json)")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    t0 = time.time()
    args.func(args)
    print(f"⏱️  总耗时: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
