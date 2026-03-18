"""
RAG Pipeline CLI
================
Usage:
    python main.py ingest --pdf data/sample.pdf
    python main.py query  --query "什么是RAG"
    python main.py query  --query "向量数据库" --top-k 5
"""

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
    pipeline.ingest(args.pdf, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print("🎉 入库完成!")


def cmd_query(args):
    """子命令: 语义检索。"""
    print("🚀 语义检索启动")
    print(f"   Query:  {args.query}")
    print(f"   Milvus: {settings.milvus_host}:{settings.milvus_port}")
    print()

    pipeline = RAGPipeline()
    pipeline.query(args.query, top_k=args.top_k)
    print("🎉 检索完成!")


def main():
    parser = argparse.ArgumentParser(
        description="Simple RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python main.py ingest --pdf data/sample.pdf\n"
               "  python main.py query  --query '什么是RAG'\n",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # ── ingest 子命令 ──────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="将 PDF 文档解析并入库到 Milvus")
    p_ingest.add_argument("--pdf", required=True, help="PDF 文件路径")
    p_ingest.add_argument("--chunk-size", type=int, default=None,
                          help=f"分块大小 (默认 {settings.chunk_size})")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None,
                          help=f"重叠大小 (默认 {settings.chunk_overlap})")
    p_ingest.set_defaults(func=cmd_ingest)

    # ── query 子命令 ───────────────────────────
    p_query = subparsers.add_parser("query", help="语义检索已入库的文档")
    p_query.add_argument("--query", required=True, help="查询文本")
    p_query.add_argument("--top-k", type=int, default=None,
                         help=f"返回结果数 (默认 {settings.top_k})")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    t0 = time.time()
    args.func(args)
    print(f"⏱️  总耗时: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
