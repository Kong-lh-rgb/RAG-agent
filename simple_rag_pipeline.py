"""
Simple RAG Pipeline
===================
PDF → PyPDF2 解析 → 固定分块 → OpenAI 嵌入 → Milvus 存储 → 向量检索 → 打印结果

Usage:
    python simple_rag_pipeline.py --pdf sample.pdf --query "什么是RAG"
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_simple_demo"

CHUNK_SIZE = 500      # 每块字符数
CHUNK_OVERLAP = 50    # 块间重叠字符数
TOP_K = 3             # 检索返回数量


# ══════════════════════════════════════════════
# Layer 1: PDF 解析
# ══════════════════════════════════════════════

def parse_pdf(pdf_path: str) -> str:
    """使用 PyPDF2 从 PDF 中提取全部文本。"""
    from PyPDF2 import PdfReader

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    print(f"📄 PDF 解析完成: {len(reader.pages)} 页, {len(full_text):,} 字符")
    return full_text


# ══════════════════════════════════════════════
# Layer 2: 文本分块
# ══════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将文本按固定大小分块，支持重叠。"""
    if not text.strip():
        raise ValueError("输入文本为空，无法分块")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 跳过纯空白块
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    print(f"✂️  文本分块完成: {len(chunks)} 个块 (大小={chunk_size}, 重叠={overlap})")
    return chunks


# ══════════════════════════════════════════════
# Layer 3: 嵌入生成
# ══════════════════════════════════════════════

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """调用 OpenAI Embedding API 批量生成文本嵌入。"""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise EnvironmentError("未设置 OPENAI_API_KEY，请在 .env 文件中配置")

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # OpenAI embedding API 单次最多处理 2048 条，分批处理
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"🧠 嵌入生成: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    print(f"🧠 嵌入生成完成: {len(all_embeddings)} 条, 维度={len(all_embeddings[0])}")
    return all_embeddings


# ══════════════════════════════════════════════
# Layer 4: Milvus 存储
# ══════════════════════════════════════════════

def store_in_milvus(chunks: list[str], embeddings: list[list[float]]) -> None:
    """在 Milvus 中创建 Collection 并插入数据 + 建索引。"""
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )

    # 1. 连接 Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"🔗 已连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")

    # 2. 如果存在同名 Collection 则删除（脚本模式，每次重建）
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"🗑️  已删除旧 Collection: {COLLECTION_NAME}")

    # 3. 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="Simple RAG demo collection")

    # 4. 创建 Collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"📦 已创建 Collection: {COLLECTION_NAME}")

    # 5. 插入数据
    insert_data = [chunks, embeddings]
    result = collection.insert(insert_data)
    collection.flush()
    print(f"💾 已插入 {result.insert_count} 条记录")

    # 6. 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"🏗️  已创建索引: IVF_FLAT (L2)")

    # 7. 加载 Collection 到内存
    collection.load()
    print(f"✅ Collection 已加载到内存，可供检索")


# ══════════════════════════════════════════════
# Layer 5: 向量检索
# ══════════════════════════════════════════════

def search_milvus(query: str, top_k: int = TOP_K) -> list[dict]:
    """将查询文本嵌入后在 Milvus 中进行相似度搜索。"""
    from pymilvus import connections, Collection

    # 1. 生成查询嵌入
    print(f"\n🔍 查询: \"{query}\"")
    query_embedding = get_embeddings([query])

    # 2. 连接 & 加载 Collection
    if not connections.has_connection("default"):
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 3. 执行搜索
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"],
    )

    # 4. 整理并打印结果
    search_results = []
    print(f"\n{'='*60}")
    print(f"📋 Top-{top_k} 检索结果")
    print(f"{'='*60}")

    for i, hit in enumerate(results[0]):
        result = {
            "rank": i + 1,
            "score": hit.distance,
            "text": hit.entity.get("text"),
        }
        search_results.append(result)

        text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
        print(f"\n🏷️  #{i+1}  |  L2 距离: {hit.distance:.4f}")
        print(f"   {text_preview}")

    print(f"\n{'='*60}")
    return search_results


# ══════════════════════════════════════════════
# Main: 串联全部流程
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Pipeline Demo")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--query", required=True, help="检索查询文本")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"分块大小 (默认 {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"重叠大小 (默认 {CHUNK_OVERLAP})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"返回结果数 (默认 {TOP_K})")
    args = parser.parse_args()

    print("🚀 Simple RAG Pipeline 启动")
    print(f"   PDF:    {args.pdf}")
    print(f"   Query:  {args.query}")
    print(f"   Model:  {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print()

    t0 = time.time()

    # Step 1: PDF 解析
    text = parse_pdf(args.pdf)

    # Step 2: 文本分块
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)

    # Step 3: 嵌入生成
    embeddings = get_embeddings(chunks)

    # Step 4: Milvus 存储
    store_in_milvus(chunks, embeddings)

    # Step 5: 向量检索
    results = search_milvus(args.query, args.top_k)

    elapsed = time.time() - t0
    print(f"\n⏱️  总耗时: {elapsed:.2f}s")
    print("🎉 Pipeline 执行完毕!")


if __name__ == "__main__":
    main()
