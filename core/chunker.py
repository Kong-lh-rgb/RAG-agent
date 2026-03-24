"""
Layer 2: 文本分块
=================
父子分块策略：
1) 使用 RecursiveCharacterTextSplitter 先切父块
2) 对每个父块使用 SemanticChunker 切语义子块
"""

from config.settings import settings
from core.embedder import EmbeddingClient
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_parent_chunks(
    text: str,
    chunk_size: int = settings.parent_chunk_size,
    overlap: int = settings.parent_chunk_overlap,
) -> list[str]:
    """使用 RecursiveCharacterTextSplitter 切父块。"""
    if not text.strip():
        raise ValueError("输入文本为空，无法分块")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = [c.strip() for c in splitter.split_text(text) if c and c.strip()]
    print(f"📦 父块分块完成: {len(chunks)} 个块 (大小={chunk_size}, 重叠={overlap})")
    return chunks


def split_child_chunks_semantic(
    parent_chunks: list[str],
    embedding_client: EmbeddingClient,
    breakpoint_threshold_type: str = settings.semantic_breakpoint_threshold_type,
    breakpoint_threshold_amount: int = settings.semantic_breakpoint_threshold_amount,
) -> tuple[list[str], list[int], list[int]]:
    """对每个父块进行语义子块切分并返回映射关系。

    Args:
        parent_chunks: 父块文本列表。
        embedding_client: 语义分块用嵌入客户端。
        breakpoint_threshold_type: 语义断点阈值类型。
        breakpoint_threshold_amount: 语义断点阈值。

    Returns:
        (all_child_chunks, all_chunk_indices, all_parent_indices)
    """
    if not parent_chunks:
        return [], [], []

    semantic_splitter = SemanticChunker(
        embeddings=embedding_client,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    all_child_chunks: list[str] = []
    all_chunk_indices: list[int] = []
    all_parent_indices: list[int] = []

    global_child_idx = 0
    for parent_idx, parent_text in enumerate(parent_chunks):
        children = [
            c.strip() for c in semantic_splitter.split_text(parent_text)
            if c and c.strip()
        ]
        for child_text in children:
            all_child_chunks.append(child_text)
            all_chunk_indices.append(global_child_idx)
            all_parent_indices.append(parent_idx)
            global_child_idx += 1

    print(f"✂️  语义子块分块完成: {len(all_child_chunks)} 个子块")
    return all_child_chunks, all_chunk_indices, all_parent_indices
