"""
Layer 2: 文本分块
=================
将长文本按固定大小切分为语义片段，支持块间重叠。
"""

from config.settings import settings


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """将文本按固定字符数分块。

    Args:
        text:       待分块的原始文本。
        chunk_size: 每块最大字符数。
        overlap:    相邻块重叠字符数。

    Returns:
        分块后的文本列表。

    Raises:
        ValueError: 输入文本为空时抛出。
    """
    if not text.strip():
        raise ValueError("输入文本为空，无法分块")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    print(f"✂️  文本分块完成: {len(chunks)} 个块 (大小={chunk_size}, 重叠={overlap})")
    return chunks
