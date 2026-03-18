"""
Layer 1: PDF 解析
=================
使用 PyPDF2 从 PDF 文件中提取纯文本。
"""

import os

from PyPDF2 import PdfReader


def parse_pdf(pdf_path: str) -> str:
    """从 PDF 提取全部文本内容。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        合并后的全文文本。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n".join(pages_text)
    print(f"📄 PDF 解析完成: {len(reader.pages)} 页, {len(full_text):,} 字符")
    return full_text
