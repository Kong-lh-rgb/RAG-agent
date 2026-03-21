"""
FastAPI 应用
============
提供 REST API 包装 RAG Pipeline：
  POST /upload  — 上传 PDF 文档，触发解析+入库
  POST /query   — 语义检索，返回匹配结果

启动:
    uvicorn api.app:app --reload --port 8000
"""

import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException

from api.schemas import QueryRequest, QueryResponse, SearchResult, UploadResponse
from pipeline import RAGPipeline

# ── 应用实例 ────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API",
    description="PDF 文档入库 + 向量语义检索",
    version="0.1.0",
)

# Pipeline 单例（复用连接和 client）
_pipeline = RAGPipeline()


# ── 路由 ────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, summary="上传文档并入库")
async def upload_document(file: UploadFile = File(..., description="PDF 文件")):
    """上传 PDF 文件，触发 解析 → 分块 → 嵌入 → Milvus 入库。"""
    # 1. 校验文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 2. 保存到临时文件
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 3. 执行入库
        chunk_count = _pipeline.ingest(tmp_path)

        return UploadResponse(
            filename=file.filename,
            chunk_count=chunk_count,
            message=f"文档 '{file.filename}' 入库成功，共 {chunk_count} 个文本块",
        )
    finally:
        # 4. 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse, summary="语义检索")
async def query_documents(req: QueryRequest):
    """对已入库的文档执行语义检索，返回 Top-K 匹配结果。"""
    try:
        results = _pipeline.query(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"检索失败: {str(e)}。请确认已通过 /upload 接口入库文档。",
        )

    return QueryResponse(
        query=req.query,
        results=[
            SearchResult(rank=r["rank"], score=r["score"], text=r["text"])
            for r in results
        ],
    )
