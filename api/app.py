"""
FastAPI 应用
============
提供 REST API 包装 RAG Pipeline：
  POST /upload        — 上传 PDF 文档，触发解析+入库
  POST /query         — 语义检索，返回匹配结果（JSON）
  POST /query/stream  — 语义检索 + LLM 生成，SSE 流式输出

启动:
    uvicorn api.app:app --reload --port 8000
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse, SearchResult, UploadResponse
from core.db_init import init_db
from pipeline import RAGPipeline


# ── 生命周期事件 ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 PostgreSQL 表。"""
    init_db()
    yield


# ── 应用实例 ────────────────────────────────────

app = FastAPI(
    title="RAG Pipeline API",
    description="PDF 文档入库 + 向量语义检索 + LLM 流式回答（企业级知识库）",
    version="0.3.0",
    lifespan=lifespan,
)

# Pipeline 单例（复用连接和 client）
_pipeline = RAGPipeline()


# ── 路由 ────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, summary="上传文档并入库")
async def upload_document(file: UploadFile = File(..., description="PDF 文件")):
    """上传 PDF 文件，触发 解析 → 父子分块 → 嵌入 → Milvus + PG 入库。"""
    # 1. 校验文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 2. 保存到临时文件
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 3. 执行入库（累积入库，不删除旧数据）
        result = _pipeline.ingest(tmp_path, filename=file.filename)

        return UploadResponse(
            doc_id=result["doc_id"],
            filename=file.filename,
            chunk_count=result["chunk_count"],
            parent_chunk_count=result["parent_chunk_count"],
            message=f"文档 '{file.filename}' 入库成功，共 {result['chunk_count']} 个子块",
        )
    finally:
        # 4. 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse, summary="语义检索")
async def query_documents(req: QueryRequest):
    """对已入库的文档执行语义检索，返回 Top-K 匹配结果。支持 doc_id 过滤。"""
    try:
        results = _pipeline.query(req.query, top_k=req.top_k, doc_id=req.doc_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"检索失败: {str(e)}。请确认已通过 /upload 接口入库文档。",
        )

    return QueryResponse(
        query=req.query,
        results=[
            SearchResult(
                rank=r["rank"],
                score=r["score"],
                text=r["text"],
                doc_id=r["doc_id"],
                chunk_index=r["chunk_index"],
                parent_index=r["parent_index"],
            )
            for r in results
        ],
    )


@app.post("/query/stream", summary="语义检索 + LLM 流式回答 (SSE)")
async def query_stream(req: QueryRequest):
    """检索文档 + 父块文本替换 + LLM 流式生成回答。

    返回 SSE (Server-Sent Events) 事件流：
      - event: context  — 检索到的子块信息
      - event: token    — LLM 逐步生成的文本 token
      - event: done     — 生成完毕
    """
    try:
        event_stream = _pipeline.query_and_generate_stream(
            req.query, top_k=req.top_k, doc_id=req.doc_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"流式检索失败: {str(e)}",
        )

    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
