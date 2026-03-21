"""
请求/响应模型
=============
FastAPI 接口的 Pydantic 数据模型。
"""

from pydantic import BaseModel, Field


# ── 请求模型 ────────────────────────────────────

class QueryRequest(BaseModel):
    """检索请求。"""
    query: str = Field(..., description="查询文本", examples=["什么是RAG"])
    top_k: int = Field(default=3, ge=1, le=20, description="返回结果数")


# ── 响应模型 ────────────────────────────────────

class UploadResponse(BaseModel):
    """文档上传响应。"""
    filename: str = Field(..., description="上传的文件名")
    chunk_count: int = Field(..., description="入库的文本块数")
    message: str = Field(default="文档入库成功")


class SearchResult(BaseModel):
    """单条检索结果。"""
    rank: int = Field(..., description="排名")
    score: float = Field(..., description="L2 距离分数")
    text: str = Field(..., description="匹配到的文本片段")


class QueryResponse(BaseModel):
    """检索响应。"""
    query: str = Field(..., description="原始查询")
    results: list[SearchResult] = Field(..., description="检索结果列表")
