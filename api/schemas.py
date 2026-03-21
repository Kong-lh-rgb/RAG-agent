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
    doc_id: str | None = Field(default=None, description="可选，仅检索指定文档")


# ── 响应模型 ────────────────────────────────────

class UploadResponse(BaseModel):
    """文档上传响应。"""
    doc_id: str = Field(..., description="文档唯一标识")
    filename: str = Field(..., description="上传的文件名")
    chunk_count: int = Field(..., description="入库的子块数")
    parent_chunk_count: int = Field(..., description="父块数")
    message: str = Field(default="文档入库成功")


class SearchResult(BaseModel):
    """单条检索结果。"""
    rank: int = Field(..., description="排名")
    score: float = Field(..., description="L2 距离分数")
    text: str = Field(..., description="匹配到的子块文本")
    doc_id: str = Field(..., description="所属文档 ID")
    chunk_index: int = Field(..., description="子块序号")
    parent_index: int = Field(..., description="对应父块序号")


class QueryResponse(BaseModel):
    """检索响应。"""
    query: str = Field(..., description="原始查询")
    results: list[SearchResult] = Field(..., description="检索结果列表")
