"""
ORM 模型
=========
PostgreSQL 表结构定义：documents（文档元数据）、parent_chunks（父块文本）。
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """ORM 基类。"""
    pass


class Document(Base):
    """文档元数据表。"""

    __tablename__ = "documents"

    doc_id = Column(String(64), primary_key=True, comment="UUID 文档标识")
    filename = Column(String(512), nullable=False, comment="原始文件名")
    chunk_count = Column(Integer, nullable=False, default=0, comment="子块数量")
    parent_chunk_count = Column(Integer, nullable=False, default=0, comment="父块数量")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="入库时间",
    )

    # 关系
    parent_chunks = relationship(
        "ParentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document doc_id={self.doc_id!r} filename={self.filename!r}>"


class ParentChunk(Base):
    """父块文本表。"""

    __tablename__ = "parent_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(
        String(64),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        comment="关联文档 ID",
    )
    parent_index = Column(Integer, nullable=False, comment="父块在文档内的序号")
    text = Column(Text, nullable=False, comment="父块完整文本")

    # 联合唯一约束：同一文档内 parent_index 不重复
    __table_args__ = (
        UniqueConstraint("doc_id", "parent_index", name="uq_doc_parent_index"),
    )

    # 关系
    document = relationship("Document", back_populates="parent_chunks")

    def __repr__(self) -> str:
        return f"<ParentChunk doc_id={self.doc_id!r} parent_index={self.parent_index}>"
