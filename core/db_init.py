"""
数据库初始化
=============
自动创建 PostgreSQL 表结构。
"""

from core.database import get_engine
from core.models import Base


def init_db() -> None:
    """创建所有 ORM 表（已存在则跳过）。"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("🗄️  PostgreSQL 表初始化完成")
