"""
PostgreSQL 数据库管理
=====================
SQLAlchemy Engine + Session 工厂。
"""

from contextlib import contextmanager
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

_engine = None
_SessionLocal = None


def get_engine():
    """获取或创建全局 Engine 单例。"""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.pg_dsn, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker:
    """获取 Session 工厂。"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """获取数据库 Session（上下文管理器，自动提交/回滚）。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
