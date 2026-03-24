"""
会话记忆管理
==========
基于 Redis 存储多轮对话上下文。
"""

from langchain_community.chat_message_histories import RedisChatMessageHistory
from config.settings import settings

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    """获取指定 session_id 的 Redis 聊天历史。
    
    Args:
        session_id: 唯一会话标识。
        
    Returns:
        RedisChatMessageHistory 实例。
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url
    )
