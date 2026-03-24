"""
LLM 生成客户端
===============
使用 OpenAI 兼容接口连接智谱 GLM，支持流式与非流式生成。
"""

from collections.abc import Generator

from openai import OpenAI

from config.settings import settings

# ── RAG 系统提示词 ──────────────────────────────

_SYSTEM_TEMPLATE = (
    "你是一个专业的文档问答助手。请基于以下检索到的文档片段回答用户的问题。\n"
    "如果检索到的内容不足以回答问题，请如实说明。\n"
    "请用中文回答，回答要准确、简洁。\n\n"
    "--- 检索到的文档片段 ---\n{context}\n--- 文档片段结束 ---"
)


def _build_context(chunks: list[dict]) -> str:
    """将检索结果列表拼接为上下文文本。"""
    parts: list[str] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        rank = chunk.get("rank", "?")
        score = chunk.get("score", 0)
        parts.append(f"[片段 {rank} | 相似度距离: {score:.4f}]\n{text}")
    return "\n\n".join(parts)


class LLMClient:
    """智谱 GLM 生成客户端（OpenAI 兼容模式）。"""

    def __init__(
        self,
        api_key: str = settings.llm_api_key,
        base_url: str = settings.llm_base_url,
        model: str = settings.llm_model,
        max_tokens: int = settings.llm_max_tokens,
        temperature: float = settings.llm_temperature,
    ):
        if not api_key:
            raise EnvironmentError(
                "未设置 LLM_API_KEY，请在 .env 文件中配置智谱 API Key"
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ── 公开方法 ────────────────────────────────

    def generate(self, query: str, context_chunks: list[dict], session_id: str | None = None) -> str:
        """非流式生成回答。

        Args:
            query:           用户查询。
            context_chunks:  检索结果列表，每项含 rank/score/text。
            session_id:      可选的会话标识，用于多轮对话上下文。

        Returns:
            LLM 生成的完整回答文本。
        """
        messages = self._build_messages(query, context_chunks, session_id)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
        )
        answer = response.choices[0].message.content or ""
        
        # 记录到会话历史
        if session_id:
            from core.memory import get_redis_history
            history = get_redis_history(session_id)
            history.add_user_message(query)
            history.add_ai_message(answer)
            
        return answer

    def generate_stream(
        self, query: str, context_chunks: list[dict], session_id: str | None = None
    ) -> Generator[str, None, None]:
        """流式生成回答，逐 token 返回。

        Args:
            query:           用户查询。
            context_chunks:  检索结果列表。
            session_id:      可选的会话标识。

        Yields:
            逐步生成的文本片段（delta content）。
        """
        messages = self._build_messages(query, context_chunks, session_id)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )

        answer_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                answer_parts.append(text)
                yield text
                
        # 记录到会话历史
        if session_id:
            from core.memory import get_redis_history
            history = get_redis_history(session_id)
            history.add_user_message(query)
            history.add_ai_message("".join(answer_parts))

    # ── 私有方法 ────────────────────────────────

    def _build_messages(
        self, query: str, context_chunks: list[dict], session_id: str | None = None
    ) -> list[dict]:
        """构建 ChatCompletion 的 messages 列表。"""
        context = _build_context(context_chunks)
        system_prompt = _SYSTEM_TEMPLATE.format(context=context)
        messages = [{"role": "system", "content": system_prompt}]
        
        if session_id:
            from core.memory import get_redis_history
            history = get_redis_history(session_id)
            
            # 使用滑动窗口策略截断历史记录（避免 Token 爆炸）
            # 一轮对话 = 1 个 user + 1 个 assistant，所以保留历史数为 max_turns * 2
            max_messages = settings.history_max_turns * 2
            recent_messages = history.messages[-max_messages:] if history.messages else []
            
            for msg in recent_messages:
                if msg.type == "human":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.type in ("ai", "assistant"):
                    messages.append({"role": "assistant", "content": msg.content})

        messages.append({"role": "user", "content": query})
        return messages
