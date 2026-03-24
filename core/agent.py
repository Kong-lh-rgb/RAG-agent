import json
import asyncio
from typing import AsyncGenerator

from openai import OpenAI

from config.settings import settings
from pipeline import RAGPipeline


def execute_kb_search(pipeline: RAGPipeline, query: str, doc_ids: list[str]) -> str:
    """内部执行知识库搜索"""
    results = pipeline.query(query, top_k=3, doc_ids=doc_ids)
    parent_chunks = pipeline._fetch_parent_chunks(results)
    
    parts = []
    for chunk in parent_chunks:
        parts.append(f"[文档片段 | doc_id={chunk['doc_id']}]\n{chunk['text']}")
    if not parts:
        return "❌ 本地知识库未查找到相关信息。"
    return "\n\n".join(parts)


def execute_web_search(query: str) -> str:
    """内部执行网络搜索"""
    if not settings.brave_search_api_key:
        return "❌ 网络搜索未配置 API Key。"
    try:
        import requests
        headers = {"Accept": "application/json", "X-Subscription-Token": settings.brave_search_api_key}
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 3},
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        parts = []
        for r in results:
            parts.append(f"标题: {r.get('title')}\n链接: {r.get('url')}\n摘要: {r.get('description')}")
        if not parts:
            return "❌ 没有找到相关的网络结果。"
        return "\n\n".join(parts)
    except Exception as e:
        return f"❌ 网络搜索失败: {str(e)}"


async def build_agent_stream(
    pipeline: RAGPipeline,
    query: str, 
    session_id: str, 
    doc_ids: list[str] = None
) -> AsyncGenerator[str, None]:
    """构建前端挂载的 SSE 异步计算流，使用 OpenAI 原生 Tool Calling。"""
    
    # 1. 准备历史和工具
    from core.memory import get_redis_history
    history = get_redis_history(session_id)
    doc_ids = doc_ids or []
    
    tools = [{
        "type": "function",
        "function": {
            "name": "knowledge_base_search",
            "description": "搜索企业级本地知识库。如果用户问题涉及特定的上传文档、规范、内部资料等，请优先使用本工具。必须提取查询关键词传入。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "提取自用户输入的自然语言查询关键词"}},
                "required": ["query"]
            }
        }
    }]
    
    if settings.brave_search_api_key:
        tools.append({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "互联网大搜工具。对于最新的新闻、外部世界知识、实时天气等信息，请优先使用本工具搜索。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "用于在搜索引擎中搜索的关键词"}},
                    "required": ["query"]
                }
            }
        })
        
    system_prompt = (
        "你是一个极其智能的企业级智能体助手。\n"
        "你有获取信息的强力手段：搜索本地知识库 或 搜索互联网。\n"
        "请根据用户的问题，智慧地调用合适的工具来查阅事实并解答。如果本地知识库有答案，优先以本地为准。\n"
        "最后，请用友好、专业的中文对用户进行综合回答。如果回答引用了文件或网页，适当注明来源。"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 截取历史保留最近的几轮
    max_history = settings.history_max_turns * 2
    recent = history.messages[-max_history:] if history.messages else []
    for msg in recent:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})
        
    context_str = f"用户当前查询范围限定。如果使用 knowledge_base_search，将仅在这些文档中检索：{doc_ids}。\n\n" if doc_ids else ""
    messages.append({"role": "user", "content": context_str + query})
    
    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    
    try:
        # 外层循环用于处理最多5次工具调用
        for _ in range(5):
            # 发起请求
            # Zhipu 支持流式时返回 tool_calls，我们通过 stream=True 手动解析累积
            # 或者为了稳定性先 stream=False 获取 tool_calls
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=tools,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                stream=False
            )
            
            msg = resp.choices[0].message
            
            if msg.tool_calls:
                # 必须转换为字典以兼容后续请求
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in msg.tool_calls
                    ]
                }
                messages.append(assistant_msg)
                
                # 逐个执行工具
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    args_str = tc.function.arguments
                    try:
                        args = json.loads(args_str)
                    except:
                        args = {"query": query}
                    
                    q = args.get("query", "")
                    
                    # 触发前端思考中事件
                    yield f"event: thought\ndata: {json.dumps({'action': 'start', 'tool': tool_name, 'input': q}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.05)
                    
                    if tool_name == "knowledge_base_search":
                        tool_res = execute_kb_search(pipeline, q, doc_ids)
                    elif tool_name == "web_search":
                        tool_res = execute_web_search(q)
                    else:
                        tool_res = f"Unknown tool {tool_name}"
                        
                    # 触发前端思考完成事件
                    yield f"event: thought\ndata: {json.dumps({'action': 'finish', 'tool': tool_name}, ensure_ascii=False)}\n\n"
                    
                    # 添加 tool 结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_res
                    })
                # 循环继续
            else:
                # Agent 不再使用工具，说明直接回答
                break
                
        # 退出工具循环后，流式请求最终结果
        final_stream = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True
        )
        
        final_answer = ""
        for chunk in final_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                final_answer += text
                yield f"event: token\ndata: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
                
        # 记录到数据库
        history.add_user_message(query)
        history.add_ai_message(final_answer)
        
        yield "event: done\ndata: {}\n\n"
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'traceback': tb}, ensure_ascii=False)}\n\n"
