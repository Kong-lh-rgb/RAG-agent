import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def test():
    llm = ChatOpenAI(
        model="qwen-flash-character-2026-02-26",
        api_key="sk-1ec01ce229c8406a94471b503b7628bc", # user's judge api key from previous env
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        res = await llm.ainvoke([HumanMessage(content="hello")])
        print(res)
    except Exception as e:
        print("Error:", type(e), e)
        print("Switching model...")
        llm.model_name = "qwen3.5-397b-a17b"
        res = await llm.ainvoke([HumanMessage(content="hello")])
        print("Success with new model:", res.content)

asyncio.run(test())
