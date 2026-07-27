from app.llm.base import BaseLLMAdapter, LLMResponse, LLMUsage


class StubAdapter(BaseLLMAdapter):
    """固定返回预设内容的适配器，不发起真实网络调用。用于测试 / CI / 演示。"""

    def __init__(self, fixed_sql: str = "SELECT 1 AS test"):
        self.fixed_sql = fixed_sql
        self.model = "stub"

    async def generate(self, messages: list[dict]) -> str:
        return self.fixed_sql

    async def generate_with_usage(self, messages: list[dict]) -> LLMResponse:
        return LLMResponse(
            content=self.fixed_sql,
            model=self.model,
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
