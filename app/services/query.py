from app.core.errors import ValidationError, LLMError

def handle_query(query: str):
    if not query.strip():
        raise ValidationError("查询内容不能为空")

    # try:
    #     result = call_llm(query)
    # except TimeoutError:
    #     raise LLMError("LLM 响应超时，请稍后重试")

    # return result
    return None