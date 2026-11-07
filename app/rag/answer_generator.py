from app.llm.factory import get_llm_adapter
from app.llm.base import LLMResponse
from app.core.logging import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a document-based question answering assistant.

STRICT RULES:
1. Answer ONLY based on the provided sources below. Do NOT use any knowledge outside these sources.
2. For every factual statement, add a citation [n] at the end of the sentence, where n is the source number.
3. If the sources do NOT contain enough information to answer the question, output ONLY the word: NO_ANSWER
   - Do NOT say "I don't know" or "The documents don't mention..." — just output NO_ANSWER
   - Do NOT provide a partial answer or guess
4. You may cite multiple sources for one statement: [1, 3]
5. Keep your answer concise and directly relevant to the question.
6. Answer in the same language as the user's question.

SOURCES:
{sources}"""


async def generate_answer(
    question: str,
    sources_text: str,
    conversation_history: list[dict] | None = None,
) -> LLMResponse:
    """基于检索到的来源生成 Grounded 回答。"""

    system_content = RAG_SYSTEM_PROMPT.format(sources=sources_text)

    messages = [{"role": "system", "content": system_content}]

    # 多轮历史（如果有）
    if conversation_history:
        messages.extend(conversation_history[-6:])

    messages.append({"role": "user", "content": question})

    adapter = get_llm_adapter()
    response = await adapter.generate_with_usage(messages)

    logger.info(
        "rag.answer.generated",
        question=question[:50],
        answer_length=len(response.content),
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
    )

    return response