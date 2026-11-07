"""作业1：多轮 RAG 的问题改写。

多轮对话里用户会说省略句——"那电子书呢"、"它多久到账"。这种问题单独拿去做
向量检索几乎必然命中失败：句子里没有任何实义词可以匹配。

解决办法：在**检索之前**，用 LLM 结合对话历史把省略句补全成一个独立可检索的问题。
"""
from dataclasses import dataclass

from app.llm.factory import get_llm_adapter
from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """Given a conversation history and a follow-up question, rewrite the follow-up question into a standalone question that can be understood without the history.

Rules:
1. Resolve all pronouns and ellipsis using the conversation history ("那电子书呢" → "电子书的退款条件是什么").
2. Keep the user's original language.
3. If the question is ALREADY self-contained, return it completely unchanged.
4. Output ONLY the rewritten question — no explanation, no quotes, no prefix.

Conversation history:
{history}

Follow-up question:{question}"""


@dataclass
class RewriteResult:
    question: str          # 实际用于检索的问题
    original: str          # 用户原话
    rewritten: bool        # 是否真的改写了


def _format_history(history: list[dict], max_turns: int = 3) -> str:
    recent = history[-max_turns * 2:]
    return "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in recent
    )


async def rewrite_query(
    question: str,
    conversation_history: list[dict] | None = None,
) -> RewriteResult:
    """把依赖上下文的省略句改写成独立问题。失败时安全退回原问题。"""

    settings = get_settings()

    # 没有历史就没有可解析的指代，省一次 LLM 调用
    if not settings.rag_query_rewrite_enabled or not conversation_history:
        return RewriteResult(question=question, original=question, rewritten=False)

    prompt = REWRITE_PROMPT.format(
        history=_format_history(conversation_history),
        question=question,
    )

    try:
        adapter = get_llm_adapter()
        response = await adapter.generate([{"role": "user", "content": prompt}])
        new_question = response.strip().strip('"').strip("'")

        if not new_question:
            return RewriteResult(question=question, original=question, rewritten=False)

        changed = new_question != question.strip()
        if changed:
            logger.info(
                "rag.query_rewritten",
                original=question[:60],
                rewritten=new_question[:60],
            )

        return RewriteResult(
            question=new_question,
            original=question,
            rewritten=changed,
        )

    except Exception as e:
        # 改写只是增强手段，失败不该让整个问答挂掉
        logger.warning("rag.query_rewrite_failed", error=str(e))
        return RewriteResult(question=question, original=question, rewritten=False)
