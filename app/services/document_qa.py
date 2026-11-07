from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.document import Document
from app.models.conversation import Conversation, Message
from app.rag.retriever import hybrid_search, RetrievalResult
from app.rag.reranker import rerank
from app.rag.context_builder import reorder_for_attention, build_context, SourceRef
from app.rag.answer_generator import generate_answer
from app.rag.citation import (
    Citation,
    verify_citations_auto,
    group_citations_by_document,
)
from app.rag.query_rewriter import rewrite_query
from app.rag.confidence import compute_confidence
from app.services.chat import get_conversation, _build_conversation_history
from app.config.settings import get_settings
from app.core.errors import NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGAnswer:
    """文档问答的完整结果。"""
    answer: str
    answerable: bool
    citations: list[Citation]
    confidence: float
    confidence_level: str
    sources: list[SourceRef]
    model: str
    usage: dict
    degraded: list[str] = field(default_factory=list)
    query_used: str = ""          # 作业1：实际用于检索的问题
    query_rewritten: bool = False  # 作业1：是否被改写过


async def answer_from_documents(
    db: AsyncSession,
    user: User,
    question: str,
    document_ids: list[int] | None = None,
    conversation_history: list[dict] | None = None,
) -> RAGAnswer:
    """纯计算版文档问答——不落库，供 Query Router 复用。"""

    settings = get_settings()
    degraded: list[str] = []

    # ✅ 0. 作业1：问题改写（必须在检索之前——"那电子书呢" 直接拿去检索必然命中失败）
    rewrite = await rewrite_query(question, conversation_history)
    search_question = rewrite.question

    # 1. 检索（用改写后的问题）
    results = await hybrid_search(
        db, search_question, user.id,
        top_k=settings.rag_top_k * 2,
        document_ids=document_ids,
    )

    # 检索为空 → 不调 LLM
    if not results:
        logger.info("rag.no_results", question=question[:50])
        return RAGAnswer(
            answer="在你上传的文档中没有找到与该问题相关的内容。",
            answerable=False,
            citations=[],
            confidence=0.0,
            confidence_level="low",
            sources=[],
            model="",
            usage={},
            query_used=search_question,
            query_rewritten=rewrite.rewritten,
        )

    # 2. Re-ranking
    reranked = True
    try:
        results = await rerank(search_question, results, top_k=settings.rag_top_k)
    except Exception as e:
        logger.warning("rag.rerank_failed", error=str(e))
        results = results[:settings.rag_top_k]
        reranked = False
        degraded.append("rerank_skipped")

    # 3. Lost in the Middle 重排
    results = reorder_for_attention(results)

    # 4. 构建 Context
    context_text, sources = build_context(results, settings.rag_context_max_tokens)

    # 5. 生成回答
    response = await generate_answer(
        question=search_question,
        sources_text=context_text,
        conversation_history=conversation_history,
    )

    answer = response.content.strip()

    # 6. 检测 NO_ANSWER
    if answer == "NO_ANSWER":
        return RAGAnswer(
            answer="根据你上传的文档，没有找到与该问题直接相关的信息。",
            answerable=False,
            citations=[],
            confidence=0.0,
            confidence_level="low",
            sources=sources,
            model=response.model,
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens},
            degraded=degraded,
            query_used=search_question,
            query_rewritten=rewrite.rewritten,
        )

    # 7. 引用验证
    check = await verify_citations_auto(
        answer, sources,
        support_threshold=settings.rag_citation_support_threshold,
    )

    # 8. 置信度计算
    confidence, level = compute_confidence(sources, check, reranked=reranked)

    # 9. 低置信度免责声明
    final_answer = check.cleaned_answer
    if level == "low":
        final_answer = "⚠️ 以下内容与你的问题相关度较低，仅供参考。\n\n" + final_answer
        degraded.append("low_relevance")

    logger.info(
        "rag.answer.completed",
        question=question[:50],
        answerable=True,
        confidence=confidence,
        level=level,
        citations=len(check.citations),
    )

    return RAGAnswer(
        answer=final_answer,
        answerable=True,
        citations=check.citations,
        confidence=confidence,
        confidence_level=level,
        sources=sources,
        model=response.model,
        usage={"prompt_tokens": response.usage.prompt_tokens,
               "completion_tokens": response.usage.completion_tokens},
        degraded=degraded,
        query_used=search_question,
        query_rewritten=rewrite.rewritten,
    )


async def ask_documents(
    db: AsyncSession,
    user: User,
    conversation_id: int,
    question: str,
    document_ids: list[int] | None = None,
) -> dict:
    """落库版文档问答——存 Message + meta。"""

    # 获取对话 + 历史
    conv = await get_conversation(db, user, conversation_id)
    history = await _build_conversation_history(db, conv.id)

    # 存用户消息
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()

    # 调用纯计算版
    result = await answer_from_documents(
        db, user, question,
        document_ids=document_ids,
        conversation_history=history,
    )

    # 存 assistant 消息（带 meta）
    citation_dicts = [
        {
            "index": c.index,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "section_title": c.section_title,
            "page_number": c.page_number,
            "snippet": c.snippet,
            "supported": c.supported,
            "support_score": c.support_score,
        }
        for c in result.citations
    ]

    # 作业4：按文档聚合引用（正文里的 [n] 编号保持不变）
    citation_groups = group_citations_by_document(result.citations)

    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=result.answer,
        meta={
            "type": "rag",
            "query_used": result.query_used,
            "query_rewritten": result.query_rewritten,
            "citation_groups": citation_groups,
            "answerable": result.answerable,
            "citations": citation_dicts,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level,
            "degraded": result.degraded,
        },
    )
    db.add(assistant_msg)
    await db.flush()

    return {
        "message_id": assistant_msg.id,
        "content": result.answer,
        "answerable": result.answerable,
        "citations": citation_dicts,
        "citation_groups": citation_groups,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level,
        "degraded": result.degraded,
        "query_used": result.query_used,
        "query_rewritten": result.query_rewritten,
        "model": result.model,
        "usage": result.usage,
    }