import json
import time

from app.llm.factory import get_llm_adapter
from app.rag.retriever import RetrievalResult
from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 只对前 N 个初检结果做 re-ranking，控制成本和延迟
RERANK_CANDIDATE_LIMIT = 10

RERANK_PROMPT = """You are a relevance judge. Given a user's question and a list of text passages, rate how relevant each passage is to answering the question.

For each passage, output a relevance score from 0.0 (completely irrelevant) to 1.0 (directly answers the question).

User's question:{question}

Passages:
{passages}

Output ONLY a JSON array of scores in the same order as the passages, nothing else.
Example: [0.9, 0.3, 0.7, 0.1, 0.8]"""

# 作业4：Cross-encoder 模型只加载一次（加载很慢，几百 MB 权重）
_cross_encoder = None


async def rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
    method: str | None = None,
) -> list[RetrievalResult]:
    """对检索结果做二次排序。method: llm / cross_encoder（默认读配置）。"""

    if len(results) <= 1:
        return results

    method = method or get_settings().rerank_method

    if method == "cross_encoder":
        return await rerank_cross_encoder(question, results, top_k)
    return await rerank_llm(question, results, top_k)


async def rerank_llm(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """用 LLM 对检索结果做二次排序。"""

    if len(results) <= 1:
        return results

    candidates = results[:RERANK_CANDIDATE_LIMIT]

    passages = "\n\n".join(
        f"[Passage{i + 1}]:{r.content[:300]}"
        for i, r in enumerate(candidates)
    )

    prompt = RERANK_PROMPT.format(question=question, passages=passages)

    started = time.perf_counter()
    try:
        adapter = get_llm_adapter()
        response = await adapter.generate([{"role": "user", "content": prompt}])

        scores = json.loads(response.strip())

        if len(scores) != len(candidates):
            logger.warning(
                "reranker.score_mismatch",
                expected=len(candidates),
                got=len(scores),
            )
            return results[:top_k]

        for result, score in zip(candidates, scores):
            result.score = float(score)

        reranked = sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]

        logger.info(
            "reranker.completed",
            method="llm",
            candidates=len(candidates),
            top_score=reranked[0].score if reranked else 0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return reranked

    except Exception as e:
        logger.warning("reranker.failed", method="llm", error=str(e))
        return results[:top_k]  # 降级：不 re-rank，直接取前 K


def _get_cross_encoder():
    """作业4：懒加载 Cross-encoder（未安装 sentence-transformers 时返回 None）。"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "reranker.cross_encoder_unavailable",
            hint="poetry add sentence-transformers",
        )
        return None

    model_name = get_settings().cross_encoder_model
    logger.info("reranker.cross_encoder_loading", model=model_name)
    _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


async def rerank_cross_encoder(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """作业4：用 Cross-encoder（BAAI/bge-reranker-base）做 Re-ranking。

    和 LLM re-ranking 的区别：Cross-encoder 把 (question, passage) 作为一对同时
    输入模型，直接输出相关度分数——不走生成，所以更快更稳，也不会返回坏 JSON。
    """

    if len(results) <= 1:
        return results

    model = _get_cross_encoder()
    if model is None:
        # 没装 sentence-transformers → 降级到 LLM re-ranking
        return await rerank_llm(question, results, top_k)

    candidates = results[:RERANK_CANDIDATE_LIMIT]
    pairs = [(question, r.content[:512]) for r in candidates]

    started = time.perf_counter()
    try:
        import asyncio

        # predict 是同步且 CPU 密集的，丢到线程池避免阻塞事件循环
        scores = await asyncio.to_thread(model.predict, pairs)

        for result, score in zip(candidates, scores):
            result.score = float(score)

        reranked = sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]

        logger.info(
            "reranker.completed",
            method="cross_encoder",
            candidates=len(candidates),
            top_score=reranked[0].score if reranked else 0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        return reranked

    except Exception as e:
        logger.warning("reranker.failed", method="cross_encoder", error=str(e))
        return results[:top_k]
