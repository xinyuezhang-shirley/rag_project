import hashlib
import json
import math

from app.db.redis import get_redis_client
from app.config.settings import get_settings
from app.llm.factory import get_llm_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

STATS_HIT_KEY = "sql_cache:stats:hit"
STATS_MISS_KEY = "sql_cache:stats:miss"
LOW_HIT_RATE_THRESHOLD = 0.2


def _cache_key(datasource_id: int, question: str) -> str:
    q_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"sql_cache:{datasource_id}:{q_hash}"


def _embedding_key(datasource_id: int, question: str) -> str:
    q_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"sql_cache_emb:{datasource_id}:{q_hash}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def get_cached_sql(datasource_id: int, question: str) -> str | None:
    """精确匹配查找缓存的 SQL（同一个问题的原文完全一致）。"""
    key = _cache_key(datasource_id, question)
    try:
        client = get_redis_client()
        cached = await client.get(key)
        if cached:
            logger.info("sql.cache.hit", key=key, question=question[:50])
            return cached
        logger.info("sql.cache.miss", key=key, question=question[:50])
        return None
    except Exception as e:
        logger.warning("sql.cache.get_failed", error=str(e))
        return None  # Redis 故障不阻断主链路


async def get_similar_cached_sql(
    datasource_id: int, question: str, threshold: float | None = None
) -> str | None:
    """作业1：语义相似度缓存。精确匹配未命中时，用 Embedding 余弦相似度在历史问题里找最相似的一个。"""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return None
    threshold = threshold if threshold is not None else settings.semantic_cache_similarity_threshold

    try:
        adapter = get_llm_adapter()
        query_embedding = await adapter.embed(question)
    except Exception as e:
        logger.warning("sql.cache.semantic_embed_failed", error=str(e))
        return None

    try:
        client = get_redis_client()
        pattern = f"sql_cache_emb:{datasource_id}:*"
        best_score = 0.0
        best_key = None

        async for key in client.scan_iter(match=pattern):
            raw = await client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            score = _cosine_similarity(query_embedding, data.get("embedding", []))
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= threshold:
            q_hash = best_key.rsplit(":", 1)[-1]
            sql = await client.get(f"sql_cache:{datasource_id}:{q_hash}")
            if sql:
                logger.info(
                    "sql.cache.semantic_hit",
                    score=round(best_score, 4),
                    question=question[:50],
                )
                return sql

        logger.info("sql.cache.semantic_miss", best_score=round(best_score, 4), question=question[:50])
        return None
    except Exception as e:
        logger.warning("sql.cache.semantic_search_failed", error=str(e))
        return None


async def get_cached_sql_with_semantic_fallback(datasource_id: int, question: str) -> str | None:
    """先精确匹配，未命中再试语义相似度匹配；命中/未命中都计入命中率统计（作业4）。"""
    sql = await get_cached_sql(datasource_id, question)
    if sql:
        await _record_cache_event(hit=True)
        return sql

    sql = await get_similar_cached_sql(datasource_id, question)
    if sql:
        await _record_cache_event(hit=True)
        return sql

    await _record_cache_event(hit=False)
    return None


async def set_cached_sql(datasource_id: int, question: str, sql: str) -> None:
    """缓存生成的 SQL，同时存一份 Embedding 供语义相似度匹配使用。"""
    settings = get_settings()
    key = _cache_key(datasource_id, question)
    try:
        client = get_redis_client()
        await client.set(key, sql, ex=settings.cache_ttl_seconds)
        logger.info("sql.cache.set", key=key, ttl=settings.cache_ttl_seconds)
    except Exception as e:
        logger.warning("sql.cache.set_failed", error=str(e))
        return

    if not settings.semantic_cache_enabled:
        return

    try:
        adapter = get_llm_adapter()
        embedding = await adapter.embed(question)
        emb_key = _embedding_key(datasource_id, question)
        client = get_redis_client()
        await client.set(
            emb_key,
            json.dumps({"question": question, "embedding": embedding}),
            ex=settings.cache_ttl_seconds,
        )
    except Exception as e:
        logger.warning("sql.cache.embed_store_failed", error=str(e))


async def invalidate_cache(datasource_id: int) -> None:
    """清空某个数据源的所有缓存（精确匹配 + 语义索引，Schema 变更时调用）。"""
    try:
        client = get_redis_client()
        keys = []
        for pattern in (f"sql_cache:{datasource_id}:*", f"sql_cache_emb:{datasource_id}:*"):
            keys += [key async for key in client.scan_iter(match=pattern)]
        if keys:
            await client.delete(*keys)
            logger.info("sql.cache.invalidated", datasource_id=datasource_id, count=len(keys))
    except Exception as e:
        logger.warning("sql.cache.invalidate_failed", error=str(e))


async def _record_cache_event(hit: bool) -> None:
    """作业4：命中率监控计数器。"""
    try:
        client = get_redis_client()
        await client.incr(STATS_HIT_KEY if hit else STATS_MISS_KEY)
    except Exception as e:
        logger.warning("sql.cache.stats_incr_failed", error=str(e))


async def get_cache_stats() -> dict:
    """作业4：返回缓存命中率统计，命中率低于阈值时输出 warning 日志。"""
    try:
        client = get_redis_client()
        hit = int(await client.get(STATS_HIT_KEY) or 0)
        miss = int(await client.get(STATS_MISS_KEY) or 0)
    except Exception as e:
        logger.warning("sql.cache.stats_read_failed", error=str(e))
        hit, miss = 0, 0

    total = hit + miss
    hit_rate = round(hit / total, 4) if total else 0.0

    if total > 0 and hit_rate < LOW_HIT_RATE_THRESHOLD:
        logger.warning("sql.cache.low_hit_rate", hit_rate=hit_rate, hit_count=hit, miss_count=miss)

    return {"hit_count": hit, "miss_count": miss, "total_count": total, "hit_rate": hit_rate}
