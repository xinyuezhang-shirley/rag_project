import asyncio
import hashlib
import json

from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.db.redis import get_redis_client
from app.core.logging import get_logger
from app.core.errors import AppError

logger = get_logger(__name__)


class EmbeddingError(AppError):
    def __init__(self, message: str = "Embedding 生成失败"):
        super().__init__(code="EMBEDDING_ERROR", message=message, status_code=502)


class Embedder:
    """Embedding 生成器，支持批量调用、rate limit 控制和缓存（作业1）。"""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=30,
        )
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.batch_size = 100  # OpenAI 一次最多 2048 条，但 100 条更稳
        self.cache_enabled = settings.embedding_cache_enabled

    # ── 作业1：Embedding 缓存 ──
    # key 里带上 model：换模型后向量维度和语义空间都变了，旧缓存必须失效
    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding_cache:{self.model}:{h}"

    async def _cache_get_many(self, texts: list[str]) -> dict[int, list[float]]:
        """批量查缓存，返回 {索引: 向量}。Redis 故障时返回空 dict，不阻断主链路。"""
        if not self.cache_enabled or not texts:
            return {}
        try:
            client = get_redis_client()
            cached_raw = await client.mget([self._cache_key(t) for t in texts])
            hits: dict[int, list[float]] = {}
            for i, raw in enumerate(cached_raw):
                if raw:
                    try:
                        hits[i] = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
            return hits
        except Exception as e:
            logger.warning("embedding.cache.get_failed", error=str(e))
            return {}

    async def _cache_set_many(self, pairs: list[tuple[str, list[float]]]) -> None:
        """批量写缓存。Embedding 是幂等的，所以不设 TTL（永久有效）。"""
        if not self.cache_enabled or not pairs:
            return
        try:
            client = get_redis_client()
            pipe = client.pipeline()
            for text, vector in pairs:
                pipe.set(self._cache_key(text), json.dumps(vector))
            await pipe.execute()
            logger.info("embedding.cache.set", count=len(pairs))
        except Exception as e:
            logger.warning("embedding.cache.set_failed", error=str(e))

    async def embed_single(self, text: str) -> list[float]:
        """生成单条文本的 Embedding。"""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 Embedding，命中缓存的文本不再调用 API（作业1）。"""
        if not texts:
            return []

        # 1. 先查缓存
        cached = await self._cache_get_many(texts)
        missing_indices = [i for i in range(len(texts)) if i not in cached]

        logger.info(
            "embedding.cache.lookup",
            total=len(texts),
            hits=len(cached),
            misses=len(missing_indices),
        )

        # 2. 只对未命中的文本调用 API
        fresh: dict[int, list[float]] = {}
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self._embed_uncached(missing_texts)
            fresh = dict(zip(missing_indices, new_embeddings))
            await self._cache_set_many(list(zip(missing_texts, new_embeddings)))

        # 3. 按原始顺序拼回去
        return [cached.get(i) or fresh[i] for i in range(len(texts))]

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """真正调用 OpenAI API（分批 + rate limit 保护）。"""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(
                    "embedding.batch.completed",
                    batch_index=i // self.batch_size,
                    batch_size=len(batch),
                    total_tokens=response.usage.total_tokens,
                )

                # Rate limit 保护：批次之间稍微等一下
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(
                    "embedding.batch.failed",
                    batch_index=i // self.batch_size,
                    error=str(e),
                )
                raise EmbeddingError(f"Embedding 生成失败：{str(e)}")

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """生成查询文本的 Embedding（和 embed_single 一样，语义上区分用途）。"""
        return await self.embed_single(query)


def get_embedder() -> Embedder:
    return Embedder()
