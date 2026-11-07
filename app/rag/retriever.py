from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedder import get_embedder
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """一条检索结果。"""
    chunk_id: int
    content: str
    score: float
    source: str | None = None
    page_number: int | None = None
    section_title: str | None = None
    document_id: int | None = None
    retrieval_method: str = "vector"  # vector / bm25 / hybrid


def _row_to_result(row, method: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=row.id,
        content=row.content,
        score=round(float(row.score), 4),
        source=row.source,
        page_number=row.page_number,
        section_title=row.section_title,
        document_id=row.document_id,
        retrieval_method=method,
    )


async def vector_search(
    db: AsyncSession,
    query: str,
    user_id: int,
    top_k: int = 5,
    document_ids: list[int] | None = None,
    score_threshold: float = 0.3,
) -> list[RetrievalResult]:
    """纯向量检索：Embedding 余弦相似度。"""

    embedder = get_embedder()
    query_embedding = await embedder.embed_query(query)

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # 注：:embedding 必须显式 CAST 成 vector，否则驱动会把它当 text 传过去，
    # Postgres 会报 "operator does not exist: vector <=> text"
    sql = """
        SELECT
            c.id,
            c.content,
            c.source,
            c.page_number,
            c.section_title,
            c.document_id,
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.user_id = :user_id
          AND c.embedding IS NOT NULL
    """

    params: dict = {"embedding": embedding_str, "user_id": user_id}

    # 作业2：按文档过滤
    if document_ids:
        sql += " AND c.document_id = ANY(:doc_ids)"
        params["doc_ids"] = document_ids

    sql += """
          AND 1 - (c.embedding <=> CAST(:embedding AS vector)) >= :threshold
        ORDER BY c.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """
    params["threshold"] = score_threshold
    params["top_k"] = top_k

    result = await db.execute(text(sql), params)
    rows = result.fetchall()

    results = [_row_to_result(r, "vector") for r in rows]

    logger.info(
        "retrieval.vector",
        query=query[:50],
        results=len(results),
        top_score=results[0].score if results else 0,
        filtered_documents=len(document_ids) if document_ids else 0,
    )

    return results


async def bm25_search(
    db: AsyncSession,
    query: str,
    user_id: int,
    top_k: int = 5,
    document_ids: list[int] | None = None,
) -> list[RetrievalResult]:
    """BM25 关键词检索：PostgreSQL 全文搜索（ts_rank_cd 近似 BM25）。"""

    sql = """
        SELECT
            c.id,
            c.content,
            c.source,
            c.page_number,
            c.section_title,
            c.document_id,
            ts_rank_cd(
                to_tsvector('simple', c.content),
                plainto_tsquery('simple', :query)
            ) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.user_id = :user_id
          AND to_tsvector('simple', c.content) @@ plainto_tsquery('simple', :query)
    """

    params: dict = {"query": query, "user_id": user_id}

    # 作业2：按文档过滤
    if document_ids:
        sql += " AND c.document_id = ANY(:doc_ids)"
        params["doc_ids"] = document_ids

    sql += " ORDER BY score DESC LIMIT :top_k"
    params["top_k"] = top_k

    result = await db.execute(text(sql), params)
    rows = result.fetchall()

    results = [_row_to_result(r, "bm25") for r in rows]

    logger.info(
        "retrieval.bm25",
        query=query[:50],
        results=len(results),
        filtered_documents=len(document_ids) if document_ids else 0,
    )

    return results


async def hybrid_search(
    db: AsyncSession,
    query: str,
    user_id: int,
    top_k: int = 5,
    document_ids: list[int] | None = None,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    score_threshold: float = 0.2,
) -> list[RetrievalResult]:
    """混合检索：向量 + BM25 加权合并。"""

    # 两路检索各取多一些，合并后再截取 top_k
    vector_results = await vector_search(
        db, query, user_id,
        top_k=top_k * 2,
        document_ids=document_ids,
        score_threshold=0.1,  # 放宽阈值，合并后再过滤
    )

    bm25_results = await bm25_search(
        db, query, user_id,
        top_k=top_k * 2,
        document_ids=document_ids,
    )

    score_map: dict[int, RetrievalResult] = {}

    # 归一化向量分数到 [0, 1] 后加权
    if vector_results:
        max_v = max(r.score for r in vector_results)
        for r in vector_results:
            norm_score = (r.score / max_v) if max_v > 0 else 0
            r.score = norm_score * vector_weight
            score_map[r.chunk_id] = r

    # 归一化 BM25 分数到 [0, 1] 后加权
    if bm25_results:
        max_b = max(r.score for r in bm25_results)
        for r in bm25_results:
            norm_score = (r.score / max_b) if max_b > 0 else 0
            weighted = norm_score * bm25_weight
            if r.chunk_id in score_map:
                # 同一个 chunk 两路都命中，分数叠加
                score_map[r.chunk_id].score += weighted
                score_map[r.chunk_id].retrieval_method = "hybrid"
            else:
                r.score = weighted
                r.retrieval_method = "bm25"
                score_map[r.chunk_id] = r

    merged = sorted(score_map.values(), key=lambda r: r.score, reverse=True)
    merged = [r for r in merged if r.score >= score_threshold][:top_k]
    for r in merged:
        r.score = round(r.score, 4)  # 加权后重新取整，否则响应里是 0.6264379247472236

    logger.info(
        "retrieval.hybrid",
        query=query[:50],
        vector_hits=len(vector_results),
        bm25_hits=len(bm25_results),
        merged=len(merged),
    )

    return merged
