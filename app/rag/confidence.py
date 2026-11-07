from app.rag.context_builder import SourceRef
from app.rag.citation import CitationCheck
from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_confidence(
    sources: list[SourceRef],
    check: CitationCheck,
    reranked: bool = True,
) -> tuple[float, str]:
    """计算回答的置信度。"""

    if not sources:
        return 0.0, "low"

    # 三个维度加权
    top_score = max(s.score for s in sources) if sources else 0.0
    coverage = check.coverage
    verified = check.verified_ratio

    confidence = 0.4 * top_score + 0.3 * coverage + 0.3 * verified
    confidence = round(min(confidence, 1.0), 3)

    # 确定级别
    if confidence >= 0.75:
        level = "high"
    elif confidence >= 0.5:
        level = "medium"
    else:
        level = "low"

    # Re-ranking 未生效时，置信度封顶 medium
    if not reranked and level == "high":
        level = "medium"

    logger.info(
        "confidence.computed",
        top_score=round(top_score, 3),
        coverage=coverage,
        verified=verified,
        confidence=confidence,
        level=level,
        reranked=reranked,
    )

    return confidence, level