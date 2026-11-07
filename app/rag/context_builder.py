from dataclasses import dataclass
import tiktoken

from app.rag.retriever import RetrievalResult
from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SourceRef:
    """一个来源引用的完整信息。"""
    index: int
    chunk_id: int
    document_id: int | None
    source: str | None
    section_title: str | None
    page_number: int | None
    score: float
    content: str


def reorder_for_attention(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Lost in the Middle 重排：最相关的放开头、次相关的放结尾。"""
    if len(results) <= 2:
        return results

    # 输入已按分数降序排列
    reordered = []
    left = []
    right = []

    for i, r in enumerate(results):
        if i % 2 == 0:
            left.append(r)
        else:
            right.append(r)

    # left: [1st, 3rd, 5th, ...]  right: [2nd, 4th, ...]
    # 结果: [1st, 3rd, 5th, ..., 4th, 2nd]
    reordered = left + list(reversed(right))

    return reordered


def build_context(
    results: list[RetrievalResult],
    max_tokens: int | None = None,
) -> tuple[str, list[SourceRef]]:
    """构建喂给 LLM 的 context 文本和来源引用列表。"""

    settings = get_settings()
    if max_tokens is None:
        max_tokens = settings.rag_context_max_tokens

    # Token 计数器
    try:
        enc = tiktoken.encoding_for_model(settings.openai_model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    sources: list[SourceRef] = []
    context_parts: list[str] = []
    total_tokens = 0

    for i, r in enumerate(results):
        # 构建单个来源的文本
        header_parts = [f"[Source{i + 1}]"]
        if r.source:
            header_parts.append(f"File:{r.source}")
        if r.section_title:
            header_parts.append(f"Section:{r.section_title}")
        if r.page_number:
            header_parts.append(f"Page:{r.page_number}")

        header = " | ".join(header_parts)
        block = f"{header}\n{r.content}"

        # Token 预算检查
        block_tokens = len(enc.encode(block))
        if total_tokens + block_tokens > max_tokens:
            logger.info(
                "context.truncated",
                included=i,
                total=len(results),
                tokens_used=total_tokens,
                max_tokens=max_tokens,
            )
            break

        total_tokens += block_tokens
        context_parts.append(block)

        sources.append(SourceRef(
            index=i + 1,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            source=r.source,
            section_title=r.section_title,
            page_number=r.page_number,
            score=r.score,
            content=r.content,
        ))

    context_text = "\n\n---\n\n".join(context_parts)

    logger.info(
        "context.built",
        sources=len(sources),
        total_tokens=total_tokens,
    )

    return context_text, sources