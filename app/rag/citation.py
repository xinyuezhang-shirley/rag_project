import re
from dataclasses import dataclass, field

from app.llm.factory import get_llm_adapter
from app.config.settings import get_settings
from app.rag.context_builder import SourceRef
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """一个引用的验证结果。"""
    index: int
    chunk_id: int
    document_id: int | None
    source: str | None
    section_title: str | None
    page_number: int | None
    snippet: str
    supported: bool
    support_score: float


@dataclass
class CitationCheck:
    """引用验证的整体结果。"""
    citations: list[Citation]
    invalid_indexes: list[int]
    uncited_sentences: list[str]
    coverage: float
    verified_ratio: float
    cleaned_answer: str


def parse_citations(answer: str) -> list[tuple[str, list[int]]]:
    """解析答案中的引用标记，返回 [(句子, [引用编号])] 列表。"""

    # 按句号切分（中英文）
    sentences = re.split(r'(?<=[。！？.!?])\s*', answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    # LLM 经常把引用写在句末标点之后（"...不支持退款。[1]"），切分后 "[1]" 会变成
    # 独立的一"句"：真正的句子被判为无引用（拉低 coverage），而引用本身拿空字符串
    # 去和来源比对（拉低 verified_ratio）。这里把纯引用片段并回上一句。
    merged: list[str] = []
    for sentence in sentences:
        if merged and re.fullmatch(r'(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+', sentence):
            merged[-1] = merged[-1] + sentence
        else:
            merged.append(sentence)
    sentences = merged

    results = []
    for sentence in sentences:
        # 匹配 [1] [2,3] [1, 3] 等格式
        refs = []
        for match in re.finditer(r'\[(\d+(?:\s*,\s*\d+)*)\]', sentence):
            nums = [int(n.strip()) for n in match.group(1).split(',')]
            refs.extend(nums)

        results.append((sentence, sorted(set(refs))))

    return results


def _bigram_set(text: str) -> set[str]:
    """生成文本的字符 bigram 集合。"""
    text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', text)  # 去掉引用标记
    text = re.sub(r'\s+', '', text)  # 去掉空白
    if len(text) < 2:
        return {text}
    return {text[i:i+2] for i in range(len(text) - 1)}


def _bigram_score(sentence: str, source_content: str) -> float:
    """字符 bigram 重叠度：句子里有多少比例的 bigram 能在来源里找到。"""
    sentence_bigrams = _bigram_set(sentence)
    if not sentence_bigrams:
        return 0.0
    source_bigrams = _bigram_set(source_content)
    return len(sentence_bigrams & source_bigrams) / len(sentence_bigrams)


def _iter_valid_pairs(answer: str, sources: list[SourceRef]):
    """遍历所有「句子-合法引用」对，顺序固定。

    打分和构建结果都走这一个遍历，保证 bigram 和 LLM 两种打分方式拿到的
    pair 顺序完全一致（LLM 批量打分依赖顺序对齐）。
    """
    source_map = {s.index: s for s in sources}
    max_index = max(source_map.keys()) if source_map else 0

    for sent_idx, (sentence, refs) in enumerate(parse_citations(answer)):
        clean_sentence = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', sentence).strip()
        for ref_idx in refs:
            if ref_idx > max_index or ref_idx not in source_map:
                continue
            yield sent_idx, ref_idx, clean_sentence, source_map[ref_idx]


def _verify_core(
    answer: str,
    sources: list[SourceRef],
    support_threshold: float,
    score_fn,
) -> CitationCheck:
    """引用验证主逻辑。score_fn(sent_idx, clean_sentence, source) -> 0..1 的支撑度。"""

    parsed = parse_citations(answer)
    source_map = {s.index: s for s in sources}
    max_index = max(source_map.keys()) if source_map else 0

    citations: list[Citation] = []
    invalid_indexes: list[int] = []
    uncited_sentences: list[str] = []
    cleaned_parts: list[str] = []

    total_factual = 0
    cited_factual = 0
    supported_count = 0
    total_citation_count = 0

    for sent_idx, (sentence, refs) in enumerate(parsed):
        # 判断是否是事实性句子（排除纯引用标记、过短的句子）
        clean_sentence = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', sentence).strip()
        is_factual = len(clean_sentence) > 5

        if is_factual:
            total_factual += 1

        if not refs:
            if is_factual:
                uncited_sentences.append(clean_sentence)
            cleaned_parts.append(sentence)
            continue

        if is_factual:
            cited_factual += 1

        valid_refs = []
        for ref_idx in refs:
            if ref_idx > max_index or ref_idx not in source_map:
                invalid_indexes.append(ref_idx)
                continue

            source = source_map[ref_idx]
            total_citation_count += 1

            score = score_fn(sent_idx, clean_sentence, source)

            supported = score >= support_threshold
            if supported:
                supported_count += 1

            citations.append(Citation(
                index=ref_idx,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                source=source.source,
                section_title=source.section_title,
                page_number=source.page_number,
                snippet=source.content[:100],
                supported=supported,
                support_score=round(score, 3),
            ))
            valid_refs.append(ref_idx)

        # 重建句子：去掉越界引用
        cleaned_sentence = sentence
        for inv_idx in set(refs) - set(valid_refs):
            cleaned_sentence = re.sub(rf'\[{inv_idx}\]', '', cleaned_sentence)
        cleaned_parts.append(cleaned_sentence.strip())

    coverage = (cited_factual / total_factual) if total_factual > 0 else 0.0
    verified_ratio = (supported_count / total_citation_count) if total_citation_count > 0 else 0.0
    cleaned_answer = ' '.join(cleaned_parts)

    logger.info(
        "citation.verified",
        total_citations=total_citation_count,
        supported=supported_count,
        invalid=len(invalid_indexes),
        coverage=round(coverage, 3),
    )

    return CitationCheck(
        citations=citations,
        invalid_indexes=sorted(set(invalid_indexes)),
        uncited_sentences=uncited_sentences,
        coverage=round(coverage, 3),
        verified_ratio=round(verified_ratio, 3),
        cleaned_answer=cleaned_answer,
    )


def verify_citations(
    answer: str,
    sources: list[SourceRef],
    support_threshold: float = 0.3,
) -> CitationCheck:
    """验证答案中的引用是否真的被来源支撑（bigram 字面重叠，不调 LLM）。"""
    return _verify_core(
        answer, sources, support_threshold,
        lambda _i, sentence, source: _bigram_score(sentence, source.content),
    )


# ── 作业2：LLM-as-judge 引用验证 ──

JUDGE_PROMPT = """You are a citation verifier. For each numbered pair below, judge whether the SOURCE actually supports the STATEMENT.

Scoring:
- 1.0 = the source directly and fully states the claim
- 0.5 = the source partially supports it or implies it
- 0.0 = the source does not support the claim at all

Judge meaning, not word overlap: a paraphrase that is fully supported still scores 1.0.

{pairs}

Output ONLY a JSON array of {count} numbers between 0.0 and 1.0, in the same order as the pairs. Nothing else.
Example: [1.0, 0.0, 0.5]"""


async def judge_pairs_llm(
    pairs: list[tuple[str, str]]
) -> list[float] | None:
    """批量判断 [(statement, source_content)]，一次 LLM 调用打完所有分。

    批量而不是逐个调用：5 个引用逐个判断就是 5 次往返，批量只要 1 次。
    返回 None 表示判断失败，调用方应降级到 bigram。
    """
    if not pairs:
        return []

    import json

    blocks = []
    for i, (statement, source_content) in enumerate(pairs, start=1):
        blocks.append(
            f"[Pair {i}]\nSTATEMENT: {statement}\nSOURCE: {source_content[:600]}"
        )

    prompt = JUDGE_PROMPT.format(pairs="\n\n".join(blocks), count=len(pairs))

    try:
        adapter = get_llm_adapter()
        response = await adapter.generate([{"role": "user", "content": prompt}])
        scores = json.loads(response.strip())

        if not isinstance(scores, list) or len(scores) != len(pairs):
            logger.warning(
                "citation.judge.count_mismatch",
                expected=len(pairs),
                got=len(scores) if isinstance(scores, list) else "not-a-list",
            )
            return None

        return [max(0.0, min(1.0, float(x))) for x in scores]

    except Exception as e:
        logger.warning("citation.judge.failed", error=str(e))
        return None


async def verify_citations_llm(
    answer: str,
    sources: list[SourceRef],
    support_threshold: float = 0.3,
) -> CitationCheck:
    """作业2：用 LLM-as-judge 替代 bigram 做引用验证。LLM 失败时降级到 bigram。"""

    pair_keys = [(si, ri) for si, ri, _s, _src in _iter_valid_pairs(answer, sources)]
    pair_inputs = [(s, src.content) for _si, _ri, s, src in _iter_valid_pairs(answer, sources)]

    scores = await judge_pairs_llm(pair_inputs)

    if scores is None:
        logger.info("citation.judge.fallback_to_bigram")
        return verify_citations(answer, sources, support_threshold)

    score_map = dict(zip(pair_keys, scores))

    logger.info("citation.judge.completed", pairs=len(pair_keys))

    return _verify_core(
        answer, sources, support_threshold,
        lambda si, sentence, source: score_map.get(
            (si, source.index),
            _bigram_score(sentence, source.content),  # 兜底
        ),
    )


async def verify_citations_auto(
    answer: str,
    sources: list[SourceRef],
    support_threshold: float = 0.3,
    method: str | None = None,
) -> CitationCheck:
    """按配置选择验证方式：bigram（默认，免费）或 llm（更准，多一次调用）。"""
    method = method or get_settings().rag_citation_verify_method
    if method == "llm":
        return await verify_citations_llm(answer, sources, support_threshold)
    return verify_citations(answer, sources, support_threshold)


# ── 作业4：引用聚合（按文档分组）──

def group_citations_by_document(citations: list[Citation]) -> list[dict]:
    """把同一文档的多个引用片段合并成一组。

    答案正文里的 [1] [3] 编号保持不变——分组只改变「展示结构」，不重新编号，
    否则正文的标记就会指向错误的来源。每组里保留原始 index。
    """
    groups: dict[tuple, dict] = {}

    for c in citations:
        key = (c.document_id, c.source)
        if key not in groups:
            groups[key] = {
                "document_id": c.document_id,
                "source": c.source,
                "indexes": [],
                "citations": [],
            }
        group = groups[key]
        if c.index not in group["indexes"]:
            group["indexes"].append(c.index)
        group["citations"].append({
            "index": c.index,
            "chunk_id": c.chunk_id,
            "section_title": c.section_title,
            "page_number": c.page_number,
            "snippet": c.snippet,
            "supported": c.supported,
            "support_score": c.support_score,
        })

    result = list(groups.values())
    for g in result:
        g["indexes"].sort()
        g["citations"].sort(key=lambda x: x["index"])
    result.sort(key=lambda g: min(g["indexes"]) if g["indexes"] else 0)

    return result
