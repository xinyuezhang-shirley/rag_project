"""作业2：对比 bigram 和 LLM-as-judge 两种引用验证方式。

对评测集里 answerable 的问题各跑一次 RAG，拿到同一份 answer + sources，
然后分别用两种方式验证引用，逐条对比分数和 supported 判定。

用法：
    poetry run python scripts/compare_citation_verify.py --email you@example.com
"""
import argparse
import asyncio
import json
import pathlib
import sys
import time

from sqlalchemy import select

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.db.session import async_session_factory                      # noqa: E402
from app.models.user import User                                      # noqa: E402
from app.rag.retriever import hybrid_search                           # noqa: E402
from app.rag.reranker import rerank                                   # noqa: E402
from app.rag.context_builder import reorder_for_attention, build_context  # noqa: E402
from app.rag.answer_generator import generate_answer                  # noqa: E402
from app.rag.citation import verify_citations, verify_citations_llm    # noqa: E402
from app.config.settings import get_settings                          # noqa: E402

DATASET_PATH = pathlib.Path(__file__).parent / "rag_eval_dataset.json"
FALLBACK_QUESTIONS = [
    "数字产品可以退款吗？", "退款多久能到账？", "退货的运费谁承担？",
    "电子产品保修多久？", "什么情况下不保修？",
]


async def main(email: str | None) -> None:
    settings = get_settings()

    if DATASET_PATH.exists():
        data = json.loads(DATASET_PATH.read_text())
        questions = [d["question"] for d in data if d.get("answerable")]
    else:
        questions = FALLBACK_QUESTIONS

    agree = 0
    compared = 0

    async with async_session_factory() as db:
        stmt = select(User).where(User.email == email) if email else select(User).limit(1)
        user = (await db.execute(stmt)).scalars().first()
        if not user:
            raise SystemExit("找不到用户")

        for q in questions:
            results = await hybrid_search(db, q, user.id, top_k=settings.rag_top_k * 2)
            if not results:
                print(f"\n[跳过] {q} —— 检索为空")
                continue

            results = await rerank(q, results, top_k=settings.rag_top_k)
            results = reorder_for_attention(results)
            context_text, sources = build_context(results)
            response = await generate_answer(question=q, sources_text=context_text)
            answer = response.content.strip()

            if answer == "NO_ANSWER":
                print(f"\n[跳过] {q} —— NO_ANSWER")
                continue

            t0 = time.perf_counter()
            big = verify_citations(answer, sources, settings.rag_citation_support_threshold)
            t_big = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            llm = await verify_citations_llm(answer, sources, settings.rag_citation_support_threshold)
            t_llm = (time.perf_counter() - t0) * 1000

            print(f"\n{'='*66}\nQ: {q}\nA: {answer[:90]}")
            print(f"  bigram: {t_big:7.1f}ms   llm-judge: {t_llm:7.1f}ms")
            for b, l in zip(big.citations, llm.citations):
                same = "same " if b.supported == l.supported else "DIFFER"
                compared += 1
                agree += int(b.supported == l.supported)
                print(f"    [{b.index}] bigram={b.support_score:<6} ({b.supported})  "
                      f"llm={l.support_score:<6} ({l.supported})  {same}")

    if compared:
        print(f"\n{'='*66}")
        print(f"supported 判定一致率: {agree/compared:.1%}  ({agree}/{compared} 条引用)")
        print("注：bigram 衡量字面重叠，LLM 衡量语义支撑。改写过的句子 bigram 会偏低。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=None)
    asyncio.run(main(ap.parse_args().email))
