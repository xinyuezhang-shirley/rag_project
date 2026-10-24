"""作业3：检索质量评估（Hit@K）。

用法：
    # 1. 先生成标注模板（会列出你库里的 chunks，方便人工填 expected_chunk_ids）
    poetry run python scripts/eval_retrieval.py --init --email you@example.com

    # 2. 编辑 scripts/eval_dataset.json，给每个问题填上正确的 chunk id

    # 3. 跑评估
    poetry run python scripts/eval_retrieval.py --email you@example.com --k 3
"""
import argparse
import asyncio
import json
import pathlib
import sys

from sqlalchemy import select

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.db.session import async_session_factory          # noqa: E402
from app.models.user import User                          # noqa: E402
from app.models.document import Chunk, Document           # noqa: E402
from app.rag.retriever import vector_search, bm25_search, hybrid_search  # noqa: E402
from app.rag.reranker import rerank                       # noqa: E402

DATASET_PATH = pathlib.Path(__file__).parent / "eval_dataset.json"

TEMPLATE_QUESTIONS = [
    "怎么退钱", "退款需要多久", "数字产品能退款吗", "运费谁承担", "怎么联系客服",
    "保修期多长", "怎么修改收货地址", "支持哪些支付方式", "发票怎么开", "会员有什么权益",
]


async def _get_user(db, email: str | None) -> User:
    stmt = select(User).where(User.email == email) if email else select(User).limit(1)
    user = (await db.execute(stmt)).scalars().first()
    if not user:
        raise SystemExit(f"找不到用户（email={email}）。先注册一个用户并上传文档。")
    return user


async def init_dataset(email: str | None) -> None:
    """列出库里的 chunks + 生成标注模板。"""
    async with async_session_factory() as db:
        user = await _get_user(db, email)
        stmt = (
            select(Chunk.id, Chunk.content, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.user_id == user.id)
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        rows = (await db.execute(stmt)).all()

    if not rows:
        raise SystemExit("这个用户还没有任何 chunks，先上传文档。")

    print(f"\n用户 {user.email} 的 chunks（共 {len(rows)} 个）：\n")
    for cid, content, filename in rows:
        preview = content.replace("\n", " ")[:80]
        print(f"  chunk_id={cid:<5} [{filename}] {preview}")

    dataset = [{"question": q, "expected_chunk_ids": []} for q in TEMPLATE_QUESTIONS]
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"\n已生成标注模板：{DATASET_PATH}")
    print("请给每个问题填上 expected_chunk_ids（正确答案所在的 chunk id），再跑评估。")


async def evaluate(email: str | None, k: int, with_rerank: bool) -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"找不到 {DATASET_PATH}，先跑 --init 生成标注模板。")

    dataset = json.loads(DATASET_PATH.read_text())
    labeled = [d for d in dataset if d.get("expected_chunk_ids")]
    if not labeled:
        raise SystemExit("数据集里还没有任何标注（expected_chunk_ids 全是空的）。")

    methods = {"vector": vector_search, "bm25": bm25_search, "hybrid": hybrid_search}
    hits: dict[str, int] = {m: 0 for m in methods}
    per_question: list[dict] = []

    async with async_session_factory() as db:
        user = await _get_user(db, email)

        for item in labeled:
            question = item["question"]
            expected = set(item["expected_chunk_ids"])
            row = {"question": question, "expected": sorted(expected)}

            for name, fn in methods.items():
                results = await fn(db, question, user.id, top_k=k * 2)
                if with_rerank and results:
                    results = await rerank(question, results, top_k=k)
                top = results[:k]
                got = [r.chunk_id for r in top]
                hit = bool(expected & set(got))
                hits[name] += int(hit)
                row[name] = {"hit": hit, "got": got}

            per_question.append(row)

    total = len(labeled)
    print(f"\n{'='*60}")
    print(f"检索质量评估  |  {total} 个标注问题  |  Hit@{k}  |  rerank={'on' if with_rerank else 'off'}")
    print(f"{'='*60}")
    for name in methods:
        rate = hits[name] / total
        bar = "█" * int(rate * 20)
        print(f"  {name:<8} Hit@{k} = {rate:>6.1%}  ({hits[name]}/{total})  {bar}")

    print(f"\n{'-'*60}\n逐题明细（✗ = 正确 chunk 没进 Top-{k}）：\n")
    for row in per_question:
        print(f"  Q: {row['question']}   (expected={row['expected']})")
        for name in methods:
            mark = "✓" if row[name]["hit"] else "✗"
            print(f"      {mark} {name:<8} got={row[name]['got']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="列出 chunks 并生成标注模板")
    ap.add_argument("--email", default=None, help="评估哪个用户的文档（默认第一个用户）")
    ap.add_argument("--k", type=int, default=3, help="Hit@K 的 K（默认 3）")
    ap.add_argument("--rerank", action="store_true", help="开启 re-ranking 后再评估")
    args = ap.parse_args()

    if args.init:
        asyncio.run(init_dataset(args.email))
    else:
        asyncio.run(evaluate(args.email, args.k, args.rerank))


if __name__ == "__main__":
    main()
