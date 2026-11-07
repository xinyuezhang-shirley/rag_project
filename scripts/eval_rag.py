"""作业3：RAG 评测集。

统计两个指标：
  - answerable 准确率：该答的答了、不该答的拒答了（混淆矩阵的对角线）
  - 引用命中率：答案引用的 chunk 里，是否包含人工标注的正确 chunk

用法：
    poetry run python scripts/eval_rag.py --init --email you@example.com   # 生成标注模板
    # 编辑 scripts/rag_eval_dataset.json 填 expected_chunk_ids
    poetry run python scripts/eval_rag.py --email you@example.com
"""
import argparse
import asyncio
import json
import pathlib
import sys

from sqlalchemy import select

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.db.session import async_session_factory              # noqa: E402
from app.models.user import User                              # noqa: E402
from app.models.document import Chunk, Document               # noqa: E402
from app.services.document_qa import answer_from_documents    # noqa: E402

DATASET_PATH = pathlib.Path(__file__).parent / "rag_eval_dataset.json"

# 5 个文档里有答案的 + 5 个文档里没有的
TEMPLATE = [
    {"question": "数字产品可以退款吗？", "answerable": True, "expected_chunk_ids": []},
    {"question": "退款多久能到账？", "answerable": True, "expected_chunk_ids": []},
    {"question": "退货的运费谁承担？", "answerable": True, "expected_chunk_ids": []},
    {"question": "电子产品保修多久？", "answerable": True, "expected_chunk_ids": []},
    {"question": "什么情况下不保修？", "answerable": True, "expected_chunk_ids": []},
    {"question": "公司年假有几天？", "answerable": False, "expected_chunk_ids": []},
    {"question": "CEO 是谁？", "answerable": False, "expected_chunk_ids": []},
    {"question": "怎么申请报销差旅费？", "answerable": False, "expected_chunk_ids": []},
    {"question": "服务器部署在哪个云厂商？", "answerable": False, "expected_chunk_ids": []},
    {"question": "员工股票期权怎么算？", "answerable": False, "expected_chunk_ids": []},
]


async def _get_user(db, email: str | None) -> User:
    stmt = select(User).where(User.email == email) if email else select(User).limit(1)
    user = (await db.execute(stmt)).scalars().first()
    if not user:
        raise SystemExit(f"找不到用户（email={email}）")
    return user


async def init_dataset(email: str | None) -> None:
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
        raise SystemExit("该用户没有任何 chunks，先上传文档。")

    print(f"\n{user.email} 的 chunks（{len(rows)} 个）：\n")
    for cid, content, filename in rows:
        print(f"  chunk_id={cid:<5} [{filename}] {content.replace(chr(10),' ')[:70]}")

    DATASET_PATH.write_text(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
    print(f"\n模板已写入 {DATASET_PATH}")
    print("给 5 个 answerable=true 的问题填上 expected_chunk_ids，然后跑评测。")


async def evaluate(email: str | None) -> None:
    if not DATASET_PATH.exists():
        raise SystemExit(f"找不到 {DATASET_PATH}，先跑 --init")

    dataset = json.loads(DATASET_PATH.read_text())

    tp = tn = fp = fn = 0          # answerable 混淆矩阵
    hit = 0                        # 引用命中
    hit_total = 0
    rows = []

    async with async_session_factory() as db:
        user = await _get_user(db, email)

        for item in dataset:
            q = item["question"]
            expected_answerable = item["answerable"]
            gold = set(item.get("expected_chunk_ids") or [])

            result = await answer_from_documents(db, user, q)
            got_answerable = result.answerable
            cited = [c.chunk_id for c in result.citations]

            if expected_answerable and got_answerable:
                tp += 1
            elif not expected_answerable and not got_answerable:
                tn += 1
            elif not expected_answerable and got_answerable:
                fp += 1   # 最危险：文档里没有却编了答案
            else:
                fn += 1   # 过于保守：有答案却拒答

            cite_hit = None
            if expected_answerable and gold:
                hit_total += 1
                cite_hit = bool(gold & set(cited))
                hit += int(cite_hit)

            rows.append({
                "q": q, "exp": expected_answerable, "got": got_answerable,
                "conf": result.confidence, "level": result.confidence_level,
                "cited": cited, "gold": sorted(gold), "cite_hit": cite_hit,
            })

    total = len(dataset)
    acc = (tp + tn) / total if total else 0.0
    cite_rate = hit / hit_total if hit_total else 0.0

    print(f"\n{'='*64}")
    print(f"RAG 评测  |  {total} 题")
    print(f"{'='*64}")
    print(f"  answerable 准确率 : {acc:>6.1%}  ({tp+tn}/{total})")
    print(f"    ✓ 该答且答了 (TP): {tp}")
    print(f"    ✓ 该拒且拒了 (TN): {tn}")
    print(f"    ✗ 不该答却答了 (FP): {fp}   ← 幻觉，最危险")
    print(f"    ✗ 该答却拒答 (FN): {fn}   ← 过度保守")
    print(f"  引用命中率        : {cite_rate:>6.1%}  ({hit}/{hit_total} 道已标注题)")

    print(f"\n{'-'*64}\n逐题明细：\n")
    for r in rows:
        ok = "✓" if r["exp"] == r["got"] else "✗"
        ch = "" if r["cite_hit"] is None else ("  引用✓" if r["cite_hit"] else "  引用✗")
        print(f"  {ok} {r['q']}")
        print(f"      expected_answerable={r['exp']} got={r['got']} "
              f"conf={r['conf']}({r['level']}){ch}")
        if r["gold"]:
            print(f"      gold={r['gold']} cited={r['cited']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--email", default=None)
    args = ap.parse_args()
    asyncio.run(init_dataset(args.email) if args.init else evaluate(args.email))


if __name__ == "__main__":
    main()
