"""作业 4：Prompt 优化实验。

用同一组 10 个问题，分别跑「当前 Prompt（含 few-shot）」和「去掉 few-shot」两个版本，
把生成的 SQL 打印出来供人工判断是否正确。不依赖真实数据库，只测 SQL 生成这一步。

运行：venv/bin/python scripts/prompt_experiment.py
"""
import asyncio

from app.sql_engine.prompt_builder import build_text_to_sql_messages
from app.sql_engine.generator import clean_sql_output
from app.llm.factory import get_llm_adapter

SCHEMA_TABLES = [
    {
        "table_name": "customers",
        "columns": [
            {"column_name": "id", "column_type": "integer", "is_nullable": False},
            {"column_name": "name", "column_type": "varchar", "is_nullable": False},
            {"column_name": "email", "column_type": "varchar", "is_nullable": False},
            {"column_name": "created_at", "column_type": "timestamp", "is_nullable": False},
        ],
    },
    {
        "table_name": "orders",
        "columns": [
            {"column_name": "id", "column_type": "integer", "is_nullable": False},
            {"column_name": "customer_id", "column_type": "integer", "is_nullable": False},
            {"column_name": "amount", "column_type": "numeric", "is_nullable": False},
            {"column_name": "status", "column_type": "varchar", "is_nullable": False},
            {"column_name": "created_at", "column_type": "timestamp", "is_nullable": False},
        ],
    },
    {
        "table_name": "products",
        "columns": [
            {"column_name": "id", "column_type": "integer", "is_nullable": False},
            {"column_name": "name", "column_type": "varchar", "is_nullable": False},
            {"column_name": "category", "column_type": "varchar", "is_nullable": False},
            {"column_name": "price", "column_type": "numeric", "is_nullable": False},
        ],
    },
    {
        "table_name": "order_items",
        "columns": [
            {"column_name": "id", "column_type": "integer", "is_nullable": False},
            {"column_name": "order_id", "column_type": "integer", "is_nullable": False},
            {"column_name": "product_id", "column_type": "integer", "is_nullable": False},
            {"column_name": "quantity", "column_type": "integer", "is_nullable": False},
            {"column_name": "unit_price", "column_type": "numeric", "is_nullable": False},
        ],
    },
]

QUESTIONS = [
    "What is the total revenue this month?",
    "How many orders has each customer placed?",
    "Which product category generated the most revenue?",
    "List the top 5 customers by total spend.",
    "How many orders were placed in the last 7 days?",
    "What is the average order value?",
    "Show the number of orders grouped by status.",
    "Which products have never been ordered?",
    "Find customers who have not placed any orders in the last 90 days.",
    "What is the month-over-month revenue growth for the last 6 months?",
]


async def run_variant(label: str, include_few_shot: bool) -> list[tuple[str, str]]:
    adapter = get_llm_adapter()
    print(f"\n{'=' * 70}\n{label} (include_few_shot={include_few_shot}, model={getattr(adapter, 'model', '?')})\n{'=' * 70}")

    results = []
    for question in QUESTIONS:
        messages = build_text_to_sql_messages(
            question=question,
            schema_tables=SCHEMA_TABLES,
            db_type="postgresql",
            include_few_shot=include_few_shot,
        )
        response = await adapter.generate_with_usage(messages)
        sql = clean_sql_output(response.content)
        results.append((question, sql))
        print(f"\nQ: {question}\nSQL: {sql}")

    return results


async def main():
    baseline = await run_variant("Baseline (current prompt)", include_few_shot=True)
    no_few_shot = await run_variant("Variant: no few-shot examples", include_few_shot=False)

    print(f"\n{'=' * 70}\nSide-by-side (review manually and mark correct/incorrect)\n{'=' * 70}")
    for (q, base_sql), (_, alt_sql) in zip(baseline, no_few_shot):
        print(f"\nQ: {q}")
        print(f"  [baseline]   {base_sql}")
        print(f"  [no-fewshot] {alt_sql}")


if __name__ == "__main__":
    asyncio.run(main())
