from datetime import datetime

import tiktoken

from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a SQL generation assistant for a data analytics platform.

Your task: Given a database schema and a natural language question, generate a single executable SQL query.

Rules:
1. ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, or any DDL/DML.
2. ONLY use table names and column names that appear in the provided schema.
3. Output ONLY the raw SQL query — no explanations, no markdown code blocks, no comments.
4. Use the correct SQL dialect as specified.
5. If the question is ambiguous, make reasonable assumptions and generate the most likely intended query.
6. Use table aliases for readability when joining multiple tables.
7. Always consider NULL values in aggregations.
8. For date/time questions, assume the current date is{current_date}."""

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the total revenue this month?",
        "sql": "SELECT SUM(amount) AS total_revenue FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) AND created_at < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'",
    },
    {
        "question": "Show me the top 5 customers by order count",
        "sql": "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 5",
    },
]


def format_schema_context(tables: list[dict]) -> str:
    """将 Schema 自省结果格式化为 Prompt 可用的文本。"""
    lines = []
    for table in tables:
        table_name = table["table_name"]
        lines.append(f"Table:{table_name}")
        for col in table["columns"]:
            nullable = "NULLABLE" if col["is_nullable"] else "NOT NULL"
            comment = f" —{col['comment']}" if col.get("comment") else ""
            lines.append(f"  -{col['column_name']} ({col['column_type']},{nullable}){comment}")
        lines.append("")
    return "\n".join(lines)




def _count_tokens(text: str, model: str) -> int:
    """作业2：估算文本的 token 数量。模型名未被 tiktoken 收录时退回 cl100k_base 编码。"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _fit_history_to_token_budget(
    conversation_history: list[dict], budget_tokens: int, model: str
) -> list[dict]:
    """作业2：动态上下文窗口。从最近的一轮开始往前取，按 user+assistant 成对保留，
    直到累计 token 数达到预算为止，而不是固定保留多少轮。"""
    if budget_tokens <= 0 or not conversation_history:
        return []

    # 按“轮”（每轮最多 2 条：user + assistant）从最近到最早切片
    pairs = []
    i = len(conversation_history)
    while i > 0:
        start = max(i - 2, 0)
        pairs.append(conversation_history[start:i])
        i = start

    selected: list[dict] = []
    used_tokens = 0
    for pair in pairs:
        pair_tokens = sum(_count_tokens(m["content"], model) for m in pair)
        if selected and used_tokens + pair_tokens > budget_tokens:
            break
        selected = pair + selected
        used_tokens += pair_tokens

    return selected


def build_text_to_sql_messages(
    question: str,
    schema_tables: list[dict],
    db_type: str = "postgresql",
    include_few_shot: bool = True,
    max_tables: int = 30,
    conversation_history: list[dict] | None = None,
    max_history_turns: int = 5,
) -> list[dict]:
    """构建发给 LLM 的完整 messages 列表，支持多轮对话上下文。"""

    if len(schema_tables) > max_tables:
        logger.warning("prompt.schema_truncated", total=len(schema_tables), max=max_tables)
        schema_tables = schema_tables[:max_tables]

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = SYSTEM_PROMPT.format(current_date=current_date)
    system_content += f"\n\nSQL Dialect:{db_type.upper()}"

    schema_text = format_schema_context(schema_tables)
    system_content += f"\n\nDatabase Schema:\n{schema_text}"

    # 多轮对话时追加上下文指令
    if conversation_history:
        system_content += """

Context Awareness Rules:
- The user may refer to previous questions and results using pronouns like "it", "that", "the same"
- When the user says "show me more detail" or "break it down", they mean the previous query's result
- When the user adds a filter like "only Q1" or "just this year", apply it to the previous query pattern
- Always consider the full conversation context when generating SQL"""

    messages = [{"role": "system", "content": system_content}]

    # Few-shot 示例
    if include_few_shot:
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["sql"]})

    # ✅ 多轮历史上下文（作业2：动态上下文窗口，按 token 预算而非固定轮数截断）
    recent_history: list[dict] = []
    if conversation_history:
        capped = conversation_history[-max_history_turns * 2:]  # 先用固定轮数做一个上限

        settings = get_settings()
        model = settings.openai_model
        used_tokens = sum(_count_tokens(m["content"], model) for m in messages)
        used_tokens += _count_tokens(question, model)
        budget = int(settings.llm_context_window_tokens * settings.history_token_budget_ratio) - used_tokens

        recent_history = _fit_history_to_token_budget(capped, budget, model)
        for turn in recent_history:
            messages.append(turn)

    # 当前问题
    messages.append({"role": "user", "content": question})

    logger.info(
        "prompt.built",
        question=question,
        schema_tables=len(schema_tables),
        history_turns=len(conversation_history or []) // 2,
        history_turns_used=len(recent_history) // 2,
        message_count=len(messages),
    )

    return messages


def append_retry_feedback(messages: list[dict], retry_history: list[dict]) -> list[dict]:
    """把之前失败的 SQL 及报错信息追加为对话轮次，让 LLM 基于报错修正 SQL（作业2）。

    retry_history: [{"sql": "...", "error": "..."}, ...]，按发生顺序排列。
    """
    for attempt in retry_history:
        messages.append({"role": "assistant", "content": attempt["sql"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    "上面这条 SQL 执行失败了，请根据报错信息修正后重新生成 SQL。"
                    "只输出修正后的 SQL，不要解释，不要使用 markdown 代码块。\n\n"
                    f"报错信息：\n{attempt['error']}"
                ),
            }
        )
    return messages