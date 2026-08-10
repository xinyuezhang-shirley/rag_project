from app.llm.factory import get_llm_adapter
from app.llm.base import LLMResponse
from app.sql_engine.prompt_builder import build_text_to_sql_messages
from app.sql_engine.validator import validate_sql
from app.core.logging import get_logger

logger = get_logger(__name__)

# 作业3：可修复 vs 不可修复的错误分类。不可修复的错误（连接失败、权限不足、超时）
# 无法通过让 LLM 重写 SQL 解决，直接返回，不浪费重试次数。
# 分类依据 app.sql_engine.executor._classify_error 产出的 error_type：
#   - syntax_error / table_not_found / unknown → 可能是列名拼写、方言差异、表名大小写等问题，值得重试
#   - permission_denied / connection_error / timeout → 换一条 SQL 也解决不了，直接放弃重试
UNFIXABLE_ERROR_TYPES = {"permission_denied", "connection_error", "timeout"}


def _is_fixable(error_type: str) -> bool:
    return error_type not in UNFIXABLE_ERROR_TYPES


RETRY_PROMPT = """The previous SQL query failed with the following error:

Error:{error_message}

Failed SQL:
{failed_sql}

Please generate a corrected SQL query that fixes this error.
Remember:
1. ONLY generate SELECT statements
2. ONLY use table and column names from the provided schema
3. Output ONLY the raw SQL query
4. Fix the specific error mentioned above"""


async def retry_with_feedback(
    question: str,
    failed_sql: str,
    error_message: str,
    schema_tables: list[dict],
    db_type: str = "postgresql",
    conversation_history: list[dict] | None = None,
    max_retries: int = 2,
    allowed_tables: list[str] | None = None,
    error_type: str = "unknown",
) -> tuple[str, LLMResponse] | None:
    """尝试让 LLM 根据错误信息修正 SQL。"""

    from app.sql_engine.generator import clean_sql_output

    if not _is_fixable(error_type):
        logger.info("sql.retry.skipped_unfixable", error_type=error_type, error=error_message[:100])
        return None

    current_error = error_message
    current_failed_sql = failed_sql

    for attempt in range(1, max_retries + 1):
        logger.info(
            "sql.retry.attempt",
            attempt=attempt,
            max_retries=max_retries,
            error=current_error[:100],
        )

        # 构建修正 prompt
        retry_content = RETRY_PROMPT.format(
            error_message=current_error,
            failed_sql=current_failed_sql,
        )

        messages = build_text_to_sql_messages(
            question=question,
            schema_tables=schema_tables,
            db_type=db_type,
            conversation_history=conversation_history,
        )

        # 把失败信息作为额外上下文追加
        messages.append({"role": "assistant", "content": current_failed_sql})
        messages.append({"role": "user", "content": retry_content})

        adapter = get_llm_adapter()
        response = await adapter.generate_with_usage(messages)
        new_sql = clean_sql_output(response.content)

        if not new_sql:
            logger.warning("sql.retry.empty_result", attempt=attempt)
            continue

        # 修正后的 SQL 也要过安全校验
        validation = validate_sql(
            sql=new_sql,
            allowed_tables=allowed_tables,
            db_type=db_type,
        )

        if not validation.is_safe:
            logger.warning(
                "sql.retry.validation_failed",
                attempt=attempt,
                reason=validation.reason,
            )
            continue

        logger.info("sql.retry.success", attempt=attempt, new_sql_length=len(new_sql))
        return new_sql, response

    logger.warning("sql.retry.exhausted", max_retries=max_retries)
    return None