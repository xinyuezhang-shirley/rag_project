import json
from datetime import datetime, timezone

from sqlalchemy import select, func, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User
from app.models.datasource import DataSource
from app.models.conversation import Conversation, Message
from app.models.query_execution import QueryExecution
from app.schemas.chat import (
    ChatResponse,
    QueryResult,
    MessageResponse,
    ConversationListItem,
)
from app.schemas.datasource import TableSchema
from app.services.datasource import get_datasource
from app.services.schema_inspector import introspect
from app.sql_engine.generator import generate_sql
from app.sql_engine.executor import execute_sql
from app.core.errors import NotFoundError, ValidationError, SQLExecutionError
from app.core.logging import get_logger
from app.core.pagination import encode_cursor, decode_cursor
from app.config.settings import get_settings
from app.sql_engine.validator import validate_sql, ValidationResult
from app.sql_engine.explainer import explain_result
from app.sql_engine.retry import retry_with_feedback
from app.sql_engine.cache import (
    get_cached_sql_with_semantic_fallback,
    set_cached_sql,
    invalidate_cache,
)
from app.sql_engine.table_selector import select_relevant_tables

logger = get_logger(__name__)


async def create_conversation(
    db: AsyncSession, user: User, datasource_id: int, title: str = "新对话"
) -> Conversation:
    ds = await get_datasource(db, user, datasource_id)

    conv = Conversation(
        user_id=user.id,
        datasource_id=ds.id,
        title=title,
    )
    db.add(conv)
    await db.flush()

    logger.info("conversation.created", conversation_id=conv.id, user_id=user.id)
    return conv


async def get_conversation(
    db: AsyncSession, user: User, conversation_id: int
) -> Conversation:
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise NotFoundError("对话", conversation_id)
    return conv


async def send_message(
    db: AsyncSession, user: User, conversation_id: int, question: str
) -> ChatResponse:
    """核心方法：接收用户问题 → 生成 SQL → 执行 → 返回结果。"""

    # 1. 获取对话和数据源
    conv = await get_conversation(db, user, conversation_id)
    if not conv.datasource_id:
        raise ValidationError("该对话未关联数据源，无法生成 SQL")

    ds = await get_datasource(db, user, conv.datasource_id)

    # 2. 存用户消息
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()

    # 3. 获取 Schema（调用第二节课的自省能力）
    schema_result = await introspect(db, user, ds.id)
    schema_tables = [
        {"table_name": t.table_name, "columns": [c.model_dump() for c in t.columns]}
        for t in schema_result.tables
    ]

    # 4. 生成 SQL（✅ 4.5 新增：安全校验；✅ 作业2：执行失败时把报错反馈给 LLM 重试，最多 MAX_RETRIES 次）
    allowed_table_names = [t["table_name"] for t in schema_tables]
    allowed_columns = {
        t["table_name"]: [c["column_name"] for c in t["columns"]] for t in schema_tables
    }

    settings = get_settings()
    sensitive_columns = list(settings.sql_sensitive_columns)
    if ds.sensitive_columns:
        sensitive_columns += [c.strip() for c in ds.sensitive_columns.split(",") if c.strip()]

    MAX_RETRIES = 2
    retry_history: list[dict] = []
    retry_count = 0
    sql = None
    llm_response = None
    validation = None
    exec_result = None
    exec_error = None

    conversation_history = await _build_conversation_history(db, conv.id)

    # 3.3 Table Selection（大 Schema 优化：只把相关表塞进 Prompt；安全校验仍用全量 schema_tables）
    selected_tables = await select_relevant_tables(
        question=question,
        schema_tables=schema_tables,
        max_tables=10,
    )

    # 3.8 查缓存（精确匹配优先，未命中再试语义相似度匹配）
    cached_sql = await get_cached_sql_with_semantic_fallback(ds.id, question)
    if cached_sql:
        cached_validation = validate_sql(
            sql=cached_sql,
            allowed_tables=allowed_table_names,
            allowed_columns=allowed_columns,
            sensitive_columns=sensitive_columns,
            db_type=ds.db_type,
        )
        if not cached_validation.is_safe:
            # 缓存的 SQL 不再安全（可能 Schema 变了），清缓存走正常链路
            await invalidate_cache(ds.id)
            cached_sql = None

    for attempt in range(MAX_RETRIES + 1):
        retry_count = attempt

        if cached_sql:
            sql = cached_sql
            validation = ValidationResult(is_safe=True)
        else:
            sql, llm_response = await generate_sql(
                question=question,
                schema_tables=selected_tables,
                db_type=ds.db_type,
                retry_history=retry_history,
                conversation_history=conversation_history,
            )

            validation = validate_sql(
                sql=sql,
                allowed_tables=allowed_table_names,
                allowed_columns=allowed_columns,
                sensitive_columns=sensitive_columns,
                db_type=ds.db_type,
            )

            if validation.is_safe:
                await set_cached_sql(ds.id, question, sql)

        if not validation.is_safe:
            exec_result = None
            break

        try:
            exec_result = execute_sql(ds, sql)
            exec_error = None
            break
        except SQLExecutionError as e:
            if cached_sql:
                # 缓存的 SQL 执行失败（可能 Schema 已变更），丢弃缓存，走正常生成链路重试
                logger.info("sql.cache.stale_execution_failure", error=str(e))
                await invalidate_cache(ds.id)
                cached_sql = None
                retry_history.append({"sql": sql, "error": str(e)})
                continue

            # ✅ 执行失败 → 尝试自修复
            logger.info("sql.execution_failed.attempting_retry", error=str(e))
            retry_result = await retry_with_feedback(
                question=question,
                failed_sql=sql,
                error_message=str(e),
                schema_tables=selected_tables,
                db_type=ds.db_type,
                conversation_history=conversation_history,
                allowed_tables=allowed_table_names,
                error_type=e.error_type,
            )
            if retry_result:
                sql, llm_response = retry_result
                exec_result = execute_sql(ds, sql)  # 用修正后的 SQL 重新执行
                await set_cached_sql(ds.id, question, sql)
            else:
                # 重试也失败了，返回错误信息
                exec_error = e.message
                exec_result = None
                retry_history.append({"sql": sql, "error": exec_error})
                logger.warning(
                    "sql.execution.retry",
                    attempt=attempt,
                    error=exec_error,
                    sql=sql[:200],
                )

    if not validation.is_safe:
        logger.warning(
            "sql.validation.blocked",
            level=validation.level,
            reason=validation.reason,
            sql=sql[:200],
        )

        # 存一条 assistant 消息告知用户
        block_message = f"抱歉，生成的查询未通过安全校验：{validation.reason}。请换一种方式描述您的问题。"
        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=block_message,
        )
        db.add(assistant_msg)
        await db.flush()

        # 存 QueryExecution（记录被拦截的 SQL）
        qe = QueryExecution(
            message_id=assistant_msg.id,
            datasource_id=ds.id,
            generated_sql=sql,
            status="blocked",
            error_message=f"[{validation.level}]{validation.reason}:{validation.blocked_detail}",
            retry_count=retry_count,
        )
        db.add(qe)
        await db.flush()

        return ChatResponse(
            message_id=assistant_msg.id,
            content=block_message,
            generated_sql=sql,
            query_result=None,
            blocked=True,
            block_reason=validation.reason,
            model=llm_response.model,
            usage={
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
            },
        )

    # ✅ 作业2：重试耗尽仍执行失败
    if exec_result is None:
        logger.warning(
            "sql.execution.exhausted",
            retries=retry_count,
            error=exec_error,
            sql=sql[:200],
        )

        fail_message = (
            f"抱歉，尝试 {retry_count + 1} 次后仍无法执行查询：{exec_error}。"
            "请换一种方式描述您的问题，或联系管理员确认数据源结构。"
        )
        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=fail_message,
        )
        db.add(assistant_msg)
        await db.flush()

        qe = QueryExecution(
            message_id=assistant_msg.id,
            datasource_id=ds.id,
            generated_sql=sql,
            status="failed",
            error_message=exec_error,
            retry_count=retry_count,
        )
        db.add(qe)
        await db.flush()

        return ChatResponse(
            message_id=assistant_msg.id,
            content=fail_message,
            generated_sql=sql,
            query_result=None,
            model=llm_response.model,
            usage={
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
            },
        )

    # 缓存命中且一次执行成功时，全程没有调用生成 SQL 的 LLM，llm_response 为 None
    model_name = llm_response.model if llm_response else "cache"
    usage_payload = (
        {
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "completion_tokens": llm_response.usage.completion_tokens,
        }
        if llm_response
        else {"prompt_tokens": 0, "completion_tokens": 0}
    )

    # ✅ 6. 升级：用 LLM 生成自然语言解释

    answer = await explain_result(
        question=question,
        sql=sql,
        rows=exec_result.rows,
        row_count=exec_result.row_count,
    )

    # 7. 存 assistant 消息
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.flush()

    # 8. 存 QueryExecution
    qe = QueryExecution(
        message_id=assistant_msg.id,
        datasource_id=ds.id,
        generated_sql=sql,
        status="success",
        result_summary=json.dumps(exec_result.rows[:5], ensure_ascii=False, default=str),
        row_count=exec_result.row_count,
        execution_ms=exec_result.execution_ms,
        retry_count=retry_count,
    )
    db.add(qe)

    # 触发 Conversation.updated_at 的 onupdate（新增消息也算“对话更新”）
    conv.updated_at = datetime.now(timezone.utc)
    await db.flush()

    logger.info(
        "chat.completed",
        conversation_id=conv.id,
        sql_length=len(sql),
        row_count=exec_result.row_count,
        execution_ms=exec_result.execution_ms,
        model=model_name,
    )

    return ChatResponse(
        message_id=assistant_msg.id,
        content=answer,
        generated_sql=sql,
        query_result=QueryResult(**exec_result.to_dict()),
        model=model_name,
        usage=usage_payload,
    )


async def list_messages(
    db: AsyncSession, user: User, conversation_id: int
) -> list[MessageResponse]:
    """返回一个对话下的全部消息（按时间升序），带上关联的 QueryExecution 信息。"""
    await get_conversation(db, user, conversation_id)  # 权限校验 + 404

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .options(selectinload(Message.query_execution))
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()

    return [
        MessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            generated_sql=m.query_execution.generated_sql if m.query_execution else None,
            execution_ms=m.query_execution.execution_ms if m.query_execution else None,
            row_count=m.query_execution.row_count if m.query_execution else None,
        )
        for m in messages
    ]


async def list_conversations(
    db: AsyncSession, user: User, cursor: str | None = None, limit: int = 20
) -> tuple[list[ConversationListItem], str | None]:
    """返回当前用户的对话列表（按 updated_at 降序），每条附带最后一条消息的前 100 字摘要。"""

    # 每个 conversation 的最新一条 message id
    last_message_subq = (
        select(Message.conversation_id, func.max(Message.id).label("last_message_id"))
        .group_by(Message.conversation_id)
        .subquery()
    )

    stmt = (
        select(Conversation, Message.content)
        .where(Conversation.user_id == user.id)
        .outerjoin(last_message_subq, last_message_subq.c.conversation_id == Conversation.id)
        .outerjoin(Message, Message.id == last_message_subq.c.last_message_id)
    )

    if cursor is not None:
        try:
            cursor_updated_at, cursor_id = decode_cursor(cursor)
        except (ValueError, UnicodeDecodeError) as e:
            raise ValidationError("无效的分页游标") from e
        stmt = stmt.where(
            tuple_(Conversation.updated_at, Conversation.id) < (cursor_updated_at, cursor_id)
        )

    stmt = stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit + 1)

    result = await db.execute(stmt)
    rows = result.all()

    has_more = len(rows) > limit
    rows = rows[:limit]

    items = [
        ConversationListItem(
            id=conv.id,
            datasource_id=conv.datasource_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            last_message_preview=content[:100] if content else None,
        )
        for conv, content in rows
    ]

    next_cursor = None
    if has_more and rows:
        last_conv = rows[-1][0]
        next_cursor = encode_cursor(last_conv.updated_at, last_conv.id)

    return items, next_cursor


def _format_answer(question: str, sql: str, result: dict) -> str:
    """把 SQL 执行结果格式化为自然语言回答。"""
    rows = result["rows"]
    if not rows:
        return f"查询已执行，但没有返回结果。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    if len(rows) == 1 and len(rows[0]) == 1:
        key = list(rows[0].keys())[0]
        value = rows[0][key]
        return f"查询结果为 **{value}**。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    row_count = result["row_count"]
    truncated = result.get("truncated", False)
    summary = f"查询返回了{row_count} 条结果"
    if truncated:
        summary += f"（已截取前{row_count} 条）"
    summary += f"。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    return summary

# app/services/chat.py 新增辅助函数

async def _build_conversation_history(
    db: AsyncSession, conversation_id: int, max_turns: int = 5
) -> list[dict]:
    """从数据库读取最近 N 轮对话，构建 LLM messages 格式的历史。"""

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(max_turns * 2 + 2)  # 多取一些，后面截断
    )
    result = await db.execute(stmt)
    messages = list(reversed(result.scalars().all()))

    # 排除最后一条（当前用户刚发的消息）
    if messages and messages[-1].role == "user":
        messages = messages[:-1]

    history = []
    for msg in messages:
        if msg.role == "user":
            history.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            # 尝试从 QueryExecution 获取 SQL，拼成更精确的上下文
            qe = await _get_query_execution_for_message(db, msg.id)
            if qe and qe.generated_sql:
                content = f"SQL:{qe.generated_sql}"
                if qe.result_summary:
                    content += f"\nResult:{qe.result_summary}"
            else:
                content = msg.content[:500]
            history.append({"role": "assistant", "content": content})

    return history[-max_turns * 2:]  # 只保留最近 N 轮


async def _get_query_execution_for_message(
    db: AsyncSession, message_id: int
) -> QueryExecution | None:
    stmt = select(QueryExecution).where(QueryExecution.message_id == message_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()