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
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.pagination import encode_cursor, decode_cursor
from app.sql_engine.validator import validate_sql, ValidationResult
from app.sql_engine.explainer import explain_result

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

    # 4. 生成 SQL
    sql, llm_response = await generate_sql(
        question=question,
        schema_tables=schema_tables,
        db_type=ds.db_type,
    )
    # ✅ 4.5 新增：SQL 安全校验

    allowed_table_names = [t["table_name"] for t in schema_tables]
    validation = validate_sql(
        sql=sql,
        allowed_tables=allowed_table_names,
        db_type=ds.db_type,
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


    # 5. 执行 SQL
    exec_result = execute_sql(ds, sql)

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
        model=llm_response.model,
    )

    return ChatResponse(
        message_id=assistant_msg.id,
        content=answer,
        generated_sql=sql,
        query_result=QueryResult(**exec_result.to_dict()),
        model=llm_response.model,
        usage={
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "completion_tokens": llm_response.usage.completion_tokens,
        },
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