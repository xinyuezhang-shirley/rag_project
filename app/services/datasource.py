from sqlalchemy import select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import create_engine

from app.models.datasource import DataSource
from app.models.user import User
from app.schemas.datasource import DataSourceCreate, DataSourceUpdate
from app.core.security import encrypt_value, decrypt_value
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.core.pagination import encode_cursor, decode_cursor

logger = get_logger(__name__)


def _build_sync_url(
    db_type: str, host: str, port: int, database_name: str, username: str, plain_password: str
) -> str:
    """构造同步连接 URL（用于测试连接和 Schema 自省）。"""
    driver = {"postgresql": "postgresql+psycopg2", "mysql": "mysql+pymysql"}
    prefix = driver.get(db_type, db_type)
    return f"{prefix}://{username}:{plain_password}@{host}:{port}/{database_name}"


def _attempt_connection(
    db_type: str, host: str, port: int, database_name: str, username: str, plain_password: str
) -> tuple[bool, str]:
    """实际尝试连接数据库，返回 (是否成功, 消息)。"""
    url = _build_sync_url(db_type, host, port, database_name, username, plain_password)
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True, "连接成功"
    except Exception as e:
        return False, f"连接失败:{str(e)}"


async def create_datasource(
    db: AsyncSession, user: User, req: DataSourceCreate
) -> DataSource:
    ds = DataSource(
        user_id=user.id,
        name=req.name,
        db_type=req.db_type,
        host=req.host,
        port=req.port,
        database_name=req.database_name,
        username=req.username,
        encrypted_password=encrypt_value(req.password),
        sensitive_columns=",".join(req.sensitive_columns) if req.sensitive_columns else None,
    )
    db.add(ds)
    await db.flush()

    logger.info("datasource.created", datasource_id=ds.id, user_id=user.id, db_type=ds.db_type)
    return ds


async def list_datasources(
    db: AsyncSession, user: User, cursor: str | None = None, limit: int = 20
) -> tuple[list[DataSource], str | None]:
    stmt = select(DataSource).where(
        DataSource.user_id == user.id, DataSource.is_active == True
    )

    if cursor is not None:
        try:
            cursor_created_at, cursor_id = decode_cursor(cursor)
        except (ValueError, UnicodeDecodeError) as e:
            raise ValidationError("无效的分页游标") from e
        stmt = stmt.where(
            tuple_(DataSource.created_at, DataSource.id) < (cursor_created_at, cursor_id)
        )

    stmt = stmt.order_by(DataSource.created_at.desc(), DataSource.id.desc()).limit(limit + 1)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    items = rows[:limit]

    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = encode_cursor(last.created_at, last.id)

    return items, next_cursor


async def get_datasource(db: AsyncSession, user: User, datasource_id: int) -> DataSource:
    stmt = select(DataSource).where(
        DataSource.id == datasource_id,
        DataSource.user_id == user.id,
        DataSource.is_active == True,
    )
    result = await db.execute(stmt)
    ds = result.scalar_one_or_none()
    if not ds:
        raise NotFoundError("数据源", datasource_id)
    return ds


async def delete_datasource(db: AsyncSession, user: User, datasource_id: int) -> None:
    ds = await get_datasource(db, user, datasource_id)
    ds.is_active = False
    await db.flush()
    logger.info("datasource.deleted", datasource_id=ds.id, user_id=user.id)


async def test_connection(db: AsyncSession, user: User, datasource_id: int) -> tuple[bool, str]:
    ds = await get_datasource(db, user, datasource_id)
    plain_password = decrypt_value(ds.encrypted_password)
    success, message = _attempt_connection(
        ds.db_type, ds.host, ds.port, ds.database_name, ds.username, plain_password
    )
    if not success:
        logger.warning("datasource.connection_failed", datasource_id=ds.id, error=message)
    return success, message


async def update_datasource(
    db: AsyncSession, user: User, datasource_id: int, req: DataSourceUpdate
) -> DataSource:
    ds = await get_datasource(db, user, datasource_id)
    fields_set = req.model_fields_set

    connection_fields = {"host", "port", "database_name", "username", "password"}
    changed_connection_fields = fields_set & connection_fields

    if changed_connection_fields:
        candidate_password = (
            req.password if "password" in fields_set else decrypt_value(ds.encrypted_password)
        )
        success, message = _attempt_connection(
            ds.db_type,
            req.host if "host" in fields_set else ds.host,
            req.port if "port" in fields_set else ds.port,
            req.database_name if "database_name" in fields_set else ds.database_name,
            req.username if "username" in fields_set else ds.username,
            candidate_password,
        )
        if not success:
            raise ValidationError(f"连接验证失败，未保存更改:{message}")

        if "host" in fields_set:
            ds.host = req.host
        if "port" in fields_set:
            ds.port = req.port
        if "database_name" in fields_set:
            ds.database_name = req.database_name
        if "username" in fields_set:
            ds.username = req.username
        if "password" in fields_set:
            ds.encrypted_password = encrypt_value(req.password)

    if "name" in fields_set:
        ds.name = req.name

    if "sensitive_columns" in fields_set:
        ds.sensitive_columns = ",".join(req.sensitive_columns) if req.sensitive_columns else None

    await db.flush()
    logger.info("datasource.updated", datasource_id=ds.id, user_id=user.id, fields=list(fields_set))
    return ds