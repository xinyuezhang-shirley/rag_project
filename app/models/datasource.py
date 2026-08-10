from typing import Optional

from sqlalchemy import String, Integer, Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class DataSource(TimestampMixin, Base):
    __tablename__ = "datasources"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    db_type: Mapped[str] = mapped_column(String(20))  # postgresql / mysql
    host: Mapped[str] = mapped_column(String(255))
    port: Mapped[int] = mapped_column(Integer)
    database_name: Mapped[str] = mapped_column(String(255))
    username: Mapped[str] = mapped_column(String(255))
    encrypted_password: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # 逗号分隔的敏感列名列表（per-DataSource 覆盖/追加全局 SQL_SENSITIVE_COLUMNS 配置）
    sensitive_columns: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 关系
    owner: Mapped["User"] = relationship(back_populates="datasources")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="datasource")