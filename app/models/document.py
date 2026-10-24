from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Integer, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(20))  # pdf / txt / md / docx
    file_size: Mapped[int] = mapped_column(Integer)      # bytes
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending → processing → completed → failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    total_characters: Mapped[int] = mapped_column(Integer, default=0)

    # 关系
    owner: Mapped["User"] = relationship()
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )



class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    # HNSW 向量索引声明在模型里，否则 alembic --autogenerate 看不到它，
    # 会判定“数据库里有、模型里没有”从而生成 DROP INDEX（之前就踩过这个坑）
    __table_args__ = (
        Index(
            "idx_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)  # 在文档中的顺序
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # ✅ Embedding 向量（1536 维，对应 text-embedding-3-small）
    embedding = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 关系
    document: Mapped["Document"] = relationship(back_populates="chunks")