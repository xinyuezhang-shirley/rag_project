"""restore chunks embedding column and hnsw index

Revision ID: a689da93ec23
Revises: 2533b5e66df3
Create Date: 2026-10-24 13:53:49.296264

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = 'a689da93ec23'
down_revision: Union[str, Sequence[str], None] = '2533b5e66df3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 2533b5e66df3 曾经把 embedding 列和 HNSW 索引删掉了（autogenerate 的误判，
    # 因为当时 Chunk 模型里 embedding 还被注释掉）。这里把它们加回来。
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "chunks",
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True),
    )
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    op.drop_column("chunks", "embedding")
