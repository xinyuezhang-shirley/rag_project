"""add meta jsonb column to messages

Revision ID: bcb377e404bd
Revises: a689da93ec23
Create Date: 2026-10-24 14:15:51.509178

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'bcb377e404bd'
down_revision: Union[str, Sequence[str], None] = 'a689da93ec23'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "messages",
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("messages", "meta")
