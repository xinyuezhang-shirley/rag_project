"""add sensitive_columns and retry_count

Revision ID: 061ae7ee82e0
Revises: aeb876dad940
Create Date: 2026-07-31 17:59:16.335612

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '061ae7ee82e0'
down_revision: Union[str, Sequence[str], None] = 'aeb876dad940'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 注：autogenerate 还检测到 demo_orders 表（课堂演示表，不属于任何 SQLAlchemy
    # 模型）想要 drop，与本次改动无关，已手动去掉，避免误删演示数据。
    op.add_column('datasources', sa.Column('sensitive_columns', sa.Text(), nullable=True))
    op.add_column(
        'query_executions',
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('query_executions', 'retry_count')
    op.drop_column('datasources', 'sensitive_columns')
