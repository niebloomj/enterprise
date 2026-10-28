"""Create user-scoped encrypted OpenAI subscription state.

Revision ID: 204
Revises: 203
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '204'
down_revision: str | None = '203'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'v1_remote_sandbox', sa.Column('workspace_id', sa.String(), nullable=True)
    )
    op.add_column(
        'v1_remote_sandbox',
        sa.Column(
            'subscription_broker_ready',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_table(
        'openai_subscription',
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('credentials', sa.String(), nullable=True),
        sa.Column('pending_login', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id'),
    )


def downgrade() -> None:
    op.drop_table('openai_subscription')
    op.drop_column('v1_remote_sandbox', 'subscription_broker_ready')
    op.drop_column('v1_remote_sandbox', 'workspace_id')
