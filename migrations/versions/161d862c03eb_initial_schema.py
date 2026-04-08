"""initial_schema

Revision ID: 161d862c03eb
Revises:
Create Date: 2026-04-03 00:29:34.016690

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "161d862c03eb"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Initial schema is created by infrastructure/postgres/init/01_init_schemas.sql
    # on first container start. This migration marks that baseline in Alembic history.
    pass


def downgrade() -> None:
    pass
