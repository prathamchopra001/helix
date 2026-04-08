"""initial_schema

Revision ID: 161d862c03eb
Revises: 
Create Date: 2026-04-03 00:29:34.016690

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '161d862c03eb'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Initial schema is created by infrastructure/postgres/init/01_init_schemas.sql
    # on first container start. This migration marks that baseline in Alembic history.
    pass


def downgrade() -> None:
    pass
