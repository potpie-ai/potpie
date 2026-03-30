"""Pot-scoped ledger: pot_id + provider-scoped repo identity

Revision ID: 20260330_pot_scoped_ledger
Revises: 20260324_bridge_status_cols
Create Date: 2026-03-30 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260330_pot_scoped_ledger"
down_revision: Union[str, None] = "20260324_bridge_status_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("context_sync_state", "context_ingestion_log", "raw_events"):
        op.add_column(table, sa.Column("pot_id", sa.Text(), nullable=True))
        op.add_column(
            table,
            sa.Column("provider", sa.String(length=64), nullable=True),
        )
        op.add_column(
            table,
            sa.Column("provider_host", sa.String(length=255), nullable=True),
        )
        op.add_column(table, sa.Column("repo_name", sa.Text(), nullable=True))

    op.execute(
        """
        UPDATE context_sync_state AS c
        SET pot_id = c.project_id,
            provider = 'github',
            provider_host = 'github.com',
            repo_name = COALESCE((SELECT p.repo_name FROM projects p WHERE p.id = c.project_id), '')
        """
    )
    op.execute(
        """
        UPDATE context_ingestion_log AS c
        SET pot_id = c.project_id,
            provider = 'github',
            provider_host = 'github.com',
            repo_name = COALESCE((SELECT p.repo_name FROM projects p WHERE p.id = c.project_id), '')
        """
    )
    op.execute(
        """
        UPDATE raw_events AS c
        SET pot_id = c.project_id,
            provider = 'github',
            provider_host = 'github.com',
            repo_name = COALESCE((SELECT p.repo_name FROM projects p WHERE p.id = c.project_id), '')
        """
    )

    op.alter_column("context_sync_state", "pot_id", existing_type=sa.Text(), nullable=False)
    op.alter_column("context_sync_state", "provider", existing_type=sa.String(length=64), nullable=False)
    op.alter_column(
        "context_sync_state", "provider_host", existing_type=sa.String(length=255), nullable=False
    )
    op.alter_column("context_sync_state", "repo_name", existing_type=sa.Text(), nullable=False)

    op.alter_column("context_ingestion_log", "pot_id", existing_type=sa.Text(), nullable=False)
    op.alter_column("context_ingestion_log", "provider", existing_type=sa.String(length=64), nullable=False)
    op.alter_column(
        "context_ingestion_log", "provider_host", existing_type=sa.String(length=255), nullable=False
    )
    op.alter_column("context_ingestion_log", "repo_name", existing_type=sa.Text(), nullable=False)

    op.alter_column("raw_events", "pot_id", existing_type=sa.Text(), nullable=False)
    op.alter_column("raw_events", "provider", existing_type=sa.String(length=64), nullable=False)
    op.alter_column("raw_events", "provider_host", existing_type=sa.String(length=255), nullable=False)
    op.alter_column("raw_events", "repo_name", existing_type=sa.Text(), nullable=False)

    op.drop_constraint("uq_context_sync_project_source", "context_sync_state", type_="unique")
    op.drop_index(op.f("ix_context_sync_state_project_id"), table_name="context_sync_state")
    op.drop_column("context_sync_state", "project_id")

    op.create_index(op.f("ix_context_sync_state_pot_id"), "context_sync_state", ["pot_id"], unique=False)
    op.create_unique_constraint(
        "uq_context_sync_pot_repo_source",
        "context_sync_state",
        ["pot_id", "provider", "provider_host", "repo_name", "source_type"],
    )

    op.drop_constraint("uq_context_ingestion_project_source_id", "context_ingestion_log", type_="unique")
    op.drop_index("ix_context_ingestion_project_source_id", table_name="context_ingestion_log")
    op.drop_index(op.f("ix_context_ingestion_log_project_id"), table_name="context_ingestion_log")
    op.drop_column("context_ingestion_log", "project_id")

    op.create_index(
        op.f("ix_context_ingestion_log_pot_id"), "context_ingestion_log", ["pot_id"], unique=False
    )
    op.create_index(
        "ix_context_ingestion_pot_repo_source_id",
        "context_ingestion_log",
        ["pot_id", "provider", "provider_host", "repo_name", "source_type", "source_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_context_ingestion_pot_repo_source_id",
        "context_ingestion_log",
        ["pot_id", "provider", "provider_host", "repo_name", "source_type", "source_id"],
    )

    op.drop_constraint("uq_raw_events_source", "raw_events", type_="unique")
    op.drop_index("ix_raw_events_project_source_id", table_name="raw_events")
    op.drop_index(op.f("ix_raw_events_project_id"), table_name="raw_events")
    op.drop_column("raw_events", "project_id")

    op.create_index(op.f("ix_raw_events_pot_id"), "raw_events", ["pot_id"], unique=False)
    op.create_index(
        "ix_raw_events_pot_repo_source_id",
        "raw_events",
        ["pot_id", "provider", "provider_host", "repo_name", "source_type", "source_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_raw_events_pot_repo_source",
        "raw_events",
        ["pot_id", "provider", "provider_host", "repo_name", "source_type", "source_id"],
    )


def downgrade() -> None:
    for table in ("context_sync_state", "context_ingestion_log", "raw_events"):
        op.add_column(table, sa.Column("project_id", sa.Text(), nullable=True))

    op.execute("UPDATE context_sync_state SET project_id = pot_id")
    op.execute("UPDATE context_ingestion_log SET project_id = pot_id")
    op.execute("UPDATE raw_events SET project_id = pot_id")

    op.drop_constraint("uq_context_sync_pot_repo_source", "context_sync_state", type_="unique")
    op.drop_index(op.f("ix_context_sync_state_pot_id"), table_name="context_sync_state")

    for col in ("repo_name", "provider_host", "provider", "pot_id"):
        op.drop_column("context_sync_state", col)

    op.alter_column("context_sync_state", "project_id", existing_type=sa.Text(), nullable=False)
    op.create_index(
        op.f("ix_context_sync_state_project_id"), "context_sync_state", ["project_id"], unique=False
    )
    op.create_unique_constraint(
        "uq_context_sync_project_source",
        "context_sync_state",
        ["project_id", "source_type"],
    )

    op.drop_constraint("uq_context_ingestion_pot_repo_source_id", "context_ingestion_log", type_="unique")
    op.drop_index("ix_context_ingestion_pot_repo_source_id", table_name="context_ingestion_log")
    op.drop_index(op.f("ix_context_ingestion_log_pot_id"), table_name="context_ingestion_log")

    for col in ("repo_name", "provider_host", "provider", "pot_id"):
        op.drop_column("context_ingestion_log", col)

    op.alter_column("context_ingestion_log", "project_id", existing_type=sa.Text(), nullable=False)
    op.create_index(
        op.f("ix_context_ingestion_log_project_id"), "context_ingestion_log", ["project_id"], unique=False
    )
    op.create_index(
        "ix_context_ingestion_project_source_id",
        "context_ingestion_log",
        ["project_id", "source_type", "source_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_context_ingestion_project_source_id",
        "context_ingestion_log",
        ["project_id", "source_type", "source_id"],
    )

    op.drop_constraint("uq_raw_events_pot_repo_source", "raw_events", type_="unique")
    op.drop_index("ix_raw_events_pot_repo_source_id", table_name="raw_events")
    op.drop_index(op.f("ix_raw_events_pot_id"), table_name="raw_events")

    for col in ("repo_name", "provider_host", "provider", "pot_id"):
        op.drop_column("raw_events", col)

    op.alter_column("raw_events", "project_id", existing_type=sa.Text(), nullable=False)
    op.create_index(op.f("ix_raw_events_project_id"), "raw_events", ["project_id"], unique=False)
    op.create_index(
        "ix_raw_events_project_source_id",
        "raw_events",
        ["project_id", "source_type", "source_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_raw_events_source",
        "raw_events",
        ["project_id", "source_type", "source_id"],
    )
