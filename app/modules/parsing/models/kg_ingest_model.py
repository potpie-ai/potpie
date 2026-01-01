import uuid

from sqlalchemy import Column, ForeignKey, String, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.base_model import Base


class KgIngestRun(Base):
    __tablename__ = "kg_ingest_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id = Column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    commit_id = Column(String(255), nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True), default=func.now(), nullable=False, index=True
    )
    status = Column(String(50), nullable=False)


class KgArtifactRecord(Base):
    __tablename__ = "kg_artifact_records"

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("kg_ingest_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    kind = Column(String(10), primary_key=True)
    record_key = Column(Text, primary_key=True)
    record_hash = Column(String(64), nullable=False, index=True)
    record_json = Column(JSONB, nullable=False)


class KgLatestSuccessfulRun(Base):
    __tablename__ = "kg_latest_successful_run"

    repo_id = Column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    user_id = Column(
        String(255),
        ForeignKey("users.uid", ondelete="CASCADE"),
        primary_key=True,
    )
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("kg_ingest_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
