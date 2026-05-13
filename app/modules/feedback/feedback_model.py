"""SQLAlchemy model for the Potpie vs Copilot blind feedback votes."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.core.base_model import Base


class FeedbackVote(Base):
    __tablename__ = "feedback_votes"
    __table_args__ = (
        # Same email cannot have two rows for the same comparison.
        # Updates are handled via upsert in the service layer.
        UniqueConstraint(
            "voter_email", "comparison_id", name="uq_feedback_votes_email_comparison"
        ),
    )

    id = Column(String(64), primary_key=True)

    # Which Q&A pair this vote is about (matches an entry in comparison_data.py).
    comparison_id = Column(String(128), nullable=False, index=True)

    # Server-decoded choice: "potpie" | "copilot" | "tie" | "neither".
    chosen_model = Column(String(16), nullable=False)

    # The randomized position the voter actually clicked: "a" | "b" | "tie" | "neither".
    chosen_position = Column(String(8), nullable=False)

    # Server-side seed used to randomize which response was shown as A vs B.
    # Never sent to the voter; stored for audit/reconciliation.
    presentation_seed = Column(Integer, nullable=False)

    # Voter identity captured on the page (no auth).
    voter_name = Column(String(255), nullable=False)
    voter_email = Column(String(255), nullable=False, index=True)
    voter_role = Column(String(64), nullable=True)

    # Optional quality signals.
    confidence = Column(Integer, nullable=True)  # 1-5
    reason_tags = Column(JSONB, nullable=True)  # list[str]
    comment = Column(Text, nullable=True)
    time_on_page_ms = Column(Integer, nullable=True)

    # Browser-session identifier from localStorage (uuid).
    session_id = Column(String(64), nullable=False, index=True)
    client_user_agent = Column(Text, nullable=True)

    submitted_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
