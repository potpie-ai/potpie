from __future__ import annotations

from sqlalchemy.orm import Session

from app.modules.users.user_model import User
from seed.schemas import UserSeed


def apply_user(
    db: Session,
    _seed: UserSeed,
    *,
    target_email: str,
    resolved_uid: str,
) -> None:
    """Verify the target account exists; seed does not create users.

    ``user.json`` is validated by Pydantic before apply; its values are not
    written over the existing ``users`` row.
    """
    existing = db.query(User).filter(User.email == target_email).first()
    if not existing:
        raise ValueError(
            f"No user with email {target_email!r}. Create the account first; seed does not create users."
        )
    if existing.uid != resolved_uid:
        raise ValueError(
            f"User email {target_email!r} has uid {existing.uid!r}, "
            f"expected {resolved_uid!r}."
        )
