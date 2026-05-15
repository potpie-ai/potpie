"""Auto-join-on-invite + accept/decline lifecycle for context-graph pots.

Self-contained: an in-memory SQLite DB with just the four tables the pot
router touches, wired into a FastAPI ``TestClient`` via ``get_db`` override.
Deliberately independent of the Postgres ``db_session``/``client`` conftest
fixtures so this behavior is fast and runnable without a database.

Behavior under test (the invite model flip):

* inviting an *existing* user auto-creates their membership, so the pot is
  immediately visible in ``GET /pots`` — flagged with ``pending_invitation``;
* accepting keeps the membership and clears ``pending_invitation``;
* declining removes the membership (pot disappears) and marks the invite
  ``declined``;
* an owner revoking a pending invite also ejects the auto-added member.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import get_db
from app.modules.context_graph.context_graph_pot_invitation_model import (
    INVITATION_STATUS_ACCEPTED,
    INVITATION_STATUS_DECLINED,
    INVITATION_STATUS_PENDING,
    INVITATION_STATUS_REVOKED,
    ContextGraphPotInvitation,
)
from app.modules.context_graph.context_graph_pot_member_model import (
    ContextGraphPotMember,
)
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph import context_pot_routes
from app.modules.context_graph.context_pot_routes import make_pot_router
from app.modules.users.user_model import User

pytestmark = pytest.mark.integration


# SQLite has no JSONB; the only column that needs it is users.provider_info,
# which stays NULL here. Render it as the sqlite JSON affinity so the table
# can be created.
@compiles(JSONB, "sqlite")
def _jsonb_as_json_on_sqlite(element, compiler, **kw):  # noqa: ANN001, ANN201
    return "JSON"


OWNER_UID = "owner-uid"
OWNER_EMAIL = "owner@example.com"
INVITEE_UID = "invitee-uid"
INVITEE_EMAIL = "invitee@example.com"
STRANGER_UID = "stranger-uid"
STRANGER_EMAIL = "stranger@example.com"
# A person with a verified Firebase token but no users row yet (never ran
# through /signup) — the brand-new invitee arriving via the invite link.
NEWBIE_UID = "newbie-firebase-uid"
NEWBIE_EMAIL = "newbie@example.com"


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch):
    """An app + client backed by a fresh in-memory SQLite for each test."""
    # Never spawn the fire-and-forget email thread during tests.
    monkeypatch.setattr(
        context_pot_routes, "_dispatch_invitation_email", lambda **_: None
    )

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Only the tables the router actually touches, in FK-dependency order.
    for model in (
        User,
        ContextGraphPot,
        ContextGraphPotMember,
        ContextGraphPotInvitation,
    ):
        model.__table__.create(bind=engine)

    TestingSession = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )

    now = datetime.now(timezone.utc)
    with TestingSession() as s:
        s.add_all(
            [
                User(uid=OWNER_UID, email=OWNER_EMAIL),
                User(uid=INVITEE_UID, email=INVITEE_EMAIL),
                User(uid=STRANGER_UID, email=STRANGER_EMAIL),
            ]
        )
        pot_id = str(uuid.uuid4())
        s.add(
            ContextGraphPot(
                id=pot_id,
                user_id=OWNER_UID,
                created_by_user_id=OWNER_UID,
                slug="acme",
                created_at=now,
                updated_at=now,
            )
        )
        # Mirror create_context_pot: the owner is a member.
        s.add(
            ContextGraphPotMember(
                pot_id=pot_id, user_id=OWNER_UID, role="owner"
            )
        )
        s.commit()

    # Mirrors the verified-Firebase-token claims the v1 auth dep returns.
    caller = {"claims": {"user_id": OWNER_UID}}

    def auth_dep():
        return caller["claims"]

    def override_get_db():
        s = TestingSession()
        try:
            yield s
        finally:
            s.close()

    app = FastAPI()
    app.include_router(make_pot_router(auth_dep))
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)

    def as_user(uid: str) -> None:
        # An already-provisioned user (email resolved from the DB row).
        caller["claims"] = {"user_id": uid}

    def as_token(uid: str, email: str, name: str = "New Person") -> None:
        # A verified Firebase identity with NO users row yet — the brand-new
        # invitee arriving via the invite link.
        caller["claims"] = {
            "user_id": uid,
            "email": email,
            "name": name,
            "email_verified": True,
        }

    def session():
        return TestingSession()

    yield {
        "client": client,
        "as_user": as_user,
        "as_token": as_token,
        "session": session,
        "pot_id": pot_id,
    }


def _member(session, pot_id: str, uid: str) -> ContextGraphPotMember | None:
    with session() as s:
        return (
            s.query(ContextGraphPotMember)
            .filter(
                ContextGraphPotMember.pot_id == pot_id,
                ContextGraphPotMember.user_id == uid,
            )
            .first()
        )


def _invite(harness, email: str = INVITEE_EMAIL) -> dict:
    harness["as_user"](OWNER_UID)
    r = harness["client"].post(
        f"/pots/{harness['pot_id']}/invitations", json={"email": email}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_inviting_existing_user_auto_adds_member_and_pot_is_visible(harness):
    body = _invite(harness)
    assert body["status"] == INVITATION_STATUS_PENDING
    assert body["token"]

    # Membership exists immediately — before any accept.
    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is not None

    # And the pot now shows up for the invitee, flagged as a pending invite.
    harness["as_user"](INVITEE_UID)
    pots = harness["client"].get("/pots").json()
    assert len(pots) == 1
    pot = pots[0]
    assert pot["id"] == harness["pot_id"]
    assert pot["role"] == "user"
    pi = pot["pending_invitation"]
    assert pi is not None
    assert pi["status"] == INVITATION_STATUS_PENDING
    assert pi["token"] == body["token"]
    assert pi["role"] == "user"


def test_accept_keeps_membership_and_clears_pending_flag(harness):
    body = _invite(harness)

    harness["as_user"](INVITEE_UID)
    r = harness["client"].post(
        f"/pot-invitations/{body['token']}/accept"
    )
    assert r.status_code == 200, r.text
    assert r.json()["pot_id"] == harness["pot_id"]

    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is not None
    with harness["session"]() as s:
        inv = s.query(ContextGraphPotInvitation).one()
        assert inv.status == INVITATION_STATUS_ACCEPTED

    pots = harness["client"].get("/pots").json()
    assert pots[0]["pending_invitation"] is None


def test_decline_removes_member_and_marks_declined(harness):
    body = _invite(harness)

    harness["as_user"](INVITEE_UID)
    r = harness["client"].post(
        f"/pot-invitations/{body['token']}/decline"
    )
    assert r.status_code == 200, r.text
    assert r.json() == {
        "ok": True,
        "pot_id": harness["pot_id"],
        "declined": True,
    }

    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is None
    with harness["session"]() as s:
        inv = s.query(ContextGraphPotInvitation).one()
        assert inv.status == INVITATION_STATUS_DECLINED

    # Pot is gone from the invitee's list.
    assert harness["client"].get("/pots").json() == []


def test_decline_by_wrong_user_is_forbidden_and_keeps_member(harness):
    body = _invite(harness)

    harness["as_user"](STRANGER_UID)
    r = harness["client"].post(
        f"/pot-invitations/{body['token']}/decline"
    )
    assert r.status_code == 403
    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is not None


def test_owner_revoke_also_ejects_auto_added_member(harness):
    body = _invite(harness)
    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is not None

    harness["as_user"](OWNER_UID)
    r = harness["client"].delete(
        f"/pots/{harness['pot_id']}/invitations/{body['id']}"
    )
    assert r.status_code == 200, r.text

    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is None
    with harness["session"]() as s:
        inv = s.query(ContextGraphPotInvitation).one()
        assert inv.status == INVITATION_STATUS_REVOKED

    harness["as_user"](INVITEE_UID)
    assert harness["client"].get("/pots").json() == []


def _user(session, uid: str) -> User | None:
    with session() as s:
        return s.query(User).filter(User.uid == uid).first()


def _signup(session, uid: str, email: str) -> None:
    """Simulate the frontend `/api/v1/signup` provisioning a users row."""
    with session() as s:
        s.add(User(uid=uid, email=email))
        s.commit()


def test_accept_without_an_account_asks_to_sign_up_first(harness):
    """Invitee authenticates via the link but never ran through FE signup.

    Account creation is the frontend's job — the backend must not provision.
    It returns a distinct 401 the join page can act on, and creates nothing.
    """
    body = _invite(harness, email=NEWBIE_EMAIL)
    assert _user(harness["session"], NEWBIE_UID) is None

    harness["as_token"](NEWBIE_UID, NEWBIE_EMAIL)
    r = harness["client"].post(f"/pot-invitations/{body['token']}/accept")
    assert r.status_code == 401
    assert "sign up" in r.json()["detail"].lower()

    assert _user(harness["session"], NEWBIE_UID) is None
    assert _member(harness["session"], harness["pot_id"], NEWBIE_UID) is None


def test_signup_then_link_matches_by_email_and_grants_access(harness):
    """The intended flow: sign up first (FE), then the link resolves."""
    body = _invite(harness, email=NEWBIE_EMAIL)

    # Frontend signs the user up; now an account exists.
    _signup(harness["session"], NEWBIE_UID, NEWBIE_EMAIL)

    harness["as_token"](NEWBIE_UID, NEWBIE_EMAIL)
    r = harness["client"].post(f"/pot-invitations/{body['token']}/accept")
    assert r.status_code == 200, r.text

    assert _member(harness["session"], harness["pot_id"], NEWBIE_UID) is not None
    pots = harness["client"].get("/pots").json()
    assert [p["id"] for p in pots] == [harness["pot_id"]]
    assert pots[0]["pending_invitation"] is None


def test_accept_with_existing_account_but_mismatched_email_is_forbidden(
    harness,
):
    body = _invite(harness, email=NEWBIE_EMAIL)

    # STRANGER has an account, but it is not the invited address.
    harness["as_user"](STRANGER_UID)
    r = harness["client"].post(f"/pot-invitations/{body['token']}/accept")
    assert r.status_code == 403
    assert _member(harness["session"], harness["pot_id"], STRANGER_UID) is None


def test_decline_without_an_account_asks_to_sign_up_first(harness):
    body = _invite(harness, email=NEWBIE_EMAIL)

    harness["as_token"](NEWBIE_UID, NEWBIE_EMAIL)
    r = harness["client"].post(f"/pot-invitations/{body['token']}/decline")
    assert r.status_code == 401
    assert "sign up" in r.json()["detail"].lower()
    assert _user(harness["session"], NEWBIE_UID) is None


def test_declined_invite_does_not_block_a_fresh_reinvite(harness):
    first = _invite(harness)
    harness["as_user"](INVITEE_UID)
    harness["client"].post(f"/pot-invitations/{first['token']}/decline")

    # A brand-new pending invite to the same email must be allowed (the
    # partial unique index only covers status='pending').
    second = _invite(harness)
    assert second["token"] != first["token"]
    assert _member(harness["session"], harness["pot_id"], INVITEE_UID) is not None
