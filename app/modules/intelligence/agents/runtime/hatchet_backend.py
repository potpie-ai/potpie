"""Hatchet agent backend — enqueue side (used by the API / conversation routing).

Kept lightweight on purpose: the worker-side execution lives in
``app/modules/intelligence/agents/hatchet_worker.py`` so importing this from the API
does not pull in agent/ML/worker dependencies. Enqueue is event-driven
(``hatchet.event.push``); the worker registers ``on_events=[EVENT_AGENT_RUN]``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EVENT_AGENT_RUN = "potpie.agent.run"
AGENT_RUN_OPERATION_MESSAGE = "message"
AGENT_RUN_OPERATION_REGENERATE = "regenerate"
AgentRunOperation = Literal["message", "regenerate"]


class AgentRunInput(BaseModel):
    """Validated payload for a Hatchet agent run.

    ``operation`` lets all agent-backed work for an allowlisted agent route through
    the same Hatchet event without creating a separate queue for every operation.
    """

    model_config = ConfigDict(extra="allow")

    conversation_id: str
    run_id: str
    user_id: str
    query: str = ""
    agent_id: str
    operation: AgentRunOperation = AGENT_RUN_OPERATION_MESSAGE
    node_ids: Optional[List[Any]] = None
    attachment_ids: List[str] = Field(default_factory=list)
    local_mode: bool = False
    tunnel_url: Optional[str] = None


def prepare_hatchet_client_env() -> None:
    """Default to insecure gRPC for self-hosted Hatchet when the server URL is http.

    Mirrors the context-engine helper but kept local so the agent runtime does not
    depend on the context-engine package being on sys.path.
    """
    if os.getenv("HATCHET_CLIENT_TLS_STRATEGY"):
        return
    url = (os.getenv("HATCHET_CLIENT_SERVER_URL") or "").strip()
    if url.startswith("http://"):
        os.environ["HATCHET_CLIENT_TLS_STRATEGY"] = "none"


@lru_cache(maxsize=1)
def get_hatchet_client() -> Any:
    """Build and cache the Hatchet client (raises if the SDK/token is unavailable)."""
    prepare_hatchet_client_env()
    from hatchet_sdk import Hatchet

    return Hatchet()


def enqueue_agent_run(agent_input: AgentRunInput, *, client: Any = None) -> None:
    """Push an agent-run event to Hatchet. Raises on any client/transport failure
    so the caller can fail closed (HTTP 503)."""
    hatchet = client if client is not None else get_hatchet_client()
    hatchet.event.push(EVENT_AGENT_RUN, agent_input.model_dump(mode="json"))
