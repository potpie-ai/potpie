"""Domain ports for the CLI auth subsystem.

Core-owned contracts that the auth adapters depend on *inward*: the credential
persistence surface (:mod:`credentials`). The inbound CLI commands type against
these ports; outbound adapters implement them; concretes are wired at the
composition root (``bootstrap/cli_auth_wiring.py``).

Transport plumbing (the httpx-typed ``HttpClient`` seam) and the auth error
taxonomy deliberately stay in ``adapters/outbound/cli_auth`` — they are
adapter-internal (no inbound use) and keeping them out of the domain avoids
leaking ``httpx`` into the core.
"""
