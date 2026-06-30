# potpie-integrations

External integrations for Potpie: OAuth providers, `project_sources`, Linear sync, and HTTP routers.

Layout mirrors `context-engine` (hexagonal):

- `integrations/domain/` — registry, provider definitions, shared schemas
- `integrations/application/` — services, provider bootstrap
- `integrations/adapters/outbound/` — persistence models, OAuth clients, Linear GraphQL, crypto
- `integrations/adapters/inbound/http/` — FastAPI routers mounted by the main app

The installable Python package root is `integrations` (not top-level `domain`/`application`) so it does not collide with the `context-engine` editable install.
