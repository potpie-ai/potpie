# Implementation Plans

This directory contains concise implementation plans for features and infrastructure changes.

## Plans Overview

### [Context Graph Architecture Plan](./context-graph-architecture-plan.md)
**Priority:** High | **Scope:** Backend intelligence architecture

**Focus areas**: Graphiti as-is for context store; ETL + live triggers; `group_id` project isolation; `get_project_context` agent tool; Postgres sync state + ingestion log.

**Implementation:** [Context Graph Implementation Plan (Phase 1)](./context-graph-implementation-plan.md) — step-by-step tasks, file paths, and checklist.

---

### [Celery Scaling — Unified Plan (Potpie + Workflows)](./celery-scaling-unified-plan.md)
**Priority:** Critical | **Scope:** Both apps together

**Phases**:
1. Prefork + DB pool fix (this week)
2. Sync/async in API path
3. DB session leaks + unbounded queries
4. Sync Redis in async paths
5. Queue splitting + event queue rename (`potpie-external-event` / `workflows-external-event`)
6. Chat history, streaming & memory

**Source:** Merged from `celery-scaling-plan.docx` + implementation plan + workflows integration

---

### [Celery Prefork & DB Pool Plan](./celery-prefork-and-db-pool-plan.md)
**Priority:** Critical | **Estimated Time**: 1–2 hours

**Tasks**:
- Env-driven DB pool (pool_size=4, max_overflow=6) to avoid Postgres connection exhaustion
- worker_process_init: dispose engine + LiteLLM config for fork safety
- Switch Celery to prefork + concurrency=4 (4 concurrent users per pod)

**Files Modified**: `database.py`, `celery_app.py`, `start.sh`, `scripts/start.sh`

---

### 1. [Backend Plan](./backend-plan.md)
**Estimated Time**: 4-6 hours

**Tasks**:
- Store workspace_id in TunnelService
- Add GET /tunnels/workspaces endpoint
- Add POST /conversations/{id}/apply-changes endpoint
- Error handling and validation

**Files Modified**:
- `tunnel_service.py`
- `tunnel_router.py`
- `conversations_router.py`

---

### 2. [Extension Plan](./extension-plan.md)
**Estimated Time**: 2-3 hours

**Tasks**:
- Generate workspace_id from workspace path + machine ID
- Include workspace_id in tunnel registration
- Handle workspace changes

**Files Modified**:
- `TunnelManager.ts`

---

### 3. [Frontend Plan](./frontend-plan.md)
**Estimated Time**: 4-5 hours

**Tasks**:
- Create TunnelService API client
- Create connection status component
- Create apply button component
- Integrate into chat UI

**Files Created**:
- `TunnelService.ts`
- `VSCodeConnectionStatus.tsx`
- `ApplyLocallyButton.tsx`
- `WorkspaceSelector.tsx` (optional)

---

### 4. [Testing Plan](./testing-plan.md)
**Estimated Time**: 6-8 hours

**Tasks**:
- Unit tests (backend, extension, frontend)
- Integration tests (end-to-end flows)
- Edge case tests
- Manual testing checklist
- Performance tests

---

## Implementation Order

1. **Backend** (must be first - other components depend on it)
2. **Extension** (can be done in parallel with frontend)
3. **Frontend** (can be done in parallel with extension)
4. **Testing** (after all implementation complete)

---

## Quick Reference

### Key Endpoints
- `GET /api/v1/tunnels/workspaces` - List workspaces
- `GET /api/v1/tunnels/status` - Check connection
- `POST /api/v1/conversations/{id}/apply-changes` - Apply changes

### Key Components
- `TunnelService` - API client
- `VSCodeConnectionStatus` - Connection status UI
- `ApplyLocallyButton` - Apply button UI

### Key Files
- Backend: `tunnel_service.py`, `tunnel_router.py`, `conversations_router.py`
- Extension: `TunnelManager.ts`
- Frontend: `TunnelService.ts`, `VSCodeConnectionStatus.tsx`, `ApplyLocallyButton.tsx`

---

## Dependencies

### Backend → Extension
- Extension needs backend endpoints to register workspace_id

### Frontend → Backend
- Frontend needs backend endpoints for status/apply

### Testing → All
- Testing requires all components implemented

---

## Status Tracking

Use this checklist to track implementation progress:

### Backend
- [ ] Workspace ID storage
- [ ] Workspace listing endpoint
- [ ] Apply changes endpoint
- [ ] Error handling

### Extension
- [ ] Workspace ID generation
- [ ] Registration with workspace_id
- [ ] Workspace change handling

### Frontend
- [ ] TunnelService
- [ ] Connection status component
- [ ] Apply button component
- [ ] Chat UI integration

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Edge case tests
- [ ] Manual testing

---

## Related Documentation

- [Full Specification](../apply-to-local-workspace.md) - Complete feature spec
- [Review Document](../apply-to-local-workspace-review.md) - Implementation review with edge cases

---

## Questions?

Refer to the [Review Document](../apply-to-local-workspace-review.md) for:
- Edge cases and failure scenarios
- Design decisions
- Open questions
- Recommendations
