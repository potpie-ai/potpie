# Context Engine docs

**Architecture version:** Phase 6 (post-cleanup). Last reviewed 2026-05-12.

Five docs, one job each. Read in order if you're new.

| Doc | What it answers |
|---|---|
| [`vision.md`](./vision.md) | What the Context Engine is, why it exists, the principles that bind every other decision. |
| [`architecture.md`](./architecture.md) | How the system is built today, with file paths into the code. |
| [`agent-contract.md`](./agent-contract.md) | The four-tool agent port — request and response contracts. The single source of truth for any agent or skill that consumes the engine. |
| [`extending.md`](./extending.md) | How to add a source, a reader, a record type, an intent — without touching the application layer. |
| [`plan.md`](./plan.md) | The rebuild plan. Phased, sequential, with discovery first in every phase. |

Code lives at [`app/src/context-engine/`](../../app/src/context-engine/).
The canonical ontology is in [`app/src/context-engine/domain/ontology.py`](../../app/src/context-engine/domain/ontology.py) — code is the source of truth; docs link in. The catalog functions at the bottom of that module (`get_canonical_node_labels`, `get_canonical_edge_types`, `is_allowed_edge`, `validate_required_properties`, `validate_lifecycle_status`) are the API every other doc and adapter consumes.

These five files plus `plan.md` are the entire doc set. Anything else that appears here without a phase landing it is doc rot.
