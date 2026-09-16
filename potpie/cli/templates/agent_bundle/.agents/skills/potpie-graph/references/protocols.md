# Protocol definitions (optional extension v1)

Activate only when `potpie --json graph catalog --profile full --pot <pot>`
advertises `protocols.message_context`. Local startup uses
`CONTEXT_ENGINE_PROTOCOLS_ENABLED=true`; Pie uses
`PIE_CONTEXT_GRAPH_PROTOCOLS_ENABLED=true`. Both default off. Restart the host
after changing the setting. Disabling preserves data; do not run base-schema
cleanup against protocol entities. Explicitly injected definitions take precedence.

## Find and inspect

Use entity search or `potpie --json resolve '<message/field question>' --include protocols --pot <pot>`
to discover compact message cards. Their `retrieval` object carries the selected
pot, view and scope. Then:

```bash
potpie --json graph describe protocols --view message_context --examples --pot <pot>
potpie --json graph read --subgraph protocols --view message_context --scope 'anchor_entity_key:<key>' --detail full --pot <pot>
```

The named read requires a Service, Protocol, ProtocolMessage or ProtocolField
`anchor_entity_key`. Optional scope filters are exact `revision`, `profile`,
`field_path`; query filters message discovery before structural expansion. Do not
pass repo, environment, depth, source, time-window or similarity filters. `as_of`
filters claims only; properties remain the current projection. Default resolve
recipes, automatic chat hints and nudges are unchanged.

Full output contains ordered `fields`, exact typed `allowed_values`, field claims
with source revision/digest and locator, message relationships, coverage and a
follow-up descriptor. Integer 2, string "2", false, zero and missing properties
are distinct. Ask for the relevant revision/message when a value is ambiguous.
`field_path` is exact and case-sensitive. Read a field anchor or refine to its
path for large layouts. Never infer source completeness from a result count.
Budgets: 2048 discovered claim rows, 12 returned messages, 128 total returned fields,
192 KiB response target. Truncation is explicit; fetch the
immutable resource chunk IDs in evidence for omitted tables/details.

CAN_SEND/CAN_RECEIVE are capabilities, RESPONDS_TO is a relationship between
definitions, and PROTOCOL_IMPLEMENTED_BY points to a codec/handler CodeAsset.
They do not prove any actual exchange. For decoder diagnosis retrieve prior bugs,
repo context and decisions separately at supported Service/CodeAsset anchors;
combine with source-backed traces. No packet decoder or compatibility verdict is
provided. Existing explorer navigation works with the added labels.

## Ingest and reconcile

Use resources and existing semantic plans; no new record kinds. Keep the selected
pot explicit. Import source bytes under an immutable slug containing the source
digest. Refetch at least one cited chunk. Retain prior source versions: refreshing
or removing a resource used by live protocol claims, source coverage, or current corrections is refused. Import revised
bytes under a new slug; explicitly retract obsolete dependent claims and patch
source coverage before removal. Old resource deletion loses historical fetches,
so normally keep the bytes even after retraction.

Only three entity types and six predicates are added:

- Protocol → DEFINES_MESSAGE → ProtocolMessage
- ProtocolMessage → HAS_FIELD → ProtocolField
- Service → CAN_SEND / CAN_RECEIVE → ProtocolMessage
- ProtocolMessage → RESPONDS_TO → ProtocolMessage (many responses permitted)
- ProtocolMessage → PROTOCOL_IMPLEMENTED_BY → CodeAsset

Reuse Document/DocumentSection DOCUMENTS and Decision AFFECTS; do not broaden
EXPOSES, IMPLEMENTED_IN or bug predicates. HAS_FIELD connects every materialized
nested path directly to its message. Never invent offsets or conditional rules.

Use `potpie_context_core.protocols.protocol_entity` to make semantic entity refs;
it mints SHA-256 keys from versioned typed tuples, preserving identifier case:

```python
from potpie_context_core.protocols import protocol_entity
p = protocol_entity("Protocol", properties={
    "namespace": "synthetic", "identifier": "Demo", "revision": "1", "profile": "Test"})
m = protocol_entity("ProtocolMessage", parent_name=p["name"], properties={
    "protocol_key": p["key"], "namespace": "PDU", "kind": "request",
    "direction": "client-to-server", "discriminator": "Q"})
f = protocol_entity("ProtocolField", parent_name=m["name"], properties={
    "message_key": m["key"], "path": "header.Status", "ordinal": 0,
    "type": "uint8", "byte_offset": 0, "offset_origin": "PDU",
    "allowed_values": [{"raw_value": 2, "symbol": "BUSY", "meaning": "Work pending"}]})
# Build source-backed operations only after importing the synthetic resource.
op = {"op": "assert_claim", "subgraph": "protocols", "subject": m,
      "predicate": "HAS_FIELD", "object": f, "truth": "agent_claim",
      "description": "Synthetic Demo revision 1 Query header.Status enum: integer 2 means BUSY, work pending",
      "evidence": [{"source_ref": "<imported immutable chunk id>",
                    "metadata": {"digest": "<actual source SHA-256>", "locator": "contract table row 1"}}]}
```

Create the DEFINES_MESSAGE operation as well. Use qualified generated names;
short repeated names cause ambiguous quality findings. A missing revision is
`revision: null` plus an immutable `unresolved_source`; do not guess a version or
substitute environment. Revision/profile changes create new entities. Unknown
revision reconciliation creates a resolved identity; it is not an identity patch.
Describe advertises all public properties and correction allowlists are explicit.
`source_coverage` on a message is `{status, source_ref, digest, locator}`;
`expected_field_count` is the count of materialized paths, not captured values.
Mark partial/unknown when tables, fields or sources are missing. A verified
resource evidence entry needs a `chunk_digest` (or `digest`) equal to SHA-256
of the fetched UTF-8 chunk text; `source_revision` alone stays unverified.
`source_ref` is the canonical chunk ID; a supplied `chunk_id` must equal it.
Metadata cannot override `source_ref` or `authority`. Current corrections retain
evidence per property; message completeness also verifies definition and correction
evidence. A failed bulk source fetch leaves coverage unverified.
Missing stores, unavailable chunks and digest mismatches cannot report complete.

Write mutation JSON through the shell at `/tmp/pie-mutation-*.json`, then
`graph propose --file <path>` and `graph commit <plan_id> --verify`.
Every protocol operation needs `subgraph: protocols`; use searchable descriptions
that mention the identifier, revision, role and field/value meanings. An unchanged
extraction is idempotent. Conflicting same-revision properties are rejected and
preserved in plan history: create an inbox candidate with both evidence pointers.
For a verified extraction correction use evidenced `patch_entity` for allowed
non-identity properties and reassert the HAS_FIELD evidence from the corrected
source. Check review requirements from the concrete proposed plan.

Check commit success, `verification.ok`, `verification.content_readback`, and
reader coverage separately. Bulk `--verify` includes per-chunk commit verification;
the top-level backend readiness report is not content verification. After bulk,
read each affected message with full detail and fetch evidence; record partial
resume status honestly. Standalone LLM ingestion and free-text Learning records
cannot supply this typed ontology. Field constraints document a contract; they do
not automatically validate captured payloads.
