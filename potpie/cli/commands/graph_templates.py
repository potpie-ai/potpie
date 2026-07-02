"""Static graph mutation template payloads for CLI ergonomics."""

from __future__ import annotations

from typing import Any

# Static, schema-only mutation skeletons (Stage 5 ergonomics). These are
# helpers for harnesses writing mutation JSON by hand — placeholders only,
# never values read from the repo. Keys reference the canonical ontology so
# the contract tests can pin them against ENTITY_TYPES / EDGE_TYPES.
_MUTATION_TEMPLATES: dict[str, dict[str, Any]] = {
    "repo-baseline": {
        "pot_id": "<pot-id>",
        "idempotency_key": "baseline:<owner>/<repo>:v1",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "upsert_entity",
                "subject": {
                    "key": "repo:<host>/<owner>/<repo>",
                    "type": "Repository",
                    "name": "<repo>",
                    "summary": "<one-line repo purpose>",
                    "description": "<retrieval card: what the repo is, app type, synonyms a searcher would type>",
                },
            },
            {
                "op": "assert_claim",
                "subject": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "predicate": "PROVIDES",
                "object": {
                    "key": "feature:<feature-slug>",
                    "type": "Feature",
                    "name": "<feature name>",
                    "summary": "<one-line capability>",
                    "description": "<retrieval card: what it does, user-facing synonyms, scope>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<where the source says this — e.g. README features section>",
                "evidence": [
                    {
                        "source_ref": "repo:<owner>/<repo>#README",
                        "authority": "repository_metadata",
                    }
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEFINED_IN",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<which manifest/workflow defines the service>",
                "evidence": [
                    {
                        "source_ref": "repo:<owner>/<repo>#package.json",
                        "authority": "repository_metadata",
                    }
                ],
            },
        ],
    },
    "feature": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "assert_claim",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "PROVIDES",
                "object": {
                    "key": "feature:<feature-slug>",
                    "type": "Feature",
                    "name": "<feature name>",
                    "summary": "<one-line capability>",
                    "description": "<retrieval card with synonyms and scope>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "description": "<evidence summary>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            }
        ],
    },
    "preference": {
        "pot_id": "<pot-id>",
        "idempotency_key": "preference:<owner>/<repo>:<preference-slug>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "preference:<preference-slug>",
                    "type": "Preference",
                    "name": "<short preference name>",
                    "summary": "<one-line prescription>",
                    "description": "<retrieval card: when it applies, synonyms, scope>",
                    "properties": {
                        "policy_kind": "<error_handling|logging|testing|library_choice|file_structure>",
                        "prescription": "<specific guidance an agent should follow>",
                        "strength": "<hard|strong|normal|weak>",
                        "audience": "<repo|service|path|language>",
                    },
                },
                "predicate": "POLICY_APPLIES_TO",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "preference",
                "confidence": 0.9,
                "description": "<where the preference is stated>",
                "extra": {
                    "repo": "<host>/<owner>/<repo>",
                    "service": "<service-slug>",
                    "file_path": "<optional/path/or/directory>",
                    "language": "<language>",
                },
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "preference-policy": {
        "pot_id": "<pot-id>",
        "idempotency_key": "preference:<owner>/<repo>:<preference-slug>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "preference:<preference-slug>",
                    "type": "Preference",
                    "name": "<short preference name>",
                    "summary": "<one-line prescription>",
                    "description": "<retrieval card: task words, scope, synonyms>",
                    "properties": {
                        "policy_kind": "<error_handling|logging|testing|library_choice|file_structure>",
                        "prescription": "<specific guidance an agent should follow>",
                        "strength": "<hard|strong|normal|weak>",
                        "audience": "<repo|service|path|language>",
                    },
                },
                "predicate": "POLICY_APPLIES_TO",
                "object": {
                    "key": "code:<repo-or-service>:<path-or-symbol>",
                    "type": "CodeAsset",
                },
                "truth": "preference",
                "confidence": 0.9,
                "description": "<evidence summary and why this policy applies to this scope>",
                "extra": {
                    "repo": "<host>/<owner>/<repo>",
                    "service": "<service-slug>",
                    "file_path": "<path/or/directory>",
                    "language": "<language>",
                },
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "infra-snapshot": {
        "pot_id": "<pot-id>",
        "idempotency_key": "infra:<owner>/<repo>:<service-slug>:<environment>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEPLOYED_TO",
                "object": {"key": "environment:<environment>", "type": "Environment"},
                "truth": "source_observation",
                "confidence": 0.95,
                "environment": "<environment>",
                "description": "<source showing service runs in this environment>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "USES_ADAPTER",
                "object": {
                    "key": "adapter:<domain>:<adapter-slug>",
                    "type": "Adapter",
                    "summary": "<adapter/provider selected in this env>",
                    "description": "<retrieval card: adapter, provider, backend, env>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "environment": "<environment>",
                "description": "<source showing which adapter/backend is selected>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "DEPLOYED_WITH",
                "object": {
                    "key": "deployment_target:<environment>:<target-slug>",
                    "type": "DeploymentTarget",
                    "summary": "<deployment target/mechanism>",
                    "description": "<retrieval card: platform, workload, deploy mechanism>",
                },
                "truth": "source_observation",
                "confidence": 0.9,
                "environment": "<environment>",
                "description": "<source showing deployment target/mechanism>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
            {
                "op": "link_entities",
                "subject": {"key": "service:<service-slug>", "type": "Service"},
                "predicate": "CONFIGURES",
                "object": {
                    "key": "config:<service-or-env>:<config-name>",
                    "type": "ConfigVariable",
                    "summary": "<config variable>",
                    "description": "<retrieval card: env var/config key and behavior it selects>",
                },
                "truth": "source_observation",
                "confidence": 0.8,
                "environment": "<environment>",
                "description": "<source showing this config affects the service/adapter>",
                "evidence": [
                    {"source_ref": "<ref>", "authority": "repository_metadata"}
                ],
            },
        ],
    },
    "bug-fix": {
        "pot_id": "<pot-id>",
        "idempotency_key": "bug-fix:<bug-slug>:<fix-hash>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "bug_pattern:<bug-slug>",
                    "type": "BugPattern",
                    "name": "<symptom name>",
                    "summary": "<one-line symptom>",
                    "description": "<retrieval card: error text, symptoms, synonyms, where it shows up>",
                    "properties": {
                        "symptom_signature": "<stable symptom/error signature>",
                    },
                },
                "predicate": "REPRODUCES",
                "object": {"key": "service:<service-slug>", "type": "Service"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<how the bug manifests>",
            },
            {
                "op": "assert_claim",
                "subject": {
                    "key": "fix:<fix-hash>",
                    "type": "Fix",
                    "summary": "<one-line fix>",
                    "description": "<retrieval card: what fixed it, files touched, verification>",
                    "properties": {
                        "fix_steps": "<what changed or what to do>",
                        "verification_status": "<verified|unverified|failed>",
                    },
                },
                "predicate": "RESOLVED",
                "object": {"key": "bug_pattern:<bug-slug>", "type": "BugPattern"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<fix summary + verification status>",
            },
            {
                "op": "assert_claim",
                "subject": {
                    "key": "activity:<source>:<verification-id>",
                    "type": "Activity",
                },
                "predicate": "VERIFIED",
                "object": {"key": "fix:<fix-hash>", "type": "Fix"},
                "truth": "agent_claim",
                "confidence": 0.8,
                "description": "<how the fix was verified, or what remains unverified>",
            },
        ],
    },
    "decision": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "assert_claim",
                "subject": {
                    "key": "decision:<decision-hash>",
                    "type": "Decision",
                    "name": "<decision title>",
                    "summary": "<one-line decision>",
                    "description": "<retrieval card: rationale, alternatives rejected, synonyms>",
                },
                "predicate": "DECIDED",
                "object": {"key": "repo:<host>/<owner>/<repo>", "type": "Repository"},
                "truth": "user_decision",
                "confidence": 1.0,
                "description": "<who decided and why>",
                "evidence": [{"source_ref": "<ref>", "authority": "user_statement"}],
            }
        ],
    },
    "timeline-event": {
        "pot_id": "<pot-id>",
        "operations": [
            {
                "op": "append_event",
                "verb": "<verb e.g. merged_pr|deployed|decided>",
                "occurred_at": "<ISO-8601 timestamp>",
                "description": "<what happened, written for timeline recall>",
                "actor": {"key": "person:<handle>", "type": "Person"},
                "targets": [{"key": "service:<service-slug>", "type": "Service"}],
                "evidence": [{"source_ref": "<ref>", "authority": "external_system"}],
            }
        ],
    },
    "timeline-change": {
        "pot_id": "<pot-id>",
        "idempotency_key": "timeline:<source>:<id>",
        "created_by": {"surface": "cli", "harness": "<harness>"},
        "operations": [
            {
                "op": "append_event",
                "verb": "<verb e.g. merged_pr|linear_done|deployed|incident>",
                "occurred_at": "<ISO-8601 timestamp>",
                "description": "<what changed, source title, affected behavior, regression keywords>",
                "actor": {"key": "person:<handle>", "type": "Person"},
                "targets": [
                    {"key": "service:<service-slug>", "type": "Service"},
                    {
                        "key": "code:<repo-or-service>:<path-or-symbol>",
                        "type": "CodeAsset",
                    },
                ],
                "mentions": [{"key": "feature:<feature-slug>", "type": "Feature"}],
                "evidence": [{"source_ref": "<ref>", "authority": "external_system"}],
            }
        ],
    },
}

__all__ = ["_MUTATION_TEMPLATES"]
