"""Diagnostic curation probes. Creates only a NEW explicitly named local pot."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--pot", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.pot.startswith("local:"):
        parser.error("Use a new local:name pot")
    cli = str(args.cli.resolve())
    with args.output.open("x") as log, tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)

        def run(label, command, *, scoped=True, human=False):
            argv = [cli, *([] if human else ["--json"]), *command]
            if scoped:
                argv += ["--pot", args.pot]
            # The caller selects the executable; all arguments bypass the shell.
            result = subprocess.run(  # noqa: S603
                argv, capture_output=True, text=True, timeout=120
            )
            try:
                payload = json.loads(result.stdout)
            except ValueError:
                payload = None
            log.write(
                json.dumps(
                    dict(
                        id=label,
                        command=argv,
                        exit_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        payload=payload,
                    )
                )
                + "\n"
            )
            log.flush()
            print(label, result.returncode, flush=True)
            return (payload or {}).get("result", payload or {}) or {}

        created = run(
            "create", ["--host", "local", "pot", "create", args.pot[6:]], scoped=False
        )
        if created.get("created") is not True:
            raise RuntimeError("Refusing to modify an existing pot")

        def plan(label, operations):
            path = root / (label + ".json")
            path.write_text(
                json.dumps(
                    {
                        "idempotency_key": label,
                        "created_by": {"surface": "cli"},
                        "operations": operations,
                    }
                )
            )
            proposal = run(
                label + "-propose", ["graph", "propose", "--file", str(path)]
            )
            if not proposal.get("plan_id"):
                raise RuntimeError(proposal)
            return run(
                label + "-commit", ["graph", "commit", proposal["plan_id"], "--verify"]
            )

        def ref(key, kind):
            return {"key": key, "type": kind}

        def claim(subject, predicate, obj, description, source, subgraph="knowledge"):
            return dict(
                op="assert_claim",
                subgraph=subgraph,
                subject=subject,
                predicate=predicate,
                object=obj,
                description=description,
                truth="agent_claim",
                confidence=0.8,
                evidence=[{"source_ref": source}],
            )

        def import_revision(number, body, summary):
            folder = root / str(number)
            (folder / "recovery").mkdir(parents=True)
            (folder / "recovery/0000.txt").write_text(body)
            (folder / "meta.json").write_text(
                json.dumps(
                    {
                        "source_ref": "https://example.invalid/curation/recovery",
                        "source_kind": "markdown",
                        "sections": [
                            {
                                "slug": "recovery",
                                "title": "Recovery procedure",
                                "summary": summary,
                                "content_hash": hashlib.sha256(
                                    body.encode()
                                ).hexdigest(),
                                "chunks": [
                                    {
                                        "seq": 0,
                                        "label": "Recovery procedure",
                                        "offset": 12,
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            run(
                "import-v" + str(number),
                ["resource", "import", str(folder), "--doc", "curation-runbook"],
            )

        chunk = "potpie://res/curation-runbook/recovery/0000"
        import_revision(
            1,
            "Recovery for curation-api: wait 47 minutes before retrying. Exact token AMBER-581.",
            "Curation API recovery wait and retry procedure.",
        )
        plan(
            "document-link",
            [
                claim(
                    ref("document:curation-runbook", "Document"),
                    "DOCUMENTS",
                    ref("service:curation-api", "Service"),
                    "Recovery runbook for curation-api",
                    chunk,
                )
            ],
        )
        read_docs = [
            "graph",
            "read",
            "--subgraph",
            "knowledge",
            "--view",
            "document_context",
            "--scope",
            "service:curation-api",
            "--detail",
            "full",
        ]
        run("doc-level-scoped-read", read_docs)
        run(
            "passage-human",
            ["search", "AMBER-581", "--include", "resources"],
            human=True,
        )
        run("passage-json", ["search", "AMBER-581", "--include", "resources"])
        plan(
            "derived-claim",
            [
                claim(
                    ref("preference:curation-wait", "Preference"),
                    "POLICY_APPLIES_TO",
                    ref("service:curation-api", "Service"),
                    "Wait 47 minutes before retrying curation-api recovery",
                    chunk,
                    "decisions",
                )
            ],
        )
        run("old-citation", ["resource", "get", chunk])
        import_revision(
            2,
            "Recovery for curation-api: wait 5 minutes before retrying. Exact token AMBER-582.",
            "Curation API recovery now requires a five minute wait.",
        )
        run("same-citation-after-refresh", ["resource", "get", chunk])
        preference_read = [
            "graph",
            "read",
            "--subgraph",
            "decisions",
            "--view",
            "preferences_for_scope",
            "--scope",
            "service:curation-api",
            "--detail",
            "full",
            "--relations",
            "full",
        ]
        run("derived-claim-after-refresh", preference_read)
        run("remove-test-document", ["resource", "rm", "curation-runbook", "--confirm"])
        run("derived-claim-after-removal", preference_read)
        run("citation-after-removal", ["resource", "get", chunk])

        run(
            "record-decision",
            [
                "record",
                "--type",
                "decision",
                "--summary",
                "Use bounded retries",
                "--detail",
                "rationale=QUARTZ-819 upstream rate limit demands backoff",
                "--detail",
                "source_ref=https://example.invalid/adr/19",
                "--scope",
                "service:curation-api",
            ],
        )
        run(
            "decision-read",
            [
                "graph",
                "read",
                "--subgraph",
                "decisions",
                "--view",
                "active_decisions",
                "--scope",
                "service:curation-api",
                "--detail",
                "full",
                "--relations",
                "full",
            ],
        )
        run("rationale-search", ["search", "QUARTZ-819", "--include", "decisions"])
        run(
            "record-fix",
            [
                "record",
                "--type",
                "fix",
                "--summary",
                "Timeout during recovery",
                "--detail",
                "root_cause=TOPAZ-621 connection pool exhaustion",
                "--detail",
                "fix_steps=Restart pool",
                "--scope",
                "service:curation-api",
            ],
        )
        run(
            "fix-read",
            [
                "graph",
                "read",
                "--subgraph",
                "debugging",
                "--view",
                "prior_occurrences",
                "--scope",
                "service:curation-api",
                "--detail",
                "full",
                "--relations",
                "full",
            ],
        )
        run(
            "record-path-policy",
            [
                "record",
                "--type",
                "preference",
                "--summary",
                "Use fixture-only database tests",
                "--detail",
                "policy_kind=testing",
                "--scope",
                "service:curation-api,path:src/payments",
            ],
        )
        run(
            "unrelated-path-policy-read",
            [
                "graph",
                "read",
                "--subgraph",
                "decisions",
                "--view",
                "preferences_for_scope",
                "--scope",
                "service:curation-api,path:src/analytics",
                "--detail",
                "full",
                "--relations",
                "full",
            ],
        )
        run(
            "record-canonical-repo",
            [
                "record",
                "--type",
                "decision",
                "--summary",
                "Use repo-owned adapters",
                "--detail",
                "rationale=Keep adapters near callers",
                "--scope",
                "repo:github.com/curation/demo",
            ],
        )
        run(
            "canonical-repo-read",
            [
                "graph",
                "read",
                "--subgraph",
                "decisions",
                "--view",
                "active_decisions",
                "--scope",
                "repo:github.com/curation/demo",
                "--detail",
                "full",
            ],
        )
        run("canonical-repo-entity-search", ["graph", "search-entities", "adapters"])

        api = ref("api_contract:curation-api:post:recover", "APIContract")
        api["properties"] = {"path": "/recover", "http_method": "POST"}
        plan(
            "api-exposure",
            [
                claim(
                    ref("service:curation-api", "Service"),
                    "EXPOSES",
                    api,
                    "Curation API exposes POST /recover",
                    "https://example.invalid/openapi",
                    "infra_topology",
                )
            ],
        )
        run(
            "infra-api-read",
            [
                "graph",
                "read",
                "--subgraph",
                "infra_topology",
                "--view",
                "service_neighborhood",
                "--scope",
                "service:curation-api",
                "--detail",
                "full",
            ],
        )
        run(
            "infra-api-neighborhood",
            [
                "graph",
                "neighborhood",
                "--entity",
                "service:curation-api",
                "--detail",
                "full",
            ],
        )
        run(
            "failed-verification",
            [
                "record",
                "--type",
                "verification",
                "--summary",
                "Restart pool did not work",
                "--detail",
                "target_ref=fix:timeout-during-recovery",
                "--detail",
                "outcome=didnt_work",
            ],
        )
        run(
            "debug-after-failed-verification",
            ["search", "Timeout during recovery", "--include", "prior_bugs"],
        )
        for mode in ("fast", "deep", "verify"):
            run(
                "resolve-mode-" + mode,
                ["resolve", "Timeout during recovery", "--mode", mode],
            )


if __name__ == "__main__":
    main()
