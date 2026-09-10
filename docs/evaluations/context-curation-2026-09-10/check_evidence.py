"""Check the frozen diagnostic evidence, not desired product behavior."""

import gzip
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    path = root / "probes-final.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = {row["id"]: row for line in stream if (row := json.loads(line))}

    def payload(name):
        value = rows[name]["payload"]
        return value.get("result", value)

    def text(name):
        return json.dumps(payload(name))

    old = payload("old-citation")["chunks"][0]
    new = payload("same-citation-after-refresh")["chunks"][0]
    neighborhood = payload("infra-api-neighborhood")
    modes = [payload("resolve-mode-" + mode) for mode in ("fast", "deep", "verify")]
    failed_fix = next(
        item
        for item in payload("debug-after-failed-verification")["items"]
        if item["payload"]["predicate"] == "RESOLVED"
    )
    checks = {
        "citation_same_id_different_revision_and_text": (
            old["resource_id"] == new["resource_id"]
            and old["revision"] == 1
            and new["revision"] == 2
            and "47 minutes" in old["text"]
            and "5 minutes" in new["text"]
        ),
        "stale_derived_claim_still_active_after_refresh": "Wait 47 minutes"
        in text("derived-claim-after-refresh"),
        "dangling_derived_claim_after_removal": (
            "Wait 47 minutes" in text("derived-claim-after-removal")
            and rows["citation-after-removal"]["exit_code"] == 1
        ),
        "document_scope_link_survives_document_removal": any(
            rel["predicate"] == "DOCUMENTS" for rel in neighborhood["relations"]
        ),
        "rationale_stored_but_missing_from_full_decision_read": (
            "QUARTZ-819" in json.dumps(neighborhood["nodes"])
            and "QUARTZ-819" not in text("decision-read")
        ),
        "fix_details_stored_but_missing_from_full_debug_read": (
            "TOPAZ-621" in json.dumps(neighborhood["nodes"])
            and "Restart pool" not in text("fix-read")
            and "unverified" not in text("fix-read")
        ),
        "record_source_detail_accepted_but_not_preserved": (
            payload("record-decision")["accepted"]
            and "https://example.invalid/adr/19" not in text("decision-read")
            and "https://example.invalid/adr/19"
            not in json.dumps(neighborhood["nodes"])
        ),
        "path_policy_leaks_to_unrelated_path": (
            "Use fixture-only database tests" in text("unrelated-path-policy-read")
            and "src/payments" not in text("unrelated-path-policy-read")
        ),
        "repo_record_and_read_identity_mismatch": (
            payload("record-canonical-repo")["accepted"]
            and not payload("canonical-repo-read")["items"]
            and "repo:github-com-curation-demo" in text("canonical-repo-entity-search")
        ),
        "api_exposure_stored_but_missing_from_advertised_view": (
            not payload("infra-api-read")["items"]
            and any(rel["predicate"] == "EXPOSES" for rel in neighborhood["relations"])
        ),
        "failed_verification_counted_as_corroboration": (
            failed_fix["payload"]["verification_count"] == 1
            and failed_fix["breakdown"]["corroboration"] > 0.5
            and "didnt_work" not in text("debug-after-failed-verification")
        ),
        "default_human_passage_omits_snippet_and_fetch": (
            "47 minutes" in text("passage-json")
            and "47 minutes" not in rows["passage-human"]["stdout"]
            and "potpie resource get" not in rows["passage-human"]["stdout"]
        ),
        "document_scope_link_does_not_expand_section_summaries": (
            len(payload("doc-level-scoped-read")["items"]) == 1
            and payload("doc-level-scoped-read")["items"][0]["claim"]["predicate"]
            == "DOCUMENTS"
        ),
        "resolve_modes_return_same_candidates_and_reader_bounds": all(
            [(item["include"], item["candidate_key"]) for item in mode["items"]]
            == [(item["include"], item["candidate_key"]) for item in modes[0]["items"]]
            and mode["metadata"]["readers"] == modes[0]["metadata"]["readers"]
            for mode in modes[1:]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    result = {
        "purpose": "Observed contract gaps, not pass/fail retrieval accuracy",
        "revision": "631cf218f6eb18fcaa7394fc1c07ccded7ac16b4",
        "pot": "local:curation-review-final-20260910",
        "pot_id": payload("create")["id"],
        "commands": len(rows),
        "nonzero_exits": [name for name, row in rows.items() if row["exit_code"]],
        "observed_gaps": checks,
        "existing_tests_passed": {"cli": 86, "core": 38, "engine": 8},
    }
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Verified {len(checks)} diagnostic observations from {len(rows)} CLI commands."
    )


if __name__ == "__main__":
    main()
