"""Check key outcomes in the frozen experiment bundle; no backend required."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def records(name):
    with gzip.open(ROOT / name, "rt") as stream:
        return {row["id"]: row for row in map(json.loads, stream)}


def rank(row, target):
    return next(
        (i + 1 for i, item in enumerate(row["payload"].get("items", []))
         if target in json.dumps(item)),
        None,
    )


final = records("queries-final.jsonl.gz")
seed = records("seed-and-first-candidate.jsonl.gz")
followups = records("followups-first-candidate.jsonl.gz")
holdouts = records("holdouts.jsonl.gz")
assert len(final) == 45
for case, target in [
    ("body-default", "field-reference/recovery-marker"),
    ("rollback-search", "ledgerly-runbook/rollback-procedure"),
    ("rollback-default", "ledgerly-runbook/rollback-procedure"),
    ("pms-docs", "field-reference/terminology"),
]:
    assert rank(final[case], target) == 1, case
verification = seed["commit-02-infra"]["payload"]["result"]["verification"]
assert verification["readback_count"] == 37
assert verification["missing_claim_keys"] == []
assert followups["commit-minimal"]["payload"]["result"]["verification"]["ok"]
neighborhood = json.loads((ROOT / "minimal-read-final.json").read_text())["result"]
assert neighborhood["relation_count"] == 2
assert {r["environment"] for r in neighborhood["relations"]} == {"prod", "staging"}
for case in ["invalid-include", "infra-invalid-direction"]:
    assert final[case]["exit_code"] == 1, case
assert all(
    not item["breakdown"].get("lexical_coverage_floor")
    for item in holdouts["near-miss-token"]["payload"]["items"]
)
assert final["negative-default"]["payload"]["metadata"]["readers"]["resources"]["warnings"]
print("Frozen evidence checks passed; this is not a general retrieval-success score.")
