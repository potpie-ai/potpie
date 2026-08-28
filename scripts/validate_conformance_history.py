"""Validate stable conformance records and their Git-backed lineage."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_DIR = ROOT / "spec" / "conformance"
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
ACTIVE_BEHAVIOR_RE = re.compile(r"^([A-Z][A-Z0-9-]*-\d{3}) \[active\]:", re.MULTILINE)
TRACE_BEHAVIOR_RE = re.compile(r"^\| ([A-Z][A-Z0-9-]*-\d{3}) \|", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class RecordScope:
    record_id: str
    spec_id: str
    spec_path: str


SCOPES = {
    "cli.md": RecordScope("CONF-CLI", "SPEC-CLI", "spec/modules/cli.md"),
    "context-engine.md": RecordScope(
        "CONF-CONTEXT-ENGINE",
        "SPEC-CONTEXT-ENGINE",
        "spec/modules/context-engine.md",
    ),
    "cross-system.md": RecordScope("CONF-SYSTEM", "SPEC-SYSTEM", "spec/system.md"),
    "daemon.md": RecordScope("CONF-DAEMON", "SPEC-DAEMON", "spec/modules/daemon.md"),
    "potpie-capabilities.md": RecordScope(
        "CONF-POTPIE-CAPABILITIES",
        "SPEC-POTPIE-CAPABILITIES",
        "spec/modules/potpie-capabilities.md",
    ),
    "potpie-resource-manager.md": RecordScope(
        "CONF-POTPIE-RESOURCE-MANAGER",
        "SPEC-POTPIE-RESOURCE-MANAGER",
        "spec/modules/potpie-resource-manager.md",
    ),
}
EXPECTED_FILES = {*SCOPES, "index.md"}
REQUIRED_FIELDS = {
    "id",
    "title",
    "kind",
    "record_status",
    "spec_id",
    "spec_revision",
    "spec_ref",
    "implementation_ref",
    "performed_by",
    "performed_at",
    "result",
    "previous_record",
    "previous_record_id",
    "previous_record_ref",
    "previous_record_path",
}
TARGET_FIELDS = {
    "target_repository",
    "target_pull_request",
    "target_base_ref",
    "target_base_commit",
    "target_pr_head_ref",
    "target_pr_head_commit",
    "target_merge_candidate",
    "target_merge_tree",
}


class ValidationError(Exception):
    """A deterministic conformance validation failure."""


def parse_frontmatter(text: str, source: str) -> dict[str, str | None]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValidationError(f"{source}: missing opening frontmatter delimiter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValidationError(
            f"{source}: missing closing frontmatter delimiter"
        ) from exc

    metadata: dict[str, str | None] = {}
    for line_number, line in enumerate(lines[1:closing], start=2):
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) or ":" not in line:
            raise ValidationError(
                f"{source}:{line_number}: conformance frontmatter must remain flat"
            )
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if not key or not value:
            raise ValidationError(f"{source}:{line_number}: invalid frontmatter field")
        if value == "null":
            metadata[key] = None
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            metadata[key] = value[1:-1]
        else:
            metadata[key] = value
    return metadata


def parse_contract_frontmatter(text: str, source: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValidationError(f"{source}: missing opening frontmatter delimiter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValidationError(
            f"{source}: missing closing frontmatter delimiter"
        ) from exc

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        if line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if value:
            metadata[key] = value.strip("'\"")
    return metadata


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValidationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def git_object_exists(object_name: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", object_name],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def require_full_sha(value: str | None, field: str, source: str) -> str:
    if not isinstance(value, str) or FULL_SHA_RE.fullmatch(value) is None:
        raise ValidationError(f"{source}: {field} must be a full 40-character Git SHA")
    return value


def validate_behavior_scope(
    body: str,
    contract: str,
    source: str,
) -> None:
    heading = "## Behavior Trace"
    if heading not in body:
        raise ValidationError(f"{source}: missing {heading} section")
    trace = body.split(heading, 1)[1].split("\n## ", 1)[0]
    traced = set(TRACE_BEHAVIOR_RE.findall(trace))
    active = set(ACTIVE_BEHAVIOR_RE.findall(contract))
    if traced != active:
        missing = sorted(active - traced)
        extra = sorted(traced - active)
        raise ValidationError(
            f"{source}: behavior trace mismatch; missing={missing}, extra={extra}"
        )


def validate_record(filename: str, scope: RecordScope) -> int:
    path = CONFORMANCE_DIR / filename
    relative_path = path.relative_to(ROOT).as_posix()
    body = path.read_text(encoding="utf-8")
    metadata = parse_frontmatter(body, relative_path)

    missing = sorted(REQUIRED_FIELDS - metadata.keys())
    if missing:
        raise ValidationError(f"{relative_path}: missing fields {missing}")
    if metadata["id"] != scope.record_id:
        raise ValidationError(f"{relative_path}: expected stable id {scope.record_id}")
    if metadata["spec_id"] != scope.spec_id:
        raise ValidationError(f"{relative_path}: expected spec_id {scope.spec_id}")
    if metadata["kind"] != "conformance-record" or metadata["record_status"] != "final":
        raise ValidationError(
            f"{relative_path}: record must be a final conformance-record"
        )
    if metadata["result"] not in {
        "passed",
        "failed",
        "partial",
        "indeterminate",
        "unverified",
    }:
        raise ValidationError(
            f"{relative_path}: invalid aggregate result {metadata['result']}"
        )
    if metadata["previous_record"] is not None:
        raise ValidationError(
            f"{relative_path}: previous_record must be null; use the three historical fields"
        )

    spec_ref = require_full_sha(metadata["spec_ref"], "spec_ref", relative_path)
    implementation_ref = require_full_sha(
        metadata["implementation_ref"], "implementation_ref", relative_path
    )
    previous_ref = require_full_sha(
        metadata["previous_record_ref"], "previous_record_ref", relative_path
    )
    git("cat-file", "-e", f"{implementation_ref}^{{commit}}")

    contract = git("show", f"{spec_ref}:{scope.spec_path}")
    contract_metadata = parse_contract_frontmatter(
        contract, f"{spec_ref}:{scope.spec_path}"
    )
    if contract_metadata.get("id") != scope.spec_id:
        raise ValidationError(
            f"{relative_path}: spec_ref does not resolve {scope.spec_id}"
        )
    if contract_metadata.get("revision") != metadata["spec_revision"]:
        raise ValidationError(f"{relative_path}: spec revision does not match spec_ref")
    if contract_metadata.get("maturity") != "accepted":
        raise ValidationError(f"{relative_path}: spec_ref is not an accepted contract")

    previous_path = metadata["previous_record_path"]
    if not isinstance(previous_path, str) or not previous_path.startswith(
        "spec/conformance/"
    ):
        raise ValidationError(f"{relative_path}: invalid previous_record_path")
    previous = git("show", f"{previous_ref}:{previous_path}")
    previous_metadata = parse_frontmatter(previous, f"{previous_ref}:{previous_path}")
    if previous_metadata.get("id") != metadata["previous_record_id"]:
        raise ValidationError(
            f"{relative_path}: previous record ID does not match historical file"
        )

    validate_behavior_scope(body, contract, relative_path)
    return len(set(ACTIVE_BEHAVIOR_RE.findall(contract)))


def validate_cross_system_target() -> None:
    source = "spec/conformance/cross-system.md"
    metadata = parse_frontmatter((ROOT / source).read_text(encoding="utf-8"), source)
    missing = sorted(TARGET_FIELDS - metadata.keys())
    if missing:
        raise ValidationError(f"{source}: missing integration target fields {missing}")

    base_commit = require_full_sha(
        metadata["target_base_commit"], "target_base_commit", source
    )
    pr_head = require_full_sha(
        metadata["target_pr_head_commit"], "target_pr_head_commit", source
    )
    merge_candidate = require_full_sha(
        metadata["target_merge_candidate"], "target_merge_candidate", source
    )
    merge_tree = require_full_sha(
        metadata["target_merge_tree"], "target_merge_tree", source
    )
    git("cat-file", "-e", f"{base_commit}^{{commit}}")
    git("cat-file", "-e", f"{pr_head}^{{commit}}")
    if git_object_exists(f"{merge_candidate}^{{commit}}"):
        candidate_parents = (
            git("show", "-s", "--format=%P", merge_candidate).strip().split()
        )
        if candidate_parents != [base_commit, pr_head]:
            raise ValidationError(
                f"{source}: merge-candidate parents do not match base and PR head"
            )
        candidate_tree = git("show", "-s", "--format=%T", merge_candidate).strip()
        if candidate_tree != merge_tree:
            raise ValidationError(
                f"{source}: target_merge_tree does not match merge-candidate tree"
            )


def validate_local_links(path: Path) -> None:
    relative_path = path.relative_to(ROOT).as_posix()
    for target in MARKDOWN_LINK_RE.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        local_target = target.split("#", 1)[0]
        if not local_target:
            continue
        resolved = (path.parent / local_target).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.exists():
            raise ValidationError(
                f"{relative_path}: link target does not resolve: {target}"
            )


def main() -> int:
    try:
        actual_files = {path.name for path in CONFORMANCE_DIR.glob("*.md")}
        if actual_files != EXPECTED_FILES:
            raise ValidationError(
                "spec/conformance: expected exactly "
                f"{sorted(EXPECTED_FILES)}, found {sorted(actual_files)}"
            )

        total_behaviors = sum(
            validate_record(filename, scope) for filename, scope in SCOPES.items()
        )
        validate_cross_system_target()
        for filename in EXPECTED_FILES:
            validate_local_links(CONFORMANCE_DIR / filename)
        validate_local_links(ROOT / "spec" / "index.md")

        index_text = (CONFORMANCE_DIR / "index.md").read_text(encoding="utf-8")
        for filename in SCOPES:
            if f"({filename})" not in index_text:
                raise ValidationError(
                    f"spec/conformance/index.md: missing current-record link to {filename}"
                )
        if total_behaviors != 195:
            raise ValidationError(
                f"spec/conformance: expected 195 covered active behaviors, found {total_behaviors}"
            )
    except (OSError, ValidationError) as exc:
        print(f"conformance validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        "conformance validation passed: 6 stable records, 195 behaviors, history resolved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
