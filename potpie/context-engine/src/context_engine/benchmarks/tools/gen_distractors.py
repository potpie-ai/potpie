"""Emit N noise fixture envelopes matching a shape spec.

The bench plan calls for distractor ratios up to 25:1 (bench-plan §5.3).
Hand-authoring 25 noise events per signal event is busywork; this tool
generates them from a single template, rotating ids and timestamps.

Usage::

    python -m context_engine.benchmarks.tools.gen_distractors \\
        --template fixtures/raw_events/github/pr_merged__998__inventory_unrelated.json \\
        --count 25 \\
        --id-prefix noise_pr \\
        --connector github \\
        --out fixtures/raw_events/noise/

The output filenames are ``<id-prefix>__001.json`` .. ``<id-prefix>__N.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path


def _rotate_envelope(template: dict, index: int, id_prefix: str) -> dict:
    cloned = copy.deepcopy(template)
    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"{id_prefix}#{index}").hex[:10]
    cloned["source_id"] = f"{template['source_id']}::noise-{suffix}"
    data = (
        cloned.get("payload", {}).get("data")
        if isinstance(cloned.get("payload"), dict)
        else None
    )
    if isinstance(data, dict):
        if "id" in data:
            data["id"] = f"{data['id']}-noise-{suffix}"
        # Mutate any obvious title to mark this as noise.
        for k in ("title", "subject", "name"):
            if k in data and isinstance(data[k], str):
                data[k] = f"[noise:{suffix}] {data[k]}"
    # PR-shape envelopes carry an extra ``pull_request`` block.
    pr = (
        cloned.get("payload", {}).get("pull_request")
        if isinstance(cloned.get("payload"), dict)
        else None
    )
    if isinstance(pr, dict):
        if "number" in pr:
            pr["number"] = int(pr["number"]) + 10000 + index
        if "title" in pr and isinstance(pr["title"], str):
            pr["title"] = f"[noise:{suffix}] {pr['title']}"
    return cloned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        "gen_distractors",
        description="Generate N rotated copies of a fixture envelope as distractors.",
    )
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--id-prefix", required=True)
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args(argv)

    with args.template.open("r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, dict):
        print(
            f"error: template {args.template} did not contain a JSON object",
            file=sys.stderr,
        )
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(1, args.count + 1):
        envelope = _rotate_envelope(template, i, args.id_prefix)
        out_path = args.out / f"{args.id_prefix}__{i:03d}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(envelope, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        written += 1
    print(f"wrote {written} distractor envelopes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
