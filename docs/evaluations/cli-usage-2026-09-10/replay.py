"""Replay CLI usability probes; optionally seed a newly created, named pot.

This captures evidence, not a pass/fail test: several probes intentionally
exercise errors. See queries.json and README.md for expected behavior.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="potpie", help="Executable CLI or wrapper")
    parser.add_argument(
        "--pot", required=True, help="Explicit local:name or managed:name"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSONL output file"
    )
    parser.add_argument(
        "--seed", action="store_true", help="Create a NEW pot and seed it"
    )
    parser.add_argument("--case", action="append", help="Run only these query IDs")
    args = parser.parse_args()
    cli = shutil.which(args.cli)
    if not cli:
        parser.error(f"CLI executable not found: {args.cli}")
    cli = str(Path(cli).resolve())
    origin, separator, name = args.pot.partition(":")
    if not separator or origin not in {"local", "managed"} or not name.strip():
        parser.error("--pot must be local:name or managed:name")
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "queries.json").read_text())
    if args.case:
        missing = set(args.case) - {case["id"] for case in cases}
        if missing:
            parser.error(f"Unknown query IDs: {sorted(missing)}")
        cases = [case for case in cases if case["id"] in args.case]

    # Refuse to overwrite a prior transcript, even when --seed is omitted.
    with args.output.open("x", encoding="utf-8") as log:

        def run(label: str, command: list[str], *, scoped: bool = True) -> dict:
            argv = [cli, "--json", *command]
            if scoped:
                argv.extend(["--pot", args.pot])
            started = time.monotonic()
            # Caller explicitly selects the executable; arguments never use a shell.
            proc = subprocess.run(  # noqa: S603
                argv, text=True, capture_output=True, timeout=120
            )
            elapsed = round(time.monotonic() - started, 3)
            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError:
                payload = None
            row = {
                "id": label,
                "command": argv,
                "exit_code": proc.returncode,
                "seconds": elapsed,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "payload": payload,
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{label}: exit={proc.returncode}, {elapsed:.3f}s")
            return row

        def result(row: dict) -> dict:
            payload = row["payload"] or {}
            return payload.get("result", payload) or {}

        def seed_plan(path: Path) -> None:
            proposal = run(
                "propose-" + path.stem,
                [
                    "graph",
                    "propose",
                    "--file",
                    str(path),
                    "--approved-by",
                    "harness:synthetic-eval",
                ],
            )
            if proposal["exit_code"] != 0 or not result(proposal).get("plan_id"):
                raise RuntimeError(f"Seed proposal failed; inspect {args.output}")
            commit = run(
                "commit-" + path.stem,
                [
                    "graph",
                    "commit",
                    result(proposal)["plan_id"],
                    "--verify",
                ],
            )
            if result(commit).get("status") != "committed":
                raise RuntimeError(f"Seed commit failed; inspect {args.output}")
            # A committed write with failed readback is a finding, not a reason
            # to hide the later read results or retry the mutation automatically.
            if commit["exit_code"]:
                print("  COMMITTED BUT VERIFICATION FAILED: see transcript")

        if args.seed:
            created = run(
                "create", ["--host", origin, "pot", "create", name], scoped=False
            )
            if created["exit_code"] or result(created).get("created") is not True:
                raise RuntimeError("Refusing to seed: pot was not newly created")
            plans = sorted((root / "fixtures/plans").glob("0[1-6]-*.json"))
            for path in plans:
                seed_plan(path)
            for directory in sorted((root / "fixtures/resources").iterdir()):
                if not directory.is_dir():
                    continue
                imported = run(
                    "import-" + directory.name,
                    [
                        "resource",
                        "import",
                        str(directory),
                        "--doc",
                        directory.name,
                    ],
                )
                if imported["exit_code"] or not result(imported).get("graph", {}).get(
                    "written"
                ):
                    raise RuntimeError(f"Resource import failed; inspect {args.output}")
            seed_plan(root / "fixtures/plans/08-doc-links.json")
        for case in cases:
            run(case["id"], case["args"])


if __name__ == "__main__":
    main()
