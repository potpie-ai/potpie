#!/usr/bin/env python3
"""Run the QA test plan from Tests.xlsx and write results to a dated sheet.

What it does
------------
1. (Optional) Switches the target repo to a given branch (default: ``main``)
   so all tests execute against that branch.
2. Reads the ``Daily Plan`` sheet from the workbook.
3. For every test row it extracts the shell command(s) embedded in backticks in
   the "Step / Command" column and decides whether the test is auto-runnable.
   Only a conservative allow-list of ``potpie`` CLI commands is executed; rows
   that describe manual/process work (PyPI submission, approvals, dashboards,
   cross-Python-version installs, ...) are recorded as MANUAL and skipped.
4. Runnable commands are executed one by one. stdout/stderr/exit-code are
   captured as evidence and PASS/FAIL is derived from the exit code.
5. Results are printed to the console and written back to the workbook in a new
   sheet whose name is the run date (e.g. ``2026-06-27``).

Usage
-----
    python scripts/run_test_plan.py                 # full run against main
    python scripts/run_test_plan.py --dry-run       # classify only, run nothing
    python scripts/run_test_plan.py --no-checkout   # don't touch git
    python scripts/run_test_plan.py --excel /path/to/Tests.xlsx
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import textwrap

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:  # pragma: no cover
    sys.exit("openpyxl is required: pip install openpyxl")


DEFAULT_EXCEL = "/Users/jagadeesh/Downloads/Tests.xlsx"
DEFAULT_REPO = "/Users/jagadeesh/repo/potpie"
SOURCE_SHEET = "Daily Plan"
HEADER_ROW = 4  # row that holds column titles in the source sheet

# Commands we are willing to execute automatically. Anything that does not start
# with one of these prefixes is treated as a manual/process step. This keeps the
# runner away from destructive or environment-mutating actions (PyPI publish,
# `uv tool install`, creating new-Python venvs, etc.).
SAFE_PREFIXES = (
    "potpie --version",
    "potpie status",
    "potpie doctor",
    "potpie whoami",
    "potpie setup",
    "potpie source add",
    "potpie source list",
    "potpie resolve",
    "potpie search",
    "potpie config get",
    "potpie graph",
)

BACKTICK_RE = re.compile(r"`([^`]+)`")
MAX_EVIDENCE_CHARS = 6000

# Some safe commands require a positional argument the test plan omits (it lists
# them as bare `potpie resolve` / `potpie search`). Supply a real default so the
# command actually exercises the feature instead of erroring on a missing arg.
# Keys are matched against the bare command (no trailing args); values are
# overridable from the CLI (--resolve-task / --search-query).
DEFAULT_ARGS = {
    "potpie resolve": 'How does the CLI host routing work?',
    "potpie search": 'host_cli',
}


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
class TestResult:
    def __init__(self, row):
        self.seq = row.get("Seq")
        self.tid = row.get("ID")
        self.type = row.get("Type")
        self.workstream = row.get("Workstream")
        self.priority = row.get("Priority")
        self.step = row.get("Step / Command") or ""
        self.expected = row.get("Expected Result / Exit Criteria") or ""
        self.commands_run = []
        self.result = "PENDING"
        self.exit_code = ""
        self.evidence = ""
        self.timestamp = ""


# --------------------------------------------------------------------------- #
# Parsing / classification
# --------------------------------------------------------------------------- #
def extract_commands(step_text: str) -> list[str]:
    """Return backtick-quoted command-looking snippets from a step description."""
    out = []
    for raw in BACKTICK_RE.findall(step_text or ""):
        cmd = raw.strip()
        if cmd:
            out.append(cmd)
    return out


def is_runnable(cmd: str) -> bool:
    return any(cmd.startswith(p) for p in SAFE_PREFIXES)


def augment_command(cmd: str, default_args: dict[str, str]) -> str:
    """Append a default positional argument to bare arg-requiring commands.

    e.g. ``potpie resolve`` -> ``potpie resolve "How does ... work?"`` so the
    test exercises the command instead of failing on a missing argument. A
    command that already carries its own argument is left untouched.
    """
    bare = cmd.strip()
    if bare in default_args:
        return f'{bare} "{default_args[bare]}"'
    return cmd


def classify(commands: list[str]) -> tuple[list[str], str]:
    """Return (runnable_commands, manual_reason).

    manual_reason is empty when the test is auto-runnable.
    """
    runnable = [c for c in commands if is_runnable(c)]
    if runnable:
        return runnable, ""
    if commands:
        return [], "Manual/process step (commands present but not auto-runnable: %s)" % "; ".join(commands)
    return [], "Manual/process step (no executable CLI command)"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_command(cmd: str, cwd: str, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return 124, f"TIMEOUT after {timeout}s\n{partial}".strip()
    except Exception as exc:  # pragma: no cover
        return 1, f"RUNNER ERROR: {exc}"


def git_checkout(repo: str, branch: str) -> tuple[bool, str]:
    code, out = run_command(f"git rev-parse --abbrev-ref HEAD", repo, 30)
    current = out.strip()
    if current == branch:
        return True, f"Already on '{branch}'."
    code, out = run_command(f"git checkout {branch}", repo, 60)
    if code != 0:
        return False, out
    code2, head = run_command("git rev-parse --abbrev-ref HEAD", repo, 30)
    return True, f"Switched from '{current}' to '{head.strip()}'.\n{out}".strip()


# --------------------------------------------------------------------------- #
# Workbook IO
# --------------------------------------------------------------------------- #
def load_tests(wb) -> list[dict]:
    ws = wb[SOURCE_SHEET]
    headers = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=HEADER_ROW, column=c).value
        if v:
            headers[c] = str(v).strip()
    rows = []
    for r in range(HEADER_ROW + 1, ws.max_row + 1):
        if ws.cell(row=r, column=4).value is None:  # column D = ID
            continue
        row = {headers[c]: ws.cell(row=r, column=c).value for c in headers}
        rows.append(row)
    return rows


def unique_sheet_name(wb, base: str) -> str:
    name = base[:31]
    if name not in wb.sheetnames:
        return name
    i = 2
    while True:
        candidate = f"{base[:28]}_{i}"
        if candidate not in wb.sheetnames:
            return candidate
        i += 1


RESULT_FILLS = {
    "PASS": PatternFill("solid", fgColor="C6EFCE"),
    "FAIL": PatternFill("solid", fgColor="FFC7CE"),
    "MANUAL": PatternFill("solid", fgColor="FFEB9C"),
    "SKIPPED": PatternFill("solid", fgColor="D9D9D9"),
}


def write_results(wb, sheet_name: str, results: list[TestResult], meta: dict):
    ws = wb.create_sheet(title=sheet_name)
    cols = [
        ("Seq", 6),
        ("ID", 26),
        ("Type", 12),
        ("Workstream", 20),
        ("Priority", 8),
        ("Step / Command", 50),
        ("Expected Result", 45),
        ("Command(s) Run", 38),
        ("Result", 10),
        ("Exit Code", 9),
        ("Evidence", 70),
        ("Run Timestamp", 20),
    ]

    # Metadata banner
    ws.cell(row=1, column=1, value=f"Test run: {meta['run_at']}").font = Font(bold=True, size=12)
    ws.cell(row=2, column=1, value=f"Branch: {meta['branch']}  |  Repo: {meta['repo']}")
    ws.cell(row=3, column=1, value=f"Checkout: {meta['checkout']}")
    summary = "  ".join(f"{k}={v}" for k, v in meta["summary"].items())
    ws.cell(row=4, column=1, value=f"Summary: {summary}").font = Font(bold=True)

    header_row = 6
    header_fill = PatternFill("solid", fgColor="305496")
    for ci, (title, width) in enumerate(cols, start=1):
        cell = ws.cell(row=header_row, column=ci, value=title)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width = width

    for ri, res in enumerate(results, start=header_row + 1):
        values = [
            res.seq,
            res.tid,
            res.type,
            res.workstream,
            res.priority,
            res.step,
            res.expected,
            "\n".join(res.commands_run),
            res.result,
            res.exit_code,
            res.evidence,
            res.timestamp,
        ]
        for ci, val in enumerate(values, start=1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        rcell = ws.cell(row=ri, column=9)
        rcell.font = Font(bold=True)
        if res.result in RESULT_FILLS:
            rcell.fill = RESULT_FILLS[res.result]

    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    return ws


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", default=DEFAULT_EXCEL, help="Path to the workbook.")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Repo to run CLI tests in.")
    ap.add_argument("--branch", default="main", help="Branch to switch to before testing.")
    ap.add_argument("--no-checkout", action="store_true", help="Do not change git branch.")
    ap.add_argument("--dry-run", action="store_true", help="Classify only; execute nothing.")
    ap.add_argument("--timeout", type=int, default=240, help="Per-command timeout (seconds).")
    ap.add_argument("--sheet", default=None, help="Override the result sheet name.")
    ap.add_argument("--resolve-task", default=DEFAULT_ARGS["potpie resolve"],
                    help="Default TASK argument injected into bare `potpie resolve`.")
    ap.add_argument("--search-query", default=DEFAULT_ARGS["potpie search"],
                    help="Default QUERY argument injected into bare `potpie search`.")
    ap.add_argument("--keep-existing", action="store_true",
                    help="Keep an existing same-date sheet and write a suffixed one instead of replacing it.")
    args = ap.parse_args(argv)

    default_args = {
        "potpie resolve": args.resolve_task,
        "potpie search": args.search_query,
    }

    if not os.path.exists(args.excel):
        sys.exit(f"Workbook not found: {args.excel}")

    run_at = dt.datetime.now()
    run_stamp = run_at.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Branch checkout
    checkout_msg = "skipped (--no-checkout)" if args.no_checkout else "skipped (dry-run)"
    if not args.no_checkout and not args.dry_run:
        ok, checkout_msg = git_checkout(args.repo, args.branch)
        print(f"[git] {checkout_msg}")
        if not ok:
            sys.exit(f"Failed to checkout '{args.branch}': {checkout_msg}")
    else:
        print(f"[git] checkout {checkout_msg}")

    # 2. Load workbook + tests
    wb = openpyxl.load_workbook(args.excel)
    wb_data = openpyxl.load_workbook(args.excel, data_only=True)  # for cached values
    tests = load_tests(wb_data)
    print(f"[plan] {len(tests)} tests loaded from '{SOURCE_SHEET}'\n")

    results: list[TestResult] = []
    counts = {"PASS": 0, "FAIL": 0, "MANUAL": 0, "SKIPPED": 0}

    for i, row in enumerate(tests, start=1):
        res = TestResult(row)
        commands = extract_commands(res.step)
        runnable, manual_reason = classify(commands)
        label = f"[{i}/{len(tests)}] {res.tid}"

        if not runnable:
            res.result = "MANUAL"
            res.evidence = manual_reason
            res.timestamp = run_stamp
            counts["MANUAL"] += 1
            print(f"{label}  MANUAL  - {manual_reason[:80]}")
            results.append(res)
            continue

        runnable = [augment_command(c, default_args) for c in runnable]
        res.commands_run = runnable
        if args.dry_run:
            res.result = "SKIPPED"
            res.evidence = "dry-run: would execute -> " + " ; ".join(runnable)
            res.timestamp = run_stamp
            counts["SKIPPED"] += 1
            print(f"{label}  DRY     - would run: {' ; '.join(runnable)}")
            results.append(res)
            continue

        # Execute each runnable command, collect evidence
        all_ok = True
        evidence_chunks = []
        last_code = 0
        for cmd in runnable:
            print(f"{label}  RUN     $ {cmd}")
            code, out = run_command(cmd, args.repo, args.timeout)
            last_code = code
            status = "ok" if code == 0 else f"exit {code}"
            evidence_chunks.append(f"$ {cmd}  ->  {status}\n{out}")
            if code != 0:
                all_ok = False
        evidence = "\n\n".join(evidence_chunks).strip()
        if len(evidence) > MAX_EVIDENCE_CHARS:
            evidence = evidence[:MAX_EVIDENCE_CHARS] + "\n... [truncated]"

        res.result = "PASS" if all_ok else "FAIL"
        res.exit_code = last_code
        res.evidence = evidence
        res.timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counts[res.result] += 1
        print(f"{label}  {res.result}")
        results.append(res)

    # 3. Write results sheet. Re-running on the same date refreshes that day's
    # sheet by default; pass --keep-existing to append a suffixed sheet instead.
    if args.sheet:
        sheet_name = args.sheet
    else:
        base = run_at.strftime("%Y-%m-%d")
        if args.keep_existing:
            sheet_name = unique_sheet_name(wb, base)
        else:
            sheet_name = base[:31]
            if sheet_name in wb.sheetnames:
                del wb[sheet_name]
    meta = {
        "run_at": run_stamp,
        "branch": "(not changed)" if (args.no_checkout or args.dry_run) else args.branch,
        "repo": args.repo,
        "checkout": checkout_msg,
        "summary": counts,
    }
    write_results(wb, sheet_name, results, meta)
    if args.dry_run:
        print("\n[dry-run] workbook NOT modified.")
    else:
        wb.save(args.excel)

    print("\n" + "=" * 60)
    if not args.dry_run:
        print(f"Results written to sheet '{sheet_name}' in {args.excel}")
    print("Summary: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("=" * 60)


if __name__ == "__main__":
    main()
