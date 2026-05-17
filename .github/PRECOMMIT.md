# Pre-commit

Pre-commit runs local validation hooks before code is committed. The repository
also supports pre-commit.ci for read-only checks on pull requests and pushes to
`main`. CI must report failures only; developers apply fixes locally and commit
the resulting changes.

## Local Setup

Install pre-commit with pip:

```bash
python -m pip install pre-commit
```

Or install the project development dependencies if you already use uv:

```bash
uv sync
```

Install the git hooks:

```bash
pre-commit install
```

If pre-commit is installed only inside the uv-managed environment, run:

```bash
uv run pre-commit install
```

## Running Pre-commit Locally

Run every hook against every file:

```bash
pre-commit run --all-files
```

Run one hook against every file:

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
pre-commit run bandit --all-files
```

Skip a hook for one commit only when there is a documented reason:

```bash
SKIP=bandit git commit -m "Explain why this commit skips bandit"
```

## Automatic Fixes

The committed pre-commit hooks are configured as validation checks. They should
not modify files in CI.

Apply common automatic fixes locally before pushing:

```bash
ruff check --fix app tests
ruff format app tests
```

With uv:

```bash
uv run ruff check --fix app tests
uv run ruff format app tests
```

After running fixes, review the diff and commit the changed files:

```bash
git diff
git add <fixed-files>
git commit
```

## Resolving Failures

- `ruff` reports lint failures. Run `ruff check --fix app tests` for fixable
  issues, then manually fix any remaining diagnostics.
- `ruff-format` reports formatting differences. Run `ruff format app tests`.
- `bandit` reports security findings. Prefer a code change. Only suppress a
  finding when the risk is understood and documented in code.
- `check-yaml` or `check-toml` reports invalid syntax. Fix the file syntax and
  rerun the specific hook.
- `check-added-large-files` blocks files larger than the configured limit.
  Remove generated artifacts or store large assets outside git.

Run the failing hook again after each fix. Commit auto-fix changes in the same
branch before opening or updating the pull request.
