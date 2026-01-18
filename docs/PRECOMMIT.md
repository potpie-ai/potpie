# Pre-commit Setup and Usage Guide

This guide covers setting up and using pre-commit in the Potpie repository. Pre-commit ensures code quality by running automated checks (linting, formatting, security scans) before commits. CI runs validation-only checks; developers must fix issues locally.

## Local Pre-commit Setup

### How to Install Pre-commit
Pre-commit is a Python tool. Install it globally using pip:

```bash
pip install pre-commit
```

For virtual environments, activate your env first or use `pip install --user pre-commit`.

### How to Install Pre-commit Hooks
After cloning the repository, install the hooks to run automatically on commits:

```bash
pre-commit install
```

This sets up Git hooks (e.g., pre-commit, commit-msg) to run checks before each commit. If hooks fail, the commit is blocked.

### How to Run Pre-commit Locally Before Pushing
Always run pre-commit locally before pushing to avoid CI failures. This catches issues early and allows you to fix them.

## Running Pre-commit Locally

### Run All Hooks
To check all files in the repository:

```bash
pre-commit run --all-files
```

This runs every hook in `.pre-commit-config.yaml` on all tracked/untracked files. It will report failures but not modify files (validation-only).

### Run Specific Hook
To run only a specific hook (e.g., for faster testing):

```bash
pre-commit run [hook-id] --all-files
```

Replace `[hook-id]` with the hook name, e.g.:
- `ruff` (linting)
- `ruff-format` (formatting check)
- `bandit` (security scan)
- `check-yaml` (YAML validation)

Example:
```bash
pre-commit run ruff --all-files
```

## Automatic Fixes

Some hooks support auto-fixing issues locally. In this repository's config, hooks are set to check-only (no auto-fixes in CI), but you can apply fixes manually.

### Which Hooks Auto-Fix and How to Apply Them
- **Ruff Formatting**: The `ruff-format` hook checks formatting but doesn't fix. To auto-fix formatting issues:
  ```bash
  ruff format .
  ```
  Then commit the changes.

- **Ruff Linting**: The `ruff` hook checks for lint issues but doesn't fix. To auto-fix some lint issues:
  ```bash
  ruff check --fix .
  ```
  Review and commit the fixes.

- **Other Hooks**: Most hooks (e.g., `bandit`, `check-yaml`) are checks only. No auto-fix available—fix manually.

Run fixes locally, then re-run `pre-commit run --all-files` to verify.

### Skipping Hooks If Needed
If a hook is temporarily problematic (e.g., false positive), skip it for a commit:

```bash
SKIP=hook-id git commit -m "Your message"
```

Example:
```bash
SKIP=ruff git commit -m "Skip ruff for now"
```

Use sparingly—hooks ensure quality. Remove skips once issues are resolved.

## Resolving Pre-commit Failures

### Common Issues and Fixes
- **Ruff Linting Errors**: Code style violations (e.g., unused imports). Fix by editing code to match Ruff rules or run `ruff check --fix .`.
- **Ruff Formatting Errors**: Inconsistent formatting. Run `ruff format .` to auto-fix.
- **Bandit Security Issues**: Potential security flaws. Review the output, fix code (e.g., avoid hardcoded secrets), or add `# nosec` comments if safe.
- **YAML/TOML Errors**: Invalid syntax. Check files with a YAML/TOML validator and correct.
- **Large Files/Debug Statements**: Remove large files or debug prints as flagged.

### How to Apply Auto-Fixes Locally
1. Run the failing hook specifically: `pre-commit run [hook-id] --all-files`.
2. Apply fixes (e.g., `ruff format .` or `ruff check --fix .`).
3. Re-run pre-commit to confirm: `pre-commit run --all-files`.
4. If fixed, commit.

### How to Manually Fix Validation Errors
For non-auto-fixable issues:
1. Review the error output (includes file/line details).
2. Edit the file manually (e.g., fix syntax, remove debug code).
3. Save and re-run pre-commit.

### When to Commit the Auto-Fix Changes
- After applying fixes locally and verifying with `pre-commit run --all-files`.
- Commit fixes in a separate commit or amend the previous one.
- Push only after all checks pass to avoid CI failures.

Example workflow:
```bash
# Make changes
git add .
pre-commit run --all-files  # Fix any issues
git add .  # Add fixes
git commit -m "feat: Add feature with fixes"
git push
```

If issues persist, check `.pre-commit-config.yaml` or ask for help. CI will enforce checks—fix locally to keep PRs clean!