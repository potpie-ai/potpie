# Pre-commit Hook Configuration

Potpie uses [pre-commit](https://pre-commit.com/) to ensure code quality and consistency. 
To maintain a strict "Review What You Merge" policy, our configuration is **read-only** in CI environments.

## Quick Start

1.  **Install pre-commit**:
    ```bash
    pip install pre-commit
    ```

2.  **Install the Git Hook**:
    ```bash
    pre-commit install
    ```
    This ensures checks run automatically on every `git commit`.

## Production Workflow (CI/CD)

In our CI/CD pipeline (including `pre-commit.ci`):
-   **Auto-fixing is DISABLED**.
-   If a check fails (e.g., linting error, formatting issue), the build will **fail**.
-   The CI will **NOT** modify your code. You must fix it locally and push the changes.

**Why?**
Auto-fixing in CI can be dangerous. It can introduce subtle bugs or revert intentional changes (like organizing imports in a specific way) without the developer noticing. We believe every line of code change should be explicitly committed by a human.

## How to Fix Issues Locally

If your commit fails or CI fails, here is how to resolve it:

### 1. Run all checks
To see what's wrong across the entire repo:
```bash
pre-commit run --all-files
```

### 2. Apply Fixes Automatically
Since we disabled auto-fix in the hook, you must run the fix commands manually if you want the tools to do the work for you:

**Fix Linting (Ruff):**
```bash
uv run ruff check --fix .
```

**Fix Formatting (Ruff):**
```bash
uv run ruff format .
```

### 3. Commit the Changes
After running the fix commands, inspect the changes to ensure they are correct, then `git add` and `git commit` again.

## Skipping Hooks (Emergency Only)
If you absolutely must bypass a hook (e.g., for a WIP commit), you can use:
```bash
git commit -n
# OR
SKIP=ruff git commit -m "..."
```
**Warning**: This will likely cause the CI to fail. Use with caution.
