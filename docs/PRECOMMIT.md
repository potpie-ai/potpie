# Pre-commit Guide

## Setup

Install pre-commit:

```
pip install pre-commit
```

Install hooks:

```
pre-commit install
```

---

## Running Checks

Run all hooks:

```
pre-commit run --all-files
```

Run specific hook:

```
pre-commit run <hook-id> --all-files
```

---

## Auto-fixes

Some hooks can automatically fix issues locally.

Run:

```
pre-commit run --all-files
```

Then commit the changes.

---

## Skipping Hooks

Skip a hook:

```
SKIP=<hook-id> git commit
```

---

## Fixing Failures

If checks fail:

1. Run:

```
pre-commit run --all-files
```

2. Fix issues manually or apply auto-fixes

3. Commit changes

---

## CI Behavior

* CI runs only validation checks
* No automatic fixes are applied in CI
* Developers must fix issues locally before pushing
