# Local Provider Unification Migration Guide

## Overview

The local repository support has been unified under the `LocalProvider` class, which implements the standard `ICodeProvider` interface. The previous `LocalRepoService` is deprecated.

## What Changed

### Before (Deprecated)
```python
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService

service = LocalRepoService(db_session)
content = service.get_file_content(
    repo_name="dummy",
    file_path="test.py",
    start_line=1,
    end_line=10,
    branch_name="main",
    project_id="project-123",
    commit_id=None
)
```

### After (Recommended)
```python
from app.modules.code_provider.provider_factory import CodeProviderFactory

provider = CodeProviderFactory.create_provider_with_fallback("/path/to/repo")
content = provider.get_file_content(
    repo_name="/path/to/repo",
    file_path="test.py",
    ref="main",
    start_line=1,
    end_line=10
)
```

## Key Differences

1. **No database dependency**: `LocalProvider` doesn't require `Session` or `project_id`
2. **Unified `ref` parameter**: Single parameter for branch or commit (was separate `branch_name` and `commit_id`)
3. **Optional line ranges**: `start_line` and `end_line` are now optional
4. **Return type changes**: `get_repository_structure()` returns `List[Dict]` instead of formatted string
5. **No side effects**: Uses `git show` instead of `git checkout`

## Backward Compatibility

`LocalRepoService` is still functional but deprecated. It will emit a `DeprecationWarning` when instantiated. Plan to migrate by version 2.0.

## Environment Variables

New configuration for explicit local provider use:

```bash
CODE_PROVIDER=local
CODE_PROVIDER_BASE_URL=/path/to/default/repo
```

Auto-detection still works - passing a valid path as `repo_name` will automatically use `LocalProvider`.

## Method Migration Guide

### get_file_content

**Old signature**:
```python
get_file_content(
    repo_name: str,
    file_path: str,
    start_line: int,
    end_line: int,
    branch_name: str,
    project_id: str,
    commit_id: Optional[str]
) -> str
```

**New signature**:
```python
get_file_content(
    repo_name: str,
    file_path: str,
    ref: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str
```

**Migration**:
- Remove `project_id` parameter
- Merge `branch_name` and `commit_id` into `ref` (prefer commit_id if available)
- `start_line` and `end_line` are now optional (None = full file)

### get_repository_structure

**Old signature** (async):
```python
async def get_project_structure_async(
    project_id: str,
    path: Optional[str] = None
) -> str  # Returns formatted string
```

**New signature** (sync):
```python
def get_repository_structure(
    repo_name: str,
    path: str = "",
    ref: Optional[str] = None,
    max_depth: int = 4
) -> List[Dict[str, Any]]  # Returns structured data
```

**Migration**:
- Remove async/await
- Replace `project_id` with actual `repo_name` (repository path)
- Handle structured return value instead of formatted string
- Add `max_depth` control if needed

### compare_branches

**New method** (not in LocalRepoService):
```python
def compare_branches(
    repo_name: str,
    base_branch: str,
    head_branch: str
) -> Dict[str, Any]
```

Returns:
```python
{
    "files": [
        {
            "filename": "path/to/file",
            "patch": "diff output",
            "status": "modified"
        }
    ],
    "commits": 12
}
```

## Common Migration Patterns

### Pattern 1: Basic File Reading

**Before**:
```python
# Requires database session and project lookup
service = LocalRepoService(db_session)
project = project_service.get_project(project_id)
content = service.get_file_content(
    repo_name=project.repo_name,
    file_path="src/main.py",
    start_line=0,
    end_line=0,
    branch_name="main",
    project_id=project_id,
    commit_id=None
)
```

**After**:
```python
# Direct access, no database required
provider = CodeProviderFactory.create_provider_with_fallback("/path/to/repo")
content = provider.get_file_content(
    repo_name="/path/to/repo",
    file_path="src/main.py",
    ref="main"
)
```

### Pattern 2: Repository Structure

**Before**:
```python
service = LocalRepoService(db_session)
structure_str = await service.get_project_structure_async(project_id)
# Parse the formatted string manually
```

**After**:
```python
provider = CodeProviderFactory.create_provider_with_fallback("/path/to/repo")
structure = provider.get_repository_structure("/path/to/repo")
# Use structured data directly
for item in structure:
    print(f"{item['type']}: {item['path']}")
    if item['type'] == 'directory':
        for child in item.get('children', []):
            print(f"  {child['name']}")
```

### Pattern 3: Branch Operations

**After** (new functionality):
```python
provider = CodeProviderFactory.create_provider_with_fallback("/path/to/repo")

# List branches
branches = provider.list_branches("/path/to/repo")

# Get branch info
branch_info = provider.get_branch("/path/to/repo", "main")

# Create new branch
new_branch = provider.create_branch("/path/to/repo", "feature-x", "main")

# Compare branches
comparison = provider.compare_branches("/path/to/repo", "main", "feature-x")
```

## Testing Your Migration

After migrating to `LocalProvider`, verify:

1. **File content retrieval works**:
   ```python
   content = provider.get_file_content(repo_path, "README.md")
   assert "expected content" in content
   ```

2. **Repository structure is correct**:
   ```python
   structure = provider.get_repository_structure(repo_path)
   assert isinstance(structure, list)
   assert all('name' in item and 'type' in item for item in structure)
   ```

3. **Branch operations function**:
   ```python
   branches = provider.list_branches(repo_path)
   assert len(branches) > 0
   ```

4. **No database calls** (for local provider):
   ```python
   # This should work without database session
   provider = LocalProvider(default_repo_path="/path/to/repo")
   content = provider.get_file_content("/path/to/repo", "file.txt")
   ```

## Breaking Changes Summary

### Version 2.0 (Future)
- `LocalRepoService` will be removed entirely
- All code must use `LocalProvider` via factory

### Current Version
- `LocalRepoService` is deprecated but functional
- `get_project_structure_async()` return type changed
- `get_file_content()` signature changed
- No database dependencies in `LocalProvider`

## Need Help?

If you encounter issues during migration:

1. Check that your repository path is valid and accessible
2. Verify you're not mixing `LocalRepoService` and `LocalProvider` calls
3. Ensure `ref` parameter uses branch names or commit SHAs (not both)
4. Review test files: `tests/modules/code_provider/test_local_provider.py`

## See Also

- **ICodeProvider interface**: `app/modules/code_provider/base/code_provider_interface.py`
- **LocalProvider implementation**: `app/modules/code_provider/local_repo/local_provider.py`
- **Provider factory**: `app/modules/code_provider/provider_factory.py`
- **Configuration**: `.env.template`
