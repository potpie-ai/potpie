import re

def validate_repo_name(repo_name: str) -> None:
    """
    Validate that repo_name follows GitHub's owner/repo naming convention.
    
    Format: owner/repo where:
    - owner: alphanumeric characters, hyphens, underscores (1-39 chars)
    - repo: alphanumeric characters, hyphens, underscores, dots (1-100 chars)
    
    Raises ValueError if format is invalid.
    """
    if not repo_name or not isinstance(repo_name, str):
        raise ValueError(
            "Repository name must be a non-empty string in the format 'owner/repo'. "
            "Example: 'octocat/Hello-World'"
        )
    
    # GitHub naming rules: owner/repo
    # owner: [a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])? up to 39 chars
    # repo: [a-zA-Z0-9_.-]+ up to 100 chars
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?/[a-zA-Z0-9._\-]+$'
    
    if not re.match(pattern, repo_name):
        raise ValueError(
            f"Invalid repository name format: '{repo_name}'. "
            "Expected format: 'owner/repo' where owner and repo contain only "
            "alphanumeric characters, hyphens, underscores, and (for repo) dots. "
            "Example: 'octocat/Hello-World'"
        )
    
    # Check length constraints
    owner, repo = repo_name.split("/", 1)
    if len(owner) > 39:
        raise ValueError(
            f"Owner name '{owner}' exceeds GitHub's 39-character limit."
        )
    if len(repo) > 100:
        raise ValueError(
            f"Repository name '{repo}' exceeds GitHub's 100-character limit."
        )
    if repo_name.endswith(".git"):
        raise ValueError(
            f"Repository name should not include '.git' suffix: '{repo_name}'. "
            f"Use '{repo_name[:-4]}' instead."
        )
