def get_repo_details(payload):
    """Extract repo details from the payload."""
    return f"""
    Repo Details:
    Repo Name: {payload.get('repository', {}).get('full_name', "Unknown")}
    Is Private: {payload.get('repository', {}).get('private', "Unknown")}
    """


def get_pr_details(payload):
    """Extract PR details from the payload."""
    return f"""
    {get_repo_details(payload)}
    
    PR Details:
    PR Number: {payload.get('number', "Unknown")}
    PR Title: {payload.get('pull_request', {}).get('title', "Unknown")}
    PR URL: {payload.get('pull_request', {}).get('html_url', "Unknown")}
    Created By User: {payload.get('pull_request', {}).get('user', {}).get('login', "Unknown")}
    Head Branch: {payload.get('pull_request', {}).get('head', {}).get('ref', "Unknown")}
    Base Branch: {payload.get('pull_request', {}).get('base', {}).get('ref', "Unknown")}
    Diff URL: {payload.get('pull_request', {}).get('diff_url', "Unknown")}
    Patch URL: {payload.get('pull_request', {}).get('patch_url', "Unknown")}
    """


def get_issue_details(payload):
    """Extract issue details from the payload."""
    return f"""
    {get_repo_details(payload)}
    Issue Details:
    Issue Number: {payload.get('issue', {}).get('number', "Unknown")}
    Issue Title: {payload.get('issue', {}).get('title', "Unknown")}
    Issue URL: {payload.get('issue', {}).get('html_url', "Unknown")}
    Created By User: {payload.get('issue', {}).get('user', {}).get('login', "Unknown")}
    Issue State: {payload.get('issue', {}).get('state', "Unknown")}
    Issue Labels: {payload.get('issue', {}).get('labels', "Unknown")}
    
    Issue Body: 
    {payload.get('issue', {}).get('body', "Unknown")}"""


def get_comment_details(payload):
    """Extract comment details from the payload."""
    return f"""
    Comment Details:
    Comment Created By User: {payload.get('comment', {}).get('user', {}).get('login', "Unknown")}
    Comment Body: {payload.get('comment', {}).get('body', "Unknown")}
    Comment URL: {payload.get('comment', {}).get('html_url', "Unknown")}
    """


def get_current_repo(payload: dict) -> str:
    """Extract the current repository from GitHub event payload."""
    # Extract repository full name from the repository object
    if payload.get("repository", {}).get("full_name"):
        return payload.get("repository", {}).get("full_name", "unknown")

    return "unknown"


def get_current_branch(payload: dict) -> str:
    """Extract the current branch from GitHub event payload."""
    # For pull request events, use the head branch (the branch being merged)
    if payload.get("pull_request"):
        return payload.get("pull_request", {}).get("head", {}).get("ref", "unknown")

    # For push events, use the ref (branch name)
    if payload.get("ref"):
        # GitHub sends refs as 'refs/heads/main', we want just 'main'
        ref = payload.get("ref", "")
        if ref.startswith("refs/heads/"):
            return ref[11:]  # Remove 'refs/heads/' prefix
        return ref

    # For other events, try to extract from various possible locations
    if payload.get("repository", {}).get("default_branch"):
        return payload.get("repository", {}).get("default_branch", "unknown")

    return "unknown"


def get_github_event_type(headers: dict) -> str:
    """Extract GitHub event type from headers."""
    if not headers:
        return "unknown"

    # GitHub sends the event type in the X-GitHub-Event header
    event_type = headers.get("X-GitHub-Event", headers.get("x-github-event", "unknown"))
    return event_type.lower()


def get_github_action(payload: dict) -> str:
    """Extract GitHub action from payload."""
    if not payload:
        return "unknown"

    # The action field contains the specific action (opened, closed, reopened, etc.)
    action = payload.get("action", "unknown")
    return action.lower()


def matches_github_trigger(
    event_type: str, action: str, trigger_type: str, payload: dict
) -> bool:
    """Check if a GitHub event matches a specific trigger type."""
    # Map trigger types to expected event types and actions
    trigger_mapping = {
        "trigger_github_pr_opened": {"event_type": "pull_request", "action": "opened"},
        "trigger_github_pr_closed": {"event_type": "pull_request", "action": "closed"},
        "trigger_github_pr_reopened": {
            "event_type": "pull_request",
            "action": "reopened",
        },
        "trigger_github_pr_merged": {
            "event_type": "pull_request",
            "action": "closed",  # Merged PRs have action="closed" and merged=true
        },
        "trigger_github_issue_opened": {"event_type": "issues", "action": "opened"},
    }

    expected = trigger_mapping.get(trigger_type)
    if not expected:
        return False

    # Check event type
    if event_type != expected["event_type"]:
        return False

    # Check action
    if action != expected["action"]:
        return False

    # Special case for merged PRs: need to check if merged=true
    if trigger_type == "trigger_github_pr_merged":
        return payload.get("pull_request", {}).get("merged", False) is True

    return True
