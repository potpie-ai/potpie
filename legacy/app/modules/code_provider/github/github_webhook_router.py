"""GitHub webhook handler for automated PR reviews.

When a PR is opened/updated, GitHub calls POST /api/v1/github/webhook.
This handler looks up the Potpie project for that repo, runs the
BlastRadiusAgent (code_changes_agent) on the PR, and posts the findings
as a comment directly on the pull request.

Setup required:
  - GITHUB_WEBHOOK_SECRET env var: the secret you configure in GitHub
    (Settings -> Webhooks -> Secret). Leave unset to skip signature check
    (only for local testing).
  - GH_TOKEN_LIST env var: GitHub token used to post the review comment.
    Already used by other parts of the codebase.
"""

import hashlib
import hmac
import json
import os
from typing import Optional

from fastapi import BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.agents.chat_agent import ChatContext
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.APIRouter import APIRouter
from observability import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _post_pr_comment(repo_name: str, pr_number: int, body: str) -> None:
    from github import Github

    token = (os.getenv("GH_TOKEN_LIST", "").split(",")[0].strip()
             or os.getenv("GITHUB_APP_TOKEN", ""))
    if not token:
        logger.warning("No GitHub token available to post PR review comment")
        return

    gh = Github(token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(f"## Potpie PR Review\n\n{body}")


async def _run_pr_review(
    repo_name: str,
    pr_number: int,
    branch: str,
    pr_title: str,
) -> None:
    """Run in background: analyse the PR with BlastRadiusAgent and post findings."""
    db: Session = SessionLocal()
    try:
        project_service = ProjectService(db)
        project = await project_service.get_global_project_from_db(
            repo_name=repo_name,
            branch_name=branch,
        )
        if not project:
            logger.info(
                "No Potpie project found for repo — skipping PR review",
                repo_name=repo_name,
                branch=branch,
            )
            return

        user_id = project.user_id
        llm_provider = ProviderService(db, user_id)
        tools_provider = ToolService(db, user_id)
        prompt_provider = PromptService(db)
        agents_service = AgentsService(db, llm_provider, prompt_provider, tools_provider)

        ctx = ChatContext(
            project_id=str(project.id),
            project_name=project.repo_name,
            curr_agent_id="code_changes_agent",
            history=[],
            query=(
                f"Review pull request '{pr_title}' (#{pr_number}) on branch '{branch}'. "
                "Identify what could break, flag risky changes, and summarise your findings."
            ),
            user_id=user_id,
            repository=repo_name,
            branch=branch,
        )

        result = await agents_service.execute(ctx)
        review_body = result.response if result else "Potpie could not generate a review."
        _post_pr_comment(repo_name, pr_number, review_body)

    except Exception:
        logger.exception(
            "PR review background task failed",
            repo_name=repo_name,
            pr_number=pr_number,
        )
    finally:
        db.close()


@router.post("/github/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
):
    payload = await request.body()

    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if secret:
        if not x_hub_signature_256:
            raise HTTPException(status_code=400, detail="Missing X-Hub-Signature-256 header")
        if not _verify_signature(payload, x_hub_signature_256, secret):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    if x_github_event != "pull_request":
        return {"status": "ignored", "event": x_github_event}

    data = json.loads(payload)
    action = data.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "action": action}

    pr = data["pull_request"]
    repo_name = data["repository"]["full_name"]
    pr_number = pr["number"]
    branch = pr["head"]["ref"]
    pr_title = pr["title"]

    background_tasks.add_task(
        _run_pr_review,
        repo_name=repo_name,
        pr_number=pr_number,
        branch=branch,
        pr_title=pr_title,
    )

    return {"status": "accepted", "pr": pr_number, "repo": repo_name}
