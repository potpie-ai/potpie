import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GitBucketWebhookEvent(str, Enum):
    """GitBucket webhook event types."""

    CREATE = "CreateEvent"
    ISSUES = "IssuesEvent"
    ISSUE_COMMENT = "IssueCommentEvent"
    PULL_REQUEST_REVIEW_COMMENT = "PullRequestReviewCommentEvent"
    PULL_REQUEST = "PullRequestEvent"
    PUSH = "PushEvent"
    GOLLUM = "GollumEvent"


class GitBucketWebhookParser:
    """
    Parse GitBucket webhook payloads.

    GitBucket webhooks are similar to GitHub's but may have slight differences.
    """

    @staticmethod
    def parse_webhook(
        event_type: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Parse GitBucket webhook payload into normalized format.

        Args:
            event_type: GitBucket event type (e.g., 'PushEvent')
            payload: Raw webhook payload

        Returns:
            Normalized event data or None if unsupported
        """
        try:
            if event_type == GitBucketWebhookEvent.PUSH:
                return GitBucketWebhookParser._parse_push_event(payload)
            elif event_type == GitBucketWebhookEvent.PULL_REQUEST:
                return GitBucketWebhookParser._parse_pull_request_event(payload)
            elif event_type == GitBucketWebhookEvent.ISSUES:
                return GitBucketWebhookParser._parse_issues_event(payload)
            elif event_type == GitBucketWebhookEvent.ISSUE_COMMENT:
                return GitBucketWebhookParser._parse_issue_comment_event(payload)
            else:
                logger.info(f"Unsupported GitBucket event type: {event_type}")
                return None
        except Exception as e:
            logger.error(f"Error parsing GitBucket webhook: {e}", exc_info=True)
            return None

    @staticmethod
    def _parse_push_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GitBucket push event."""
        return {
            "event_type": "push",
            "provider": "gitbucket",
            "repository": payload.get("repository", {}).get("full_name"),
            "ref": payload.get("ref"),
            "commits": payload.get("commits", []),
            "pusher": payload.get("pusher", {}).get("name"),
        }

    @staticmethod
    def _parse_pull_request_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GitBucket pull request event."""
        pr = payload.get("pull_request", {})
        return {
            "event_type": "pull_request",
            "provider": "gitbucket",
            "action": payload.get("action"),
            "repository": payload.get("repository", {}).get("full_name"),
            "pull_request": {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "head_branch": pr.get("head", {}).get("ref"),
                "base_branch": pr.get("base", {}).get("ref"),
            },
        }

    @staticmethod
    def _parse_issues_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GitBucket issues event."""
        issue = payload.get("issue", {})
        return {
            "event_type": "issues",
            "provider": "gitbucket",
            "action": payload.get("action"),
            "repository": payload.get("repository", {}).get("full_name"),
            "issue": {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
            },
        }

    @staticmethod
    def _parse_issue_comment_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GitBucket issue comment event."""
        return {
            "event_type": "issue_comment",
            "provider": "gitbucket",
            "action": payload.get("action"),
            "repository": payload.get("repository", {}).get("full_name"),
            "issue": payload.get("issue", {}).get("number"),
            "comment": payload.get("comment", {}).get("body"),
        }
