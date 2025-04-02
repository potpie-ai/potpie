"""
Test script for Linear tools integration with user-specific API keys.
Tests both get_linear_issue_tool and update_linear_issue_tool.
"""

import asyncio
import argparse
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.intelligence.tools.linear_tools.get_linear_issue_tool import (
    get_linear_issue_tool,
)
from app.modules.intelligence.tools.linear_tools.update_linear_issue_tool import (
    update_linear_issue_tool,
)
from app.modules.key_management.secret_manager import SecretStorageHandler


async def test_get_linear_issue(
    db: Session, user_id: str, issue_id: str
) -> Dict[str, Any]:
    """Test the get_linear_issue_tool with a specific issue ID."""
    print(f"\n🔍 Testing get_linear_issue_tool for issue: {issue_id}")

    # Get the tool instance
    tool = get_linear_issue_tool(db, user_id)
    print(f"Tool created: {tool.name} ({tool.__class__.__name__})")

    try:
        # Call the _arun method directly to test
        result = await tool._arun(issue_id=issue_id)
        print(f"✅ Successfully retrieved issue {issue_id}")
        print(f"Title: {result.get('title')}")
        print(f"Status: {result.get('status')}")
        print(f"Assignee: {result.get('assignee')}")
        return result
    except Exception as e:
        print(f"❌ Error retrieving issue: {str(e)}")
        return None


async def test_update_linear_issue(
    db: Session, user_id: str, issue_id: str
) -> Dict[str, Any]:
    """Test the update_linear_issue_tool with a specific issue ID."""
    print(f"\n✏️ Testing update_linear_issue_tool for issue: {issue_id}")

    # Get the tool instance
    tool = update_linear_issue_tool(db, user_id)
    print(f"Tool created: {tool.name} ({tool.__class__.__name__})")

    try:
        # Call the _arun method directly with a test comment
        result = await tool._arun(
            issue_id=issue_id,
            comment="Test comment from the updated tool implementation",
        )
        print(f"✅ Successfully updated issue {issue_id}")
        print(f"Title: {result.get('title')}")
        print(f"Status: {result.get('status')}")
        print(f"Comment added: {result.get('comment_added')}")
        return result
    except Exception as e:
        print(f"❌ Error updating issue: {str(e)}")
        return None


async def check_user_has_linear_key(user_id: str, db: Session) -> bool:
    """Check if the user has a Linear API key configured."""
    exists = await SecretStorageHandler.check_secret_exists(
        service="linear", customer_id=user_id, service_type="integration", db=db
    )

    if not exists:
        print(f"⚠️ User {user_id} does not have a Linear API key configured.")
        print("Please configure it using the SecretManager integration API.")
        return False

    return True


async def run_tests(user_id: str, issue_id: str):
    """Run all the tests sequentially."""
    print(f"🚀 Starting Linear Tools Test for user: {user_id}")

    # Get database session
    db = next(get_db())

    # Check if user has Linear API key
    if not await check_user_has_linear_key(user_id, db):
        return

    # Test get_linear_issue_tool
    issue = await test_get_linear_issue(db, user_id, issue_id)

    if issue:
        # Test update_linear_issue_tool if get issue was successful
        await test_update_linear_issue(db, user_id, issue_id)

    print("\n✨ All tests completed!")


if __name__ == "__main__":
    """Run the test script when executed directly."""
    parser = argparse.ArgumentParser(description="Test the Linear tools integration")
    parser.add_argument("--user-id", required=True, help="User ID to test with")
    parser.add_argument(
        "--issue-id", required=True, help="Linear issue ID to test with (e.g., LIN-123)"
    )

    args = parser.parse_args()
    asyncio.run(run_tests(args.user_id, args.issue_id))
