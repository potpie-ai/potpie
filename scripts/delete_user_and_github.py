#!/usr/bin/env python3
"""
Script to delete a user from Firebase and remove their GitHub connections.

Usage:
    python scripts/delete_user_and_github.py yashkmkrishan@gmail.com
"""

import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError, NotFoundError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.models import *  # Import all models to avoid mapper initialization errors
from app.modules.users.user_model import User
from app.modules.integrations.integration_model import Integration
from app.modules.utils.firebase_setup import FirebaseSetup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_user_by_email(email: str):
    """Get Firebase user by email address."""
    try:
        user = auth.get_user_by_email(email)
        return user
    except NotFoundError:
        logger.error(f"User with email '{email}' not found in Firebase")
        return None
    except Exception as e:
        logger.error(f"Error fetching user by email '{email}': {e}")
        return None


def delete_user_from_firebase(uid: str) -> bool:
    """Delete a user from Firebase Authentication."""
    try:
        auth.delete_user(uid)
        logger.info(f"✓ Successfully deleted user with UID '{uid}' from Firebase")
        return True
    except NotFoundError:
        logger.warning(f"User with UID '{uid}' not found in Firebase (may already be deleted)")
        return False
    except FirebaseError as e:
        logger.error(f"Firebase error deleting user '{uid}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting user '{uid}': {e}")
        return False


def delete_user_from_database(uid: str, db: Session) -> bool:
    """Delete a user from the local database, including all related data."""
    try:
        from app.modules.conversations.conversation.conversation_model import Conversation
        from app.modules.intelligence.prompts.prompt_model import Prompt
        from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
        from app.modules.users.user_preferences_model import UserPreferences
        from app.modules.projects.projects_model import Project

        user = db.query(User).filter(User.uid == uid).first()
        if not user:
            logger.warning(f"User with UID '{uid}' not found in database")
            return False

        # Delete related data first
        # Conversations (should cascade, but delete explicitly to be safe)
        conversations_count = db.query(Conversation).filter(Conversation.user_id == uid).count()
        if conversations_count > 0:
            db.query(Conversation).filter(Conversation.user_id == uid).delete()
            logger.info(f"  Deleted {conversations_count} conversation(s)")

        # Custom Agents
        agents_count = db.query(CustomAgent).filter(CustomAgent.user_id == uid).count()
        if agents_count > 0:
            db.query(CustomAgent).filter(CustomAgent.user_id == uid).delete()
            logger.info(f"  Deleted {agents_count} custom agent(s)")

        # Prompts
        prompts_count = db.query(Prompt).filter(Prompt.created_by == uid).count()
        if prompts_count > 0:
            db.query(Prompt).filter(Prompt.created_by == uid).delete()
            logger.info(f"  Deleted {prompts_count} prompt(s)")

        # Projects (need to delete search_indices first)
        from app.modules.search.search_models import SearchIndex
        projects = db.query(Project).filter(Project.user_id == uid).all()
        projects_count = len(projects)
        if projects_count > 0:
            # Delete search_indices for each project first
            for project in projects:
                search_indices_count = db.query(SearchIndex).filter(SearchIndex.project_id == project.id).count()
                if search_indices_count > 0:
                    db.query(SearchIndex).filter(SearchIndex.project_id == project.id).delete()
                    logger.info(f"  Deleted {search_indices_count} search index entries for project {project.id}")
            # Now delete the projects
            db.query(Project).filter(Project.user_id == uid).delete()
            logger.info(f"  Deleted {projects_count} project(s)")

        # User Preferences
        prefs = db.query(UserPreferences).filter(UserPreferences.user_id == uid).first()
        if prefs:
            db.delete(prefs)
            logger.info(f"  Deleted user preferences")

        # Finally, delete the user
        db.delete(user)
        db.commit()
        logger.info(f"✓ Successfully deleted user with UID '{uid}' from database")
        return True
    except Exception as e:
        logger.error(f"Error deleting user '{uid}' from database: {e}")
        db.rollback()
        return False


def remove_github_from_provider_info(uid: str, db: Session) -> bool:
    """Remove GitHub token from user's provider_info."""
    try:
        user = db.query(User).filter(User.uid == uid).first()
        if not user:
            logger.warning(f"User with UID '{uid}' not found in database")
            return False

        if user.provider_info and isinstance(user.provider_info, dict):
            # Remove GitHub-related tokens
            removed = False
            if "access_token" in user.provider_info:
                del user.provider_info["access_token"]
                removed = True
            if "github_token" in user.provider_info:
                del user.provider_info["github_token"]
                removed = True
            if "github_oauth_token" in user.provider_info:
                del user.provider_info["github_oauth_token"]
                removed = True

            if removed:
                db.commit()
                logger.info(f"✓ Removed GitHub tokens from provider_info for user '{uid}'")
                return True
            else:
                logger.info(f"No GitHub tokens found in provider_info for user '{uid}'")
                return False
        else:
            logger.info(f"User '{uid}' has no provider_info or it's not a dict")
            return False
    except Exception as e:
        logger.error(f"Error removing GitHub from provider_info for user '{uid}': {e}")
        db.rollback()
        return False


def delete_github_integrations(uid: str, db: Session) -> int:
    """Delete all GitHub integrations for a user."""
    try:
        integrations = db.query(Integration).filter(
            Integration.created_by == uid,
            Integration.integration_type == "github"
        ).all()

        count = len(integrations)
        if count > 0:
            for integration in integrations:
                db.delete(integration)
            db.commit()
            logger.info(f"✓ Deleted {count} GitHub integration(s) for user '{uid}'")
        else:
            logger.info(f"No GitHub integrations found for user '{uid}'")

        return count
    except Exception as e:
        logger.error(f"Error deleting GitHub integrations for user '{uid}': {e}")
        db.rollback()
        return 0


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python scripts/delete_user_and_github.py <email>")
        sys.exit(1)

    email = sys.argv[1].strip()

    # Check if we're in development mode
    if os.getenv("isDevelopmentMode") == "enabled":
        logger.warning(
            "Development mode is enabled. Firebase operations may not work as expected."
        )

    # Initialize Firebase
    try:
        logger.info("Initializing Firebase...")
        FirebaseSetup.firebase_init()
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        logger.error(
            "Make sure firebase_service_account.txt or firebase_service_account.json exists"
        )
        sys.exit(1)

    # Get user information
    logger.info(f"Looking up user by email: {email}")
    firebase_user = get_user_by_email(email)

    if not firebase_user:
        logger.error("User not found in Firebase. Checking database...")
        # Try to find user in database even if not in Firebase
        db = SessionLocal()
        try:
            db_user = db.query(User).filter(User.email == email).first()
            if db_user:
                uid = db_user.uid
                logger.info(f"Found user in database with UID: {uid}")
            else:
                logger.error("User not found in Firebase or database. Cannot proceed.")
                sys.exit(1)
        finally:
            db.close()
    else:
        uid = firebase_user.uid
        logger.info(f"Found user:")
        logger.info(f"  UID: {firebase_user.uid}")
        logger.info(f"  Email: {firebase_user.email}")
        logger.info(f"  Display Name: {firebase_user.display_name}")

    # Initialize database session
    db = SessionLocal()
    try:
        # Step 1: Delete GitHub integrations
        logger.info("\n=== Step 1: Deleting GitHub integrations ===")
        integration_count = delete_github_integrations(uid, db)

        # Step 2: Remove GitHub tokens from provider_info
        logger.info("\n=== Step 2: Removing GitHub tokens from provider_info ===")
        remove_github_from_provider_info(uid, db)

        # Step 3: Delete user from database
        logger.info("\n=== Step 3: Deleting user from database ===")
        delete_user_from_database(uid, db)

    finally:
        db.close()

    # Step 4: Delete user from Firebase (do this last)
    logger.info("\n=== Step 4: Deleting user from Firebase ===")
    if firebase_user:
        delete_user_from_firebase(uid)

    logger.info("\n=== Deletion completed ===")
    logger.info(f"Summary:")
    logger.info(f"  - GitHub integrations deleted: {integration_count}")
    logger.info(f"  - User deleted from database: ✓")
    if firebase_user:
        logger.info(f"  - User deleted from Firebase: ✓")


if __name__ == "__main__":
    main()
