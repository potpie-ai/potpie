#!/usr/bin/env python3
"""
Script to clear GitHub and Firebase data for users.

This script can:
1. Clear GitHub provider links from the database
2. Clear Firebase user data from Firestore
3. Optionally delete Firebase Auth users

Usage:
    python scripts/clear_github_firebase.py --user-id <uid>  # Clear for specific user
    python scripts/clear_github_firebase.py --email <email>  # Clear for specific email
    python scripts/clear_github_firebase.py --all             # Clear all GitHub/Firebase data
    python scripts/clear_github_firebase.py --github-only    # Only clear GitHub provider data
    python scripts/clear_github_firebase.py --firebase-only   # Only clear Firebase data
"""

import argparse
import sys
import os
from typing import Optional, List

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from sqlalchemy import text
from firebase_admin import auth, firestore
from firebase_admin.exceptions import NotFoundError

# Import all models to ensure SQLAlchemy relationships are properly mapped
# This must be done before importing get_db to avoid relationship mapping errors
import app.core.models  # noqa: F401 - Import all models in correct order

from app.core.database import get_db
from app.modules.users.user_model import User
from app.modules.auth.auth_provider_model import UserAuthProvider
from app.modules.utils.firebase_setup import FirebaseSetup
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def clear_github_provider(db: Session, user_id: str) -> bool:
    """Clear GitHub provider data for a specific user."""
    try:
        # Find GitHub provider
        github_provider = (
            db.query(UserAuthProvider)
            .filter(
                UserAuthProvider.user_id == user_id,
                UserAuthProvider.provider_type == "firebase_github",
            )
            .first()
        )

        if github_provider:
            logger.info(f"Deleting GitHub provider for user {user_id}")
            db.delete(github_provider)
            db.commit()
            logger.info(f"Successfully deleted GitHub provider for user {user_id}")
            return True
        else:
            logger.info(f"No GitHub provider found for user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Error clearing GitHub provider for user {user_id}: {str(e)}")
        db.rollback()
        return False


def clear_firestore_user(user_id: str) -> bool:
    """Clear user data from Firestore."""
    try:
        db_firestore = firestore.client()
        user_ref = db_firestore.collection("users").document(user_id)
        
        if user_ref.get().exists:
            logger.info(f"Deleting Firestore user document for {user_id}")
            user_ref.delete()
            logger.info(f"Successfully deleted Firestore user document for {user_id}")
            return True
        else:
            logger.info(f"No Firestore user document found for {user_id}")
            return False
    except Exception as e:
        logger.error(f"Error clearing Firestore user {user_id}: {str(e)}")
        return False


def delete_firebase_auth_user(user_id: str) -> bool:
    """Delete user from Firebase Auth."""
    try:
        logger.info(f"Deleting Firebase Auth user {user_id}")
        auth.delete_user(user_id)
        logger.info(f"Successfully deleted Firebase Auth user {user_id}")
        return True
    except NotFoundError:
        logger.info(f"Firebase Auth user {user_id} not found")
        return False
    except Exception as e:
        logger.error(f"Error deleting Firebase Auth user {user_id}: {str(e)}")
        return False


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def clear_user_data(
    db: Session,
    user_id: str,
    clear_github: bool = True,
    clear_firestore: bool = True,
    clear_firebase_auth: bool = False,
) -> dict:
    """Clear all data for a specific user."""
    results = {
        "user_id": user_id,
        "github_cleared": False,
        "firestore_cleared": False,
        "firebase_auth_cleared": False,
    }

    if clear_github:
        results["github_cleared"] = clear_github_provider(db, user_id)

    if clear_firestore:
        results["firestore_cleared"] = clear_firestore_user(user_id)

    if clear_firebase_auth:
        results["firebase_auth_cleared"] = delete_firebase_auth_user(user_id)

    return results


def clear_all_github_providers(db: Session) -> int:
    """Clear all GitHub provider data."""
    try:
        github_providers = (
            db.query(UserAuthProvider)
            .filter(UserAuthProvider.provider_type == "firebase_github")
            .all()
        )

        count = len(github_providers)
        if count > 0:
            logger.info(f"Deleting {count} GitHub provider records")
            for provider in github_providers:
                db.delete(provider)
            db.commit()
            logger.info(f"Successfully deleted {count} GitHub provider records")
        else:
            logger.info("No GitHub provider records found")

        return count
    except Exception as e:
        logger.error(f"Error clearing all GitHub providers: {str(e)}")
        db.rollback()
        return 0


def clear_all_firestore_users() -> int:
    """Clear all user data from Firestore."""
    try:
        db_firestore = firestore.client()
        users_ref = db_firestore.collection("users")
        
        users = users_ref.stream()
        count = 0
        
        for user_doc in users:
            logger.info(f"Deleting Firestore user document {user_doc.id}")
            user_doc.reference.delete()
            count += 1

        logger.info(f"Successfully deleted {count} Firestore user documents")
        return count
    except Exception as e:
        logger.error(f"Error clearing all Firestore users: {str(e)}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Clear GitHub and Firebase data for users"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID (UID) to clear data for",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email address to clear data for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all GitHub/Firebase data",
    )
    parser.add_argument(
        "--github-only",
        action="store_true",
        help="Only clear GitHub provider data",
    )
    parser.add_argument(
        "--firebase-only",
        action="store_true",
        help="Only clear Firebase data (Firestore and Auth)",
    )
    parser.add_argument(
        "--include-firebase-auth",
        action="store_true",
        help="Also delete Firebase Auth users (dangerous!)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Initialize Firebase Admin SDK
    try:
        FirebaseSetup.firebase_init()
        logger.info("Firebase Admin SDK initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}")
        logger.error("Make sure firebase_service_account.json exists")
        sys.exit(1)

    # Get database session
    db = next(get_db())

    try:
        if args.all:
            # Clear all data
            if not args.yes:
                response = input(
                    "WARNING: This will clear ALL GitHub and Firebase data. "
                    "Are you sure? (yes/no): "
                )
                if response.lower() != "yes":
                    logger.info("Operation cancelled")
                    return

            clear_github = not args.firebase_only
            clear_firestore = not args.github_only

            if clear_github:
                count = clear_all_github_providers(db)
                logger.info(f"Cleared {count} GitHub provider records")

            if clear_firestore:
                count = clear_all_firestore_users()
                logger.info(f"Cleared {count} Firestore user documents")

            if args.include_firebase_auth:
                logger.warning("Firebase Auth user deletion not implemented for --all")
                logger.warning("Use --user-id to delete specific Firebase Auth users")

        elif args.user_id:
            # Clear data for specific user
            user = db.query(User).filter(User.uid == args.user_id).first()
            if not user:
                logger.warning(f"User {args.user_id} not found in database")
                logger.info("Attempting to clear Firebase data anyway...")

            clear_github = not args.firebase_only
            clear_firestore = not args.github_only

            if not args.yes:
                actions = []
                if clear_github:
                    actions.append("GitHub provider data")
                if clear_firestore:
                    actions.append("Firestore user data")
                if args.include_firebase_auth:
                    actions.append("Firebase Auth user")

                response = input(
                    f"Clear {', '.join(actions)} for user {args.user_id}? (yes/no): "
                )
                if response.lower() != "yes":
                    logger.info("Operation cancelled")
                    return

            results = clear_user_data(
                db,
                args.user_id,
                clear_github=clear_github,
                clear_firestore=clear_firestore,
                clear_firebase_auth=args.include_firebase_auth,
            )

            logger.info(f"Results for user {args.user_id}:")
            logger.info(f"  GitHub cleared: {results['github_cleared']}")
            logger.info(f"  Firestore cleared: {results['firestore_cleared']}")
            logger.info(f"  Firebase Auth cleared: {results['firebase_auth_cleared']}")

        elif args.email:
            # Clear data for specific email
            user = get_user_by_email(db, args.email)
            if not user:
                logger.error(f"User with email {args.email} not found")
                sys.exit(1)

            clear_github = not args.firebase_only
            clear_firestore = not args.github_only

            if not args.yes:
                actions = []
                if clear_github:
                    actions.append("GitHub provider data")
                if clear_firestore:
                    actions.append("Firestore user data")
                if args.include_firebase_auth:
                    actions.append("Firebase Auth user")

                response = input(
                    f"Clear {', '.join(actions)} for user {args.email} (UID: {user.uid})? (yes/no): "
                )
                if response.lower() != "yes":
                    logger.info("Operation cancelled")
                    return

            results = clear_user_data(
                db,
                user.uid,
                clear_github=clear_github,
                clear_firestore=clear_firestore,
                clear_firebase_auth=args.include_firebase_auth,
            )

            logger.info(f"Results for user {args.email} (UID: {user.uid}):")
            logger.info(f"  GitHub cleared: {results['github_cleared']}")
            logger.info(f"  Firestore cleared: {results['firestore_cleared']}")
            logger.info(f"  Firebase Auth cleared: {results['firebase_auth_cleared']}")

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

