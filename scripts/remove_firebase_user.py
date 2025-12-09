#!/usr/bin/env python3
"""
Script to remove a user from Firebase Authentication.

Usage:
    python scripts/remove_firebase_user.py --email user@example.com
    python scripts/remove_firebase_user.py --uid abc123xyz
    python scripts/remove_firebase_user.py --email user@example.com --delete-from-db
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError, NotFoundError

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


def get_user_by_uid(uid: str):
    """Get Firebase user by UID."""
    try:
        user = auth.get_user(uid)
        return user
    except NotFoundError:
        logger.error(f"User with UID '{uid}' not found in Firebase")
        return None
    except Exception as e:
        logger.error(f"Error fetching user by UID '{uid}': {e}")
        return None


def delete_user_from_firebase(uid: str) -> bool:
    """Delete a user from Firebase Authentication."""
    try:
        auth.delete_user(uid)
        logger.info(f"Successfully deleted user with UID '{uid}' from Firebase")
        return True
    except NotFoundError:
        logger.error(f"User with UID '{uid}' not found in Firebase")
        return False
    except FirebaseError as e:
        logger.error(f"Firebase error deleting user '{uid}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting user '{uid}': {e}")
        return False


def delete_user_from_database(uid: str) -> bool:
    """Delete a user from the local database."""
    try:
        from app.core.database import SessionLocal
        from app.modules.users.user_model import User

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.uid == uid).first()
            if user:
                db.delete(user)
                db.commit()
                logger.info(f"Successfully deleted user with UID '{uid}' from database")
                return True
            else:
                logger.warning(f"User with UID '{uid}' not found in database")
                return False
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error deleting user '{uid}' from database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Remove a user from Firebase Authentication"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email address of the user to delete",
    )
    parser.add_argument(
        "--uid",
        type=str,
        help="UID of the user to delete",
    )
    parser.add_argument(
        "--delete-from-db",
        action="store_true",
        help="Also delete the user from the local database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.email and not args.uid:
        parser.error("Either --email or --uid must be provided")

    if args.email and args.uid:
        parser.error("Please provide either --email or --uid, not both")

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
    user = None
    if args.email:
        logger.info(f"Looking up user by email: {args.email}")
        user = get_user_by_email(args.email)
    elif args.uid:
        logger.info(f"Looking up user by UID: {args.uid}")
        user = get_user_by_uid(args.uid)

    if not user:
        logger.error("User not found. Cannot proceed with deletion.")
        sys.exit(1)

    # Display user information
    logger.info(f"Found user:")
    logger.info(f"  UID: {user.uid}")
    logger.info(f"  Email: {user.email}")
    logger.info(f"  Display Name: {user.display_name}")
    logger.info(f"  Email Verified: {user.email_verified}")
    logger.info(f"  Created: {user.user_metadata.creation_timestamp}")

    # Confirm deletion
    if args.dry_run:
        logger.info("DRY RUN: Would delete the following:")
        logger.info(f"  - Firebase user with UID: {user.uid}")
        if args.delete_from_db:
            logger.info(f"  - Database user with UID: {user.uid}")
        return

    # Delete from Firebase
    logger.info(f"Deleting user '{user.uid}' from Firebase...")
    firebase_success = delete_user_from_firebase(user.uid)

    if not firebase_success:
        logger.error("Failed to delete user from Firebase")
        sys.exit(1)

    # Delete from database if requested
    if args.delete_from_db:
        logger.info(f"Deleting user '{user.uid}' from database...")
        db_success = delete_user_from_database(user.uid)
        if not db_success:
            logger.warning("Failed to delete user from database (user may not exist)")

    logger.info("User deletion completed successfully")


if __name__ == "__main__":
    main()
