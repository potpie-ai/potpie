#!/usr/bin/env python3
"""
Script to clear/purge tasks from Celery queues.

Usage:
    python scripts/clear_celery_queue.py                    # Clear all queues
    python scripts/clear_celery_queue.py --queue <name>      # Clear specific queue
    python scripts/clear_celery_queue.py --list              # List all queues
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.celery.celery_app import celery_app, logger

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_all_queues():
    """Get all configured queues from Celery."""
    queue_prefix = os.getenv("CELERY_QUEUE_NAME", "staging")

    queues = [
        f"{queue_prefix}_process_repository",
        f"{queue_prefix}_agent_tasks",
        "external-event",
    ]

    return queues


def purge_queue(queue_name: str) -> int:
    """Purge all tasks from a specific queue."""
    try:
        broker = celery_app.broker_connection()
        with broker.channel() as channel:
            # Declare queue to ensure it exists
            channel.queue_declare(queue_name, durable=True)
            # Purge the queue
            purged_count = channel.queue_purge(queue_name)
            logger.info(f"Purged {purged_count} tasks from queue '{queue_name}'")
            return purged_count
    except Exception as e:
        logger.error(f"Failed to purge queue '{queue_name}': {str(e)}")
        return 0


def list_queues():
    """List all configured queues and their task counts."""
    queues = get_all_queues()

    print("\nConfigured Celery Queues:")
    print("-" * 60)

    for queue_name in queues:
        try:
            # Try to get the queue length
            broker = celery_app.broker_connection()
            with broker.channel() as channel:
                # Declare queue passively to get info without creating it
                try:
                    queue_info = channel.queue_declare(
                        queue_name, passive=True, durable=True
                    )
                    # For Redis, message_count is available in the method
                    if hasattr(queue_info, "method") and hasattr(
                        queue_info.method, "message_count"
                    ):
                        task_count = queue_info.method.message_count
                    else:
                        task_count = 0
                    print(f"  {queue_name:40s} - {task_count} tasks")
                except Exception:
                    # Queue doesn't exist yet
                    print(f"  {queue_name:40s} - 0 tasks (queue not created)")
        except Exception as e:
            print(f"  {queue_name:40s} - Error: {str(e)}")

    print("-" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clear/purge tasks from Celery queues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Clear all queues
  %(prog)s --queue staging_agent_tasks  # Clear specific queue
  %(prog)s --list              # List all queues and their task counts
        """,
    )

    parser.add_argument(
        "--queue", type=str, help="Specific queue name to purge (default: all queues)"
    )

    parser.add_argument(
        "--list", action="store_true", help="List all queues and their task counts"
    )

    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # List queues if requested
    if args.list:
        list_queues()
        return

    # Get queues to purge
    if args.queue:
        queues_to_purge = [args.queue]
    else:
        queues_to_purge = get_all_queues()

    # Confirm before purging
    if not args.yes:
        print("\n⚠️  WARNING: This will purge all tasks from the following queue(s):")
        for queue in queues_to_purge:
            print(f"   - {queue}")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

    # Purge queues
    total_purged = 0
    print("\nPurging queues...")
    print("-" * 60)

    for queue_name in queues_to_purge:
        purged_count = purge_queue(queue_name)
        if purged_count >= 0:
            total_purged += purged_count
            print(f"✓ Purged {purged_count} tasks from '{queue_name}'")
        else:
            print(f"✗ Failed to purge '{queue_name}'")

    print("-" * 60)
    print(f"\n✓ Total tasks purged: {total_purged}")
    print("Done.\n")


if __name__ == "__main__":
    main()
