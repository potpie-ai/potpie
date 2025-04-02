"""
Test script for Linear client integration with user-specific API keys.
"""
import asyncio
import argparse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.intelligence.tools.linear_tools.linear_client import (
    get_linear_client, 
    get_linear_client_for_user
)
from app.modules.key_management.secret_manager import SecretStorageHandler

async def test_get_client_with_user_id(user_id: str):
    """Test retrieving a Linear client for a specific user"""
    print(f"Testing Linear client for user: {user_id}")
    
    # Get a database session
    db = next(get_db())
    
    # Check if the user has a Linear API key
    exists = await SecretStorageHandler.check_secret_exists(
        service="linear",
        customer_id=user_id,
        service_type="integration",
        db=db
    )
    
    if not exists:
        print(f"⚠️ User {user_id} does not have a Linear API key configured.")
        print("Please configure it using the SecretManager integration API.")
        return
    
    # Try to get the client
    try:
        client = await get_linear_client_for_user(user_id, db)
        print(f"✅ Successfully retrieved Linear client for user {user_id}")
        
        # Test the client with a simple API call
        try:
            # This is just a test - if we can query the API at all, it works
            user_data = client.execute_query("""
                query {
                    viewer {
                        id
                        name
                        email
                    }
                }
            """)
            print(f"✅ API test successful. Connected as: {user_data['viewer']['name']} ({user_data['viewer']['email']})")
        except Exception as e:
            print(f"❌ API test failed: {str(e)}")
    except Exception as e:
        print(f"❌ Error getting client: {str(e)}")

def run_tests():
    """Parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Test the Linear client integration")
    parser.add_argument("--user-id", required=True, help="User ID to test with")
    
    args = parser.parse_args()
    asyncio.run(test_get_client_with_user_id(args.user_id))

if __name__ == "__main__":
    run_tests() 