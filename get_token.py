"""
Get a Firebase auth token by logging in.

Usage:
    python get_token.py --email YOUR_EMAIL --password YOUR_PASSWORD
"""

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def get_firebase_token(email: str, password: str) -> dict:
    """Login and get Firebase auth token."""
    
    identity_key = os.getenv("GOOGLE_IDENTITY_TOOL_KIT_KEY")
    
    if not identity_key:
        print("❌ Error: GOOGLE_IDENTITY_TOOL_KIT_KEY not found in .env")
        sys.exit(1)
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={identity_key}"
    
    print(f"\n🔐 Logging in as: {email}")
    
    try:
        response = requests.post(
            url,
            json={
                "email": email,
                "password": password,
                "returnSecureToken": True
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "idToken": data.get("idToken"),
                "refreshToken": data.get("refreshToken"),
                "userId": data.get("localId"),
                "email": data.get("email"),
                "expiresIn": data.get("expiresIn")
            }
        else:
            return {
                "success": False,
                "error": response.json()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Get Firebase auth token")
    parser.add_argument("--email", required=True, help="Your email")
    parser.add_argument("--password", required=True, help="Your password")
    
    args = parser.parse_args()
    
    result = get_firebase_token(args.email, args.password)
    
    if result["success"]:
        print("\n✅ Login successful!\n")
        print("=" * 60)
        print("YOUR CREDENTIALS:")
        print("=" * 60)
        print(f"User ID:    {result['userId']}")
        print(f"Email:      {result['email']}")
        print(f"Expires in: {result['expiresIn']} seconds")
        print("Auth Token: (saved to file, not printed)")
        print("=" * 60)

        # Save for easy copy-paste (do not print full token)
        with open("my_credentials.txt", "w") as f:
            f.write(f"USER_ID={result['userId']}\n")
            f.write(f"AUTH_TOKEN={result['idToken']}\n")

        print("\n💾 Credentials saved to: my_credentials.txt")

        print("\n🚀 To test the Analytics API, run:")
        print(f'\nexport TEST_USER_ID="{result["userId"]}"')
        print('export TEST_AUTH_TOKEN=$(grep AUTH_TOKEN my_credentials.txt | cut -d= -f2-)')
        print("python test_analytics_api.py")

        print("\nOr use curl (token from file):")
        print('curl -X GET "http://localhost:8001/api/v1/analytics/summary?start_date=2026-01-01&end_date=2026-02-12" \\')
        print('  -H "Authorization: Bearer $(grep AUTH_TOKEN my_credentials.txt | cut -d= -f2-)"')
        
    else:
        print(f"\n❌ Login failed!")
        print(f"Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
