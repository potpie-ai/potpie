"""
Test script for the Analytics API.

Usage:
    python test_analytics_api.py --user-id YOUR_USER_ID --token YOUR_AUTH_TOKEN

    Or set environment variables:
    export TEST_USER_ID=your_user_id
    export TEST_AUTH_TOKEN=your_auth_token
    python test_analytics_api.py
"""

import argparse
import json
import os
import sys

import requests


def test_analytics_endpoint(base_url: str, user_id: str, auth_token: str, days: int = 30):
    """Test the analytics endpoint."""
    
    print(f"\n{'='*60}")
    print(f"Testing Analytics API")
    print(f"{'='*60}")
    print(f"Base URL: {base_url}")
    print(f"User ID: {user_id}")
    print(f"Days: {days}")
    print(f"{'='*60}\n")
    
    # Test 1: Get aggregated analytics
    print("Test 1: GET /api/v1/analytics/user/{user_id}")
    print("-" * 60)
    
    url = f"{base_url}/api/v1/analytics/user/{user_id}"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"days": days}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ SUCCESS! Analytics data retrieved.\n")
            
            # Print summary
            print("Summary:")
            print(f"  Total Cost: ${data['summary']['total_cost']:.4f}")
            print(f"  Total Agent Runs: {data['summary']['total_agent_runs']}")
            print(f"  Avg Duration: {data['summary']['avg_duration_ms']:.2f}ms")
            print(f"  Success Rate: {data['summary']['success_rate']*100:.2f}%")
            
            # Print period
            print(f"\nPeriod:")
            print(f"  Start: {data['period']['start']}")
            print(f"  End: {data['period']['end']}")
            print(f"  Days: {data['period']['days']}")
            
            # Print daily costs (first 5)
            if data['daily_costs']:
                print(f"\nDaily Costs (showing first 5):")
                for cost in data['daily_costs'][:5]:
                    print(f"  {cost['date']}: ${cost['cost']:.4f} ({cost['run_count']} runs)")
            else:
                print("\n⚠️  No daily cost data found")
            
            # Print outcomes
            if data['agent_runs_by_outcome']:
                print(f"\nAgent Runs by Outcome:")
                for outcome, count in data['agent_runs_by_outcome'].items():
                    print(f"  {outcome}: {count}")
            else:
                print("\n⚠️  No agent run data found")
            
            # Print conversation stats (first 5)
            if data['conversation_stats']:
                print(f"\nConversation Stats (showing first 5):")
                for stat in data['conversation_stats'][:5]:
                    print(f"  {stat['date']}: {stat['count']} conversations")
            else:
                print("\n⚠️  No conversation data found")
            
            # Save full response to file
            output_file = "analytics_response.json"
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n📄 Full response saved to: {output_file}")
            
            return True
            
        else:
            print(f"\n❌ FAILED!")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_raw_spans_endpoint(base_url: str, user_id: str, auth_token: str, days: int = 7, limit: int = 10):
    """Test the raw spans endpoint."""
    
    print(f"\n\n{'='*60}")
    print(f"Test 2: GET /api/v1/analytics/user/{user_id}/raw")
    print(f"{'='*60}\n")
    
    url = f"{base_url}/api/v1/analytics/user/{user_id}/raw"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"days": days, "limit": limit}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            spans = response.json()
            
            print(f"\n✅ SUCCESS! Retrieved {len(spans)} raw spans.\n")
            
            if spans:
                print(f"Sample spans (first 3):")
                for i, span in enumerate(spans[:3], 1):
                    print(f"\n  Span {i}:")
                    print(f"    Name: {span.get('span_name', 'N/A')}")
                    print(f"    Timestamp: {span.get('start_timestamp', 'N/A')}")
                    print(f"    Duration: {span.get('duration_ms', 'N/A')}ms")
                    print(f"    Attributes: {list(span.get('attributes', {}).keys())[:5]}")
            else:
                print("\n⚠️  No raw spans found")
            
            return True
            
        else:
            print(f"\n❌ FAILED!")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    """Main test function."""
    
    parser = argparse.ArgumentParser(description="Test the Analytics API")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8001"),
        help="Base URL of the API (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("TEST_USER_ID"),
        help="User ID to test with"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("TEST_AUTH_TOKEN"),
        help="Authentication token"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to query (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.user_id:
        print("❌ Error: --user-id is required (or set TEST_USER_ID environment variable)")
        sys.exit(1)
    
    if not args.token:
        print("❌ Error: --token is required (or set TEST_AUTH_TOKEN environment variable)")
        sys.exit(1)
    
    # Run tests
    success1 = test_analytics_endpoint(
        base_url=args.base_url,
        user_id=args.user_id,
        auth_token=args.token,
        days=args.days
    )
    
    success2 = test_raw_spans_endpoint(
        base_url=args.base_url,
        user_id=args.user_id,
        auth_token=args.token,
        days=7,
        limit=10
    )
    
    # Print summary
    print(f"\n\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Aggregated Analytics: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Raw Spans: {'✅ PASSED' if success2 else '❌ FAILED'}")
    print(f"{'='*60}\n")
    
    if success1 and success2:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
