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


def test_analytics_endpoint(base_url: str, auth_token: str, start_date: str = None, end_date: str = None):
    """Test the analytics endpoint (user is derived from auth token)."""
    if start_date is None or end_date is None:
        from datetime import date, timedelta
        end_date = end_date or str(date.today())
        start_date = start_date or str(date.today() - timedelta(days=30))

    print(f"\n{'='*60}")
    print(f"Testing Analytics API")
    print(f"{'='*60}")
    print(f"Base URL: {base_url}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    # Test 1: Get aggregated analytics
    print("Test 1: GET /api/v1/analytics/summary")
    print("-" * 60)

    url = f"{base_url}/api/v1/analytics/summary"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"start_date": start_date, "end_date": end_date}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ SUCCESS! Analytics data retrieved.\n")
            
            # Print summary
            print("Summary:")
            print(f"  Total Cost: ${data['summary']['total_cost']:.4f}")
            print(f"  Total LLM Calls: {data['summary']['total_llm_calls']}")
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


def test_raw_spans_endpoint(base_url: str, auth_token: str, start_date: str = None, end_date: str = None, limit: int = 10):
    """Test the raw spans endpoint (user is derived from auth token)."""
    if start_date is None or end_date is None:
        from datetime import date, timedelta
        end_date = end_date or str(date.today())
        start_date = start_date or str(date.today() - timedelta(days=7))

    print(f"\n\n{'='*60}")
    print("Test 2: GET /api/v1/analytics/raw")
    print(f"{'='*60}\n")

    url = f"{base_url}/api/v1/analytics/raw"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"start_date": start_date, "end_date": end_date, "limit": limit}
    
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
    from datetime import date, timedelta

    parser = argparse.ArgumentParser(description="Test the Analytics API")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8001"),
        help="Base URL of the API (default: http://localhost:8001)"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("TEST_AUTH_TOKEN"),
        help="Authentication token (user is derived from token)"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD (default: 30 days ago)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for raw spans (default: 10)"
    )

    args = parser.parse_args()

    if not args.token:
        print("❌ Error: --token is required (or set TEST_AUTH_TOKEN environment variable)")
        sys.exit(1)

    end = args.end_date or str(date.today())
    start = args.start_date or str(date.today() - timedelta(days=30))

    # Run tests
    success1 = test_analytics_endpoint(
        base_url=args.base_url,
        auth_token=args.token,
        start_date=start,
        end_date=end,
    )

    success2 = test_raw_spans_endpoint(
        base_url=args.base_url,
        auth_token=args.token,
        start_date=start,
        end_date=end,
        limit=args.limit,
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
