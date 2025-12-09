#!/usr/bin/env python3
"""
Script to send test payloads to a webhook URL.

Usage:
    python scripts/test_webhook.py https://5393cace9a3e.ngrok-free.app/api/v1/webhook/d8dc81da-6093-405c-b430-0fad2566e772
"""

import json
import sys
import requests
from typing import Dict, Any
import time

def send_webhook_payload(url: str, payload: Dict[str, Any], description: str) -> None:
    """Send a webhook payload and print the response."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 80)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Webhook-Test-Script/1.0"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"Response Body: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response Body (text): {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
    
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_webhook.py <webhook_url>")
        sys.exit(1)
    
    webhook_url = sys.argv[1].strip()
    
    print(f"Sending test payloads to: {webhook_url}\n")
    
    # Test Payload 1: Simple generic webhook
    send_webhook_payload(
        webhook_url,
        {
            "event": "test",
            "timestamp": int(time.time()),
            "data": {
                "message": "This is a test webhook payload",
                "source": "test_script"
            }
        },
        "Simple Generic Webhook"
    )
    
    time.sleep(1)  # Small delay between requests
    
    # Test Payload 2: GitHub-style webhook
    send_webhook_payload(
        webhook_url,
        {
            "action": "opened",
            "repository": {
                "name": "test-repo",
                "full_name": "test/test-repo"
            },
            "sender": {
                "login": "testuser"
            },
            "ref": "refs/heads/main",
            "commits": [
                {
                    "id": "abc123",
                    "message": "Test commit",
                    "author": {
                        "name": "Test User",
                        "email": "test@example.com"
                    }
                }
            ]
        },
        "GitHub-style Webhook"
    )
    
    time.sleep(1)
    
    # Test Payload 3: Sentry-style webhook
    send_webhook_payload(
        webhook_url,
        {
            "action": "created",
            "data": {
                "event": {
                    "event_id": "test-event-123",
                    "message": "Test error message",
                    "level": "error",
                    "timestamp": time.time()
                },
                "project": {
                    "name": "test-project",
                    "slug": "test-project"
                }
            }
        },
        "Sentry-style Webhook"
    )
    
    time.sleep(1)
    
    # Test Payload 4: Linear-style webhook
    send_webhook_payload(
        webhook_url,
        {
            "action": "create",
            "type": "Issue",
            "data": {
                "id": "test-issue-123",
                "title": "Test Issue",
                "description": "This is a test issue",
                "state": {
                    "name": "Todo"
                },
                "creator": {
                    "name": "Test User",
                    "email": "test@example.com"
                }
            },
            "createdAt": time.time(),
            "updatedAt": time.time()
        },
        "Linear-style Webhook"
    )
    
    time.sleep(1)
    
    # Test Payload 5: Minimal payload
    send_webhook_payload(
        webhook_url,
        {
            "test": True
        },
        "Minimal Payload"
    )
    
    time.sleep(1)
    
    # Test Payload 6: Custom workflow trigger payload
    send_webhook_payload(
        webhook_url,
        {
            "trigger": "webhook",
            "workflow_id": "d8dc81da-6093-405c-b430-0fad2566e772",
            "payload": {
                "custom_field": "custom_value",
                "number": 42,
                "boolean": True,
                "array": [1, 2, 3],
                "object": {
                    "nested": "value"
                }
            }
        },
        "Custom Workflow Trigger Payload"
    )
    
    print(f"\n{'='*80}")
    print("All test payloads sent!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
