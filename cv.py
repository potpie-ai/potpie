import requests
import time
import os

BASE_URL = "http://localhost:8001/api/v1"

def parse_project():
    """Parse the project using the parsing API"""
    repo_path = os.path.abspath("C:/AutomaLar/toParse/potpie")
    
    parse_data = {
        "repo_path": repo_path,
        "branch_name": "main"
    }

    headers = {
        "Content-Type": "application/json",
        "isDevelopmentMode": "enabled"
    }

    response = requests.post(
        f"{BASE_URL}/parse",
        json=parse_data,
        headers=headers
    )
    
    print(f"Parse Status Code: {response.status_code}")
    print(f"Parse Response: {response.text}")
    return response.json()

def check_parsing_status(project_id):
    """Check the parsing status of the project"""
    # Add isDevelopmentMode header
    headers = {
        "Content-Type": "application/json",
        "isDevelopmentMode": "enabled"
    }

    response = requests.get(
        f"{BASE_URL}/parsing-status/{project_id}",
        headers=headers
    )
    
    print(f"Status Check Response: {response.text}")
    return response.json()

def list_available_agents():
    """List all available agents"""
    # Add isDevelopmentMode header
    headers = {
        "Content-Type": "application/json",
        "isDevelopmentMode": "enabled"
    }

    response = requests.get(
        f"{BASE_URL}/list-available-agents/",
        headers=headers
    )
    
    print(f"Available Agents: {response.text}")
    return response.json()

def main():
    try:
        # Step 1: Parse the project
        print("\n1. Parsing project...")
        parse_result = parse_project()
        
        # Get project_id from parse response
        project_id = parse_result.get("project_id")
        if not project_id:
            raise Exception("Failed to get project_id from parse response")
        
        # Step 2: Check parsing status (with retry)
        print("\n2. Checking parsing status...")
        max_retries = 5
        for i in range(max_retries):
            status = check_parsing_status(project_id)
            if status.get("status") == "READY":
                print("Parsing completed successfully!")
                break
            print(f"Parsing in progress... (attempt {i+1}/{max_retries})")
            time.sleep(2)  # Wait 2 seconds before checking again
            
        # Step 3: List available agents
        print("\n3. Listing available agents...")
        agents = list_available_agents()

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()