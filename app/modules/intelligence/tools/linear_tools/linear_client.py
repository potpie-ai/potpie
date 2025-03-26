"""
Linear Python SDK - A simple interface for the Linear GraphQL API.
"""
import json
import requests
from typing import Dict, Any, Optional, List, Union
import os


class LinearClient:
    """Client for interacting with the Linear GraphQL API."""
    
    API_URL = "https://api.linear.app/graphql"
    
    def __init__(self, api_key: str):
        """
        Initialize the Linear API client.
        
        Args:
            api_key (str): Your Linear API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def execute_query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the Linear API.
        
        Args:
            query (str): The GraphQL query or mutation
            variables (Dict[str, Any], optional): Variables for the query
            
        Returns:
            Dict[str, Any]: The response data
            
        Raises:
            Exception: If the request fails or returns errors
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
            
        response = requests.post(
            self.API_URL,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        
        result = response.json()
        
        if "errors" in result:
            raise Exception(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")
            
        return result["data"]
    
    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """
        Fetch an issue by its ID.
        
        Args:
            issue_id (str): The ID of the issue to fetch
            
        Returns:
            Dict[str, Any]: The issue data
        """
        query = """
        query GetIssue($id: String!) {
          issue(id: $id) {
            id
            title
            description
            state {
              id
              name
            }
            assignee {
              id
              name
            }
            team {
              id
              name
            }
            priority
            url
            createdAt
            updatedAt
          }
        }
        """
        
        variables = {"id": issue_id}
        result = self.execute_query(query, variables)
        return result["issue"]
    
    def update_issue(self, issue_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an issue.
        
        Args:
            issue_id (str): The ID of the issue to update
            input_data (Dict[str, Any]): The update data
            
        Returns:
            Dict[str, Any]: The updated issue data
        """
        mutation = """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue {
              id
              title
              description
              state {
                id
                name
              }
              assignee {
                id
                name
              }
              priority
              updatedAt
            }
          }
        }
        """
        
        variables = {
            "id": issue_id,
            "input": input_data
        }
        
        result = self.execute_query(mutation, variables)
        return result["issueUpdate"]
    
    def comment_create(self, issue_id: str, body: str) -> Dict[str, Any]:
        """
        Add a comment to an issue.
        
        Args:
            issue_id (str): The ID of the issue to comment on
            body (str): The content of the comment
            
        Returns:
            Dict[str, Any]: The created comment data
        """
        mutation = """
        mutation CreateComment($input: CommentCreateInput!) {
          commentCreate(input: $input) {
            success
            comment {
              id
              body
              createdAt
              user {
                id
                name
              }
            }
          }
        }
        """
        
        variables = {
            "input": {
                "issueId": issue_id,
                "body": body
            }
        }
        
        result = self.execute_query(mutation, variables)
        return result["commentCreate"]


class LinearClientConfig:
    _instance: Optional['LinearClientConfig'] = None
    _client: Optional[LinearClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LinearClientConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            api_key = os.getenv('LINEAR_API_KEY')
            if not api_key:
                raise ValueError("LINEAR_API_KEY environment variable is not set")
            self._client = LinearClient(api_key)

    @property
    def client(self) -> LinearClient:
        return self._client

def get_linear_client() -> LinearClient:
    return LinearClientConfig().client 