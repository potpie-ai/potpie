"""
This module provides an API wrapper for interacting with a remote service.

It includes functionality for:
- Parsing projects
- Managing conversations
- Retrieving agents
- Handling errors properly

The API wrapper supports both synchronous (requests) and asynchronous (aiohttp) HTTP requests.
Logging is implemented with lazy string formatting for better performance.

"""

import json
import logging
import asyncio
from typing import List

import requests
import aiohttp

from potpie.utility import Utility

logging.basicConfig(level=logging.INFO)


class ApiWrapper:
    """A wrapper around the API for managing projects, conversations, and agents."""

    def __init__(self):
        self.base_url = Utility.base_url()
        self.user_id = Utility.get_user_id()

    def parse_project(self, repo_path: str, branch_name: str = "main") -> str:
        """Parse a project using the API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/parse",
                json={"repo_path": repo_path, "branch_name": branch_name},
            )
            if response.status_code != 200:
                logging.error("Failed to parse project.")
                raise Exception("Failed to parse project.")
            return response.json()["project_id"]

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def parse_status(self, project_id: str) -> str:
        """Monitor parsing status using the API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/parsing-status/{project_id}"
            )
            if response.status_code != 200:
                logging.error("Failed to fetch parsing status.")
                raise Exception("Failed to fetch parsing status.")
            return response.json()["status"]

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def get_list_of_projects(self) -> List:
        """Fetch list of projects from the API."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/projects/list")
            if response.status_code != 200:
                logging.error("Failed to fetch projects.")
                raise Exception("Failed to fetch projects.")
            return response.json()

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def delete_project(self, project_id: int) -> int:
        """Delete the project using the API."""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/projects", params={"project_id": project_id}
            )
            if response.status_code != 200:
                logging.error("Failed to delete project.")
                raise Exception("Failed to delete project.")
            return response.status_code

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def available_agents(self, system_agent: bool = True):
        """Fetch available agents from the API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/list-available-agents/",
                params={"list_system_agents": system_agent},
            )
            if response.status_code != 200:
                logging.error("Failed to fetch agents.")
                raise Exception("Failed to fetch agents.")
            return response.json()

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def get_conversation(self) -> dict:
        """Fetch conversation using the API."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/user/conversations/")
            if response.status_code != 200:
                logging.error("Failed to fetch conversation.")
                raise Exception("Failed to fetch conversation.")
            return response.json()

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    def create_conversation(
        self, agent_id_list: List, project_id_list: List, title: str
    ) -> str:
        """Create conversation using the API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/conversations/",
                json={
                    "user_id": self.user_id,
                    "title": title,
                    "status": "active",
                    "project_ids": project_id_list,
                    "agent_ids": agent_id_list,
                },
            )
            if response.status_code != 200:
                logging.error(
                    "Failed to create conversation. Response: %s", response.json()
                )
                raise Exception("Failed to create conversation.")
            return response.json()["conversation_id"]

        except requests.RequestException as e:
            logging.error("Network error occurred: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise

    async def interact_with_agent(self, conversation_id: str, content: str):
        """Start an interaction with an agent using the API (streaming response)."""
        url = f"{self.base_url}/api/v1/conversations/{conversation_id}/message/"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json={"content": content}) as response:
                    print("Status: %s", response.status)

                    if response.status != 200:
                        error_text = await response.text()
                        logging.error("Failed to interact with agent: %s", error_text)
                        raise Exception("Failed to interact with agent.")

                    async for line in response.content.iter_chunks():
                        try:
                            json_string = line[0].decode()
                            data = json.loads(json_string)
                            yield data["message"]
                        except json.JSONDecodeError as e:
                            logging.debug(
                                "Received empty or invalid JSON chunk, skipping %s", e
                            )
                            continue

                        await asyncio.sleep(0)

            except aiohttp.ClientError as e:
                logging.error("HTTP Request failed: %s", e)
                raise
            except Exception as e:
                logging.error("Unexpected error: %s", e)
                raise
