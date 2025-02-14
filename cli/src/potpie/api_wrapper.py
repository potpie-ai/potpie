"""
This module provides an API wrapper for interacting with a remote service.
It includes functionality for parsing projects, managing conversations, 
retrieving agents, and handling errors properly.
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

    # ? Parsing
    def parse_project(self, repo_path: str, branch_name: str = "main") -> str:
        """Parse a project using the API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/parse",
                json={
                    "repo_path": repo_path,
                    "branch_name": branch_name,
                },
            )
            if response.status_code != 200:
                logging.error("Failed to parse project.")
                raise Exception("Failed to parse project.")
            return response.json()["project_id"]

        except requests.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def parse_status(self, project_id: int) -> str:
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
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def get_list_of_projects(self) -> List:
        """Fetches list of projects from the API."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/projects/list")
            if response.status_code != 200:
                logging.error("Failed to fetch projects.")
                raise Exception("Failed to fetch projects.")
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def delete_project(self, project_id: int) -> int:
        """delete the project using the API."""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/projects,",
                params={"project_id": project_id},
            )
            if response.status_code != 200:
                logging.error("Failed to delete project.")
                raise Exception("Failed to delete project.")
            return response.status_code
        except requests.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def available_agents(self, system_agent: bool = True):
        """Fetches available agents from the API."""
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
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def get_conversation(self) -> str:
        """Fetches conversation using the API."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/user/conversations/")
            if response.status_code != 200:
                logging.error("Failed to fetch conversation.")
                raise Exception("Failed to fetch conversation.")
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def create_conversation(
        self, agent_id_list: List, project_id_list: List, title: str
    ) -> str:
        """create conversation using the API."""
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
                    f"Failed to create conversation. response: {response.json()}"
                )
                raise Exception("Failed to create conversation.")
            return response.json()["conversation_id"]

        except requests.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    async def interact_with_agent(self, conversation_id: str, content: str):
        """Start an interaction with an agent using the API (streaming response)."""

        url = f"{self.base_url}/api/v1/conversations/{conversation_id}/message/"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json={"content": content}) as response:
                    print(f"Status: {response.status}")

                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Failed to interact with agent: {error_text}")
                        raise Exception("Failed to interact with agent.")

                    async for line in response.content.iter_chunks():
                        try:
                            json_string = line[0].decode()
                            data = json.loads(json_string)
                            yield data["message"]
                        except json.JSONDecodeError as e:
                            # Ignore this because they are just empty man
                            logging.debug(
                                f"Received empty or invalid JSON chunk, skipping {e}"
                            )
                            continue

                        await asyncio.sleep(0)

            except aiohttp.ClientError as e:
                logging.error(f"HTTP Request failed: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise
