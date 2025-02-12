import json
import logging
import time
from typing import Generator, List
import requests
import aiohttp
import asyncio
from potpie.utility import Utility
import random


class ApiWrapper:
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

    def available_agents(self, system_agent: bool = True):
        """Fetches available agents from the API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/list-available-agents/",
                params={"list_system_agents": system_agent},
            )
            if response.status_code != 200:
                logging.error("Failed to fetch agents.")
                raise
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
                            continue

                        await asyncio.sleep(0)

            except aiohttp.ClientError as e:
                logging.error(f"HTTP Request failed: {e}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise


#     def conversation_history():
#         pass


def first_try(api: ApiWrapper):

    project_id = api.parse_project(
        repo_path="/home/deepesh/Development/public/opensource/test-mongo"
    )

    print(project_id)

    while True:

        a = api.parse_status(project_id)

        if a in ["ready", "error"]:
            break

        print(a)
        time.sleep(4)

    agents = api.available_agents()

    selected_agent = random.choice([agent["id"] for agent in agents])
    print(selected_agent)

    conversation_id = api.create_conversation(
        agent_id_list=[selected_agent],
        project_id_list=[project_id],
        title="My first Conversation",
    )

    print(conversation_id)

    while True:
        a = input("Enter the message chat with bots, quite or q to quit: ")

        if a in ["q", "quit"]:
            break

        print(a)
        asyncio.run(
            api.interact_with_agent(
                conversation_id="conversation_id",
                content=a,
            )
        )


def second_try(api: ApiWrapper):

    while True:
        a = input("Enter the message chat with bots, quite or q to quit: ")

        if a in ["q", "quit"]:
            break

        print(a)
        asyncio.run(
            api.interact_with_agent(
                conversation_id="0194f9a0-8402-7687-ba25-aac7e860aea2",
                content=a,
            )
        )


if __name__ == "__main__":
    api = ApiWrapper()

    second_try(api)
