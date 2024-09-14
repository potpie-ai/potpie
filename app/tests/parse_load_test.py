import nest_asyncio
nest_asyncio.apply()

import time
from locust import HttpUser, task, between, events
import requests
import logging
from locust.exception import StopUser

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants (consider moving these to a separate config file)
GITHUB_TOKEN = ''
GITHUB_REPO = 'vineetshar/gymhero'
GITHUB_API_URL = 'https://api.github.com'
REST_API_URL = 'http://mom-server.momentum.com/api/v1'
BRANCH_PREFIX = 'potpie-test-6'
NUMBER_OF_BRANCHES = 10
REST_API_BEARER_TOKEN = ''
class TestParseAPI(HttpUser):
    wait_time = between(0, 1)
    host = REST_API_URL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.github_headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.rest_headers = {
            "Authorization": f"Bearer {REST_API_BEARER_TOKEN}",
            "Accept": "application/json",
        }
        self.parsed_branches = {}
        self.branch_exists = False

    def on_start(self):
        user_id = self.environment.runner.user_count if self.environment.runner else 0
        self.branches = [f"{BRANCH_PREFIX}-{user_id}-{i}" for i in range(NUMBER_OF_BRANCHES)]
        self.parsed_branches = {branch: False for branch in self.branches}

        initial_rate_limit = self.check_rate_limit()
        self.print_rate_limit(initial_rate_limit, "Initial Rate Limit")

        if initial_rate_limit["remaining"] == 0:
            logger.info("Rate limit exceeded. Exiting...")
            raise StopUser()

        if self.any_branch_exists():
            logger.info("One or more branches already exist. Exiting...")
            self.branch_exists = True
            raise StopUser()
        else:
            self.branch_exists = False
            self.create_all_branches()

    def on_stop(self):
        if not self.branch_exists:
            final_rate_limit = self.check_rate_limit()
            self.print_rate_limit(final_rate_limit, "Final Rate Limit")
            self.write_parsed_branches_to_file()

    def any_branch_exists(self):
        for branch_name in self.branches:
            branch_url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/git/refs/heads/{branch_name}"
            with self.client.get(branch_url, headers=self.github_headers, catch_response=True, name="check_branch") as response:
                if response.status_code == 200:
                    logger.info(f"Branch {branch_name} already exists.")
                    return True
        return False

    def create_all_branches(self):
        for branch_name in self.branches:
            self.create_branch(branch_name)

    def create_branch(self, branch_name):
        create_url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/git/refs"
        main_branch_url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/git/refs/heads/main"
        
        with self.client.get(main_branch_url, headers=self.github_headers, catch_response=True, name="get_main_branch") as response:
            if response.status_code == 200:
                main_branch_sha = response.json().get("object", {}).get("sha")
                if main_branch_sha:
                    data = {
                        "ref": f"refs/heads/{branch_name}",
                        "sha": main_branch_sha,
                    }
                    with self.client.post(create_url, headers=self.github_headers, json=data, catch_response=True, name="create_branch") as create_response:
                        if create_response.status_code == 201:
                            logger.info(f"Branch {branch_name} created successfully.")
                        else:
                            logger.info(f"Failed to create branch {branch_name}. Status code: {create_response.status_code}")
                else:
                    logger.info(f"Failed to get SHA of main branch. Response: {response.json()}")
            else:
                logger.info(f"Failed to get main branch. Status code: {response.status_code}")

    @task
    def simulate_requests(self):
        if self.branch_exists:
            logger.info("Branch exists, skipping task.")
            return

        logger.info("Starting to simulate REST API requests")
        for branch_name in self.branches:
            if not self.parsed_branches[branch_name]:
                self.parse_branch(branch_name)

        if all(self.parsed_branches.values()):
            logger.info("All branches parsed successfully. Writing results to file...")
            self.write_parsed_branches_to_file()
            raise StopUser()

    def parse_branch(self, branch_name):
        payload = {
            "repo_name": GITHUB_REPO,
            "branch_name": branch_name,
        }

        logger.info(f"Hitting parse API for branch {branch_name}")
        with self.client.post("/parse", headers=self.rest_headers, json=payload, catch_response=True, name="parse") as response:
            if response.status_code == 200:
                self.parsed_branches[branch_name] = True
                logger.info(f"Branch {branch_name} parsed successfully.")
            else:
                logger.info(f"Failed to parse branch {branch_name}. Status code: {response.status_code}, Error: {response.text}")

    def check_rate_limit(self):
        with self.client.get(f"{GITHUB_API_URL}/rate_limit", headers=self.github_headers, catch_response=True, name="check_rate_limit") as response:
            if response.status_code == 200:
                return response.json()["resources"]["core"]
            else:
                return {"limit": 0, "remaining": 0, "reset": 0}

    def print_rate_limit(self, rate_limit_data, title):
        formatted_rate_limit = {
            "Limit": rate_limit_data['limit'],
            "Remaining": rate_limit_data['remaining'],
            "Reset Time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_limit_data['reset']))
        }
        logger.info(f"\n{title}:")
        for key, value in formatted_rate_limit.items():
            logger.info(f"{key}: {value}")

    def write_parsed_branches_to_file(self):
        with open("result.txt", "w") as file:
            for branch, parsed in self.parsed_branches.items():
                if parsed:
                    file.write(f"{branch}\n")

# Optional: Add custom event handlers if needed
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("A new test is starting")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("Test is ending")