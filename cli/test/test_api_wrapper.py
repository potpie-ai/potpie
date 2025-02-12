import sys
import os
import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potpie.api_wrapper import ApiWrapper


@pytest.fixture(scope="module")
def api_wrapper() -> ApiWrapper:
    return ApiWrapper()


# parse project
class MockFailureResponse:
    status_code = 422

    @staticmethod
    def json():
        return {
            "detail": [{"loc": ["<string>"], "msg": "<string>", "type": "<string>"}]
        }


class MockParseSuccessResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"project_id": 1, "status": "submitted"}


@pytest.mark.parametrize(
    "reponse_class, expected, should_raise",
    [(MockParseSuccessResponse, 1, False), (MockFailureResponse, None, True)],
)
def test_parse_project(api_wrapper, monkeypatch, reponse_class, expected, should_raise):

    def mock_post(*args, **kwargs):
        return reponse_class()

    monkeypatch.setattr(requests, "post", mock_post)

    if should_raise:
        with pytest.raises(Exception):
            api_wrapper.parse_project("repo_path")
    else:
        assert api_wrapper.parse_project("repo_path") == expected


class MockParseStatusSuccessIResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "error", "latest": False}


class MockParseStatusSuccessIIResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "submitted", "latest": False}


class MockParseStatusSuccessIIIResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "parsed", "latest": False}


class MockParseStatusSuccessIVResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "ready", "latest": False}


@pytest.mark.parametrize(
    "reponse_class, expected, should_raise",
    [
        (MockParseStatusSuccessIResponse, "error", False),
        (MockParseStatusSuccessIIResponse, "submitted", False),
        (MockParseStatusSuccessIIIResponse, "parsed", False),
        (MockParseStatusSuccessIVResponse, "ready", False),
        (MockFailureResponse, None, True),
    ],
)
def test_parse_status(api_wrapper, monkeypatch, reponse_class, expected, should_raise):

    def mock_get(*args, **kwargs):
        return reponse_class()

    monkeypatch.setattr(requests, "get", mock_get)

    if should_raise:
        with pytest.raises(Exception):
            api_wrapper.parse_status(1)
    else:
        assert api_wrapper.parse_status(1) == expected


class MockAvailableAgentsSuccessIResponse:
    status_code = 200

    @staticmethod
    def json():
        return [
            {
                "id": "codebase_qna_agent",
                "name": "Codebase Q&A Agent",
                "description": "An agent specialized in answering questions about the codebase using the knowledge graph and code analysis tools.",
                "status": "SYSTEM",
            },
            {
                "id": "debugging_agent",
                "name": "Debugging with Knowledge Graph Agent",
                "description": "An agent specialized in debugging using knowledge graphs.",
                "status": "SYSTEM",
            },
            {
                "id": "unit_test_agent",
                "name": "Unit Test Agent",
                "description": "An agent specialized in generating unit tests for code snippets for given function names",
                "status": "SYSTEM",
            },
            {
                "id": "integration_test_agent",
                "name": "Integration Test Agent",
                "description": "An agent specialized in generating integration tests for code snippets from the knowledge graph based on given function names of entry points. Works best with Py, JS, TS",
                "status": "SYSTEM",
            },
            {
                "id": "LLD_agent",
                "name": "Low-Level Design Agent",
                "description": "An agent specialized in generating a low-level design plan for implementing a new feature.",
                "status": "SYSTEM",
            },
            {
                "id": "code_changes_agent",
                "name": "Code Changes Agent",
                "description": "An agent specialized in generating blast radius of the code changes in your current branch compared to default branch. Use this for functional review of your code changes. Works best with Py, JS, TS",
                "status": "SYSTEM",
            },
            {
                "id": "code_generation_agent",
                "name": "Code Generation Agent",
                "description": "An agent specialized in generating code for new features or fixing bugs.",
                "status": "SYSTEM",
            },
        ]


@pytest.mark.parametrize(
    "response_class, system_agents, numbers_system_agents, should_raise",
    [
        (MockAvailableAgentsSuccessIResponse, True, 7, False),
        (MockFailureResponse, True, None, True),
    ],
)
def test_available_agents(
    api_wrapper,
    monkeypatch,
    response_class,
    system_agents,
    numbers_system_agents,
    should_raise,
):

    def mock_get(*args, **kwargs):
        return response_class()

    monkeypatch.setattr(requests, "get", mock_get)

    if should_raise:
        with pytest.raises(Exception):
            api_wrapper.available_agents(system_agent=system_agents)
    else:
        assert (
            len(api_wrapper.available_agents(system_agent=system_agents))
            == numbers_system_agents
        )
