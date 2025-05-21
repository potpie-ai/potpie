import unittest
from unittest import mock
import asyncio

# Actual imports for ParsingService and its dependencies
from app.modules.parsing.graph_construction.parsing_service import ParsingService, ParsingServiceError
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest # For repo_details type

# Mock relevant dependencies of ParsingService
from sqlalchemy.orm import Session # For type hinting Session

# Need to ensure app.core.telemetry.get_tracer is available for patching
import app.core.telemetry 

class TestParsingServiceTelemetry(unittest.IsolatedAsyncioTestCase): # Use IsolatedAsyncioTestCase for async methods

    def setUp(self):
        # Mock get_tracer before ParsingService initializes its own tracer
        self.mock_tracer = mock.MagicMock()
        self.mock_span = mock.MagicMock()
        # Configure the context manager mock
        self.mock_tracer.start_as_current_span.return_value.__enter__.return_value = self.mock_span
        self.mock_tracer.start_as_current_span.return_value.__exit__ = mock.MagicMock()


        # Patch get_tracer specifically for the parsing_service module where it's called
        self.get_tracer_patcher = mock.patch(
            'app.modules.parsing.graph_construction.parsing_service.get_tracer', 
            return_value=self.mock_tracer
        )
        self.mock_get_tracer = self.get_tracer_patcher.start()

        # Mock dependencies for ParsingService
        self.mock_db_session = mock.MagicMock(spec=Session)
        
        # Mock internal helper services used by ParsingService
        # These are instantiated within ParsingService __init__ or methods
        self.mock_parse_helper = mock.MagicMock()
        self.mock_project_service = mock.MagicMock()
        self.mock_inference_service = mock.MagicMock()
        self.mock_search_service = mock.MagicMock()
        self.mock_code_provider_service = mock.MagicMock() # Renamed from github_service

        # Patch the constructors of these internal services if they are created inside ParsingService
        self.parse_helper_patcher = mock.patch('app.modules.parsing.graph_construction.parsing_service.ParseHelper', return_value=self.mock_parse_helper)
        self.project_service_patcher = mock.patch('app.modules.parsing.graph_construction.parsing_service.ProjectService', return_value=self.mock_project_service)
        self.inference_service_patcher = mock.patch('app.modules.parsing.graph_construction.parsing_service.InferenceService', return_value=self.mock_inference_service)
        self.search_service_patcher = mock.patch('app.modules.parsing.graph_construction.parsing_service.SearchService', return_value=self.mock_search_service)
        # Assuming CodeProviderService is used and might be named self.github_service or similar internally
        self.code_provider_service_patcher = mock.patch('app.modules.parsing.graph_construction.parsing_service.CodeProviderService', return_value=self.mock_code_provider_service)

        self.mock_parse_helper_constructor = self.parse_helper_patcher.start()
        self.mock_project_service_constructor = self.project_service_patcher.start()
        self.mock_inference_service_constructor = self.inference_service_patcher.start()
        self.mock_search_service_constructor = self.search_service_patcher.start()
        self.mock_code_provider_service_constructor = self.code_provider_service_patcher.start()


        # Instantiate ParsingService
        self.parsing_service = ParsingService(
            db=self.mock_db_session, # Actual name is 'db'
            user_id="test_user_id_init" # Provide dummy user_id for __init__
        )

    def tearDown(self):
        self.get_tracer_patcher.stop()
        self.parse_helper_patcher.stop()
        self.project_service_patcher.stop()
        self.inference_service_patcher.stop()
        self.search_service_patcher.stop()
        self.code_provider_service_patcher.stop()

    # Patch the methods that parse_directory calls internally to prevent actual work
    @mock.patch('app.modules.parsing.graph_construction.parsing_service.ParsingService.analyze_directory', new_callable=mock.AsyncMock)
    # Mocks for setup_project_directory and clone_or_copy_repository are on self.mock_parse_helper
    async def test_parse_directory_success_creates_span(self, mock_analyze_directory):
        # Setup: Provide necessary arguments for parse_directory
        project_id = 123 # project_id is int
        user_id_param = "test_user_parse"
        user_email_param = "test_user@example.com"
        
        # Create a ParsingRequest instance
        repo_details_request = ParsingRequest(
            repo_name="test-repo",
            repo_url="https://github.com/test/test-repo.git", # repo_url is used if clone_url not present
            clone_url="https://github.com/test/test-repo.git",
            branch_name="main",
            provider="github" # provider is required
        )

        # Mock return values for the methods that do the actual work
        # clone_or_copy_repository returns: repo, owner, auth
        self.mock_parse_helper.clone_or_copy_repository = mock.AsyncMock(return_value=(mock.MagicMock(spec=['get_languages']), "test_owner", "test_auth"))
        # setup_project_directory returns: extracted_dir, project_id (can be same as input)
        self.mock_parse_helper.setup_project_directory = mock.AsyncMock(return_value=("/fake/extracted_dir", project_id))
        # detect_repo_language is called if repo is not a GitPython Repo object
        self.mock_parse_helper.detect_repo_language = mock.MagicMock(return_value="python")
        
        # Mock the get_languages method on the repo object if it's called
        # Based on previous instrumentation, repo_details.repo_name and clone_url/repo_url are used.
        # The actual 'repo' object from clone_or_copy might be a GitPython Repo or a custom object.
        # Let's assume it might call get_languages if it's not a plain string.
        mock_repo_obj = mock.MagicMock()
        mock_repo_obj.get_languages.return_value = {"python": 100} # Simulate language detection
        self.mock_parse_helper.clone_or_copy_repository.return_value = (mock_repo_obj, "test_owner", "test_auth")


        mock_analyze_directory.return_value = None # analyze_directory is async

        # Action: Call the instrumented method
        await self.parsing_service.parse_directory(
            repo_details=repo_details_request,
            user_id=user_id_param,
            user_email=user_email_param,
            project_id=project_id
        )

        # Assertions
        self.mock_get_tracer.assert_called_once_with('app.modules.parsing.graph_construction.parsing_service')
        self.mock_tracer.start_as_current_span.assert_called_once_with("repo.parse")
        
        expected_attributes = {
            "project.id": str(project_id), # project_id was converted to str in span
            "repo.name": repo_details_request.repo_name,
            "repo.url": repo_details_request.clone_url, # clone_url is preferred
            "parsing.status": "success"
        }
        
        # Check that set_attribute was called with the expected key-value pairs
        called_attributes = {}
        for call in self.mock_span.set_attribute.call_args_list:
            called_attributes[call[0][0]] = call[0][1]

        for key, value in expected_attributes.items():
            self.assertEqual(called_attributes.get(key), value, f"Attribute {key} does not match")
        
        self.mock_span.record_exception.assert_not_called()

    @mock.patch('app.modules.parsing.graph_construction.parsing_service.ParsingService.analyze_directory', new_callable=mock.AsyncMock)
    async def test_parse_directory_failure_records_exception(self, mock_analyze_directory):
        project_id = 456
        user_id_param = "test_user_fail"
        user_email_param = "test_user_fail@example.com"

        repo_details_request = ParsingRequest(
            repo_name="another-repo",
            repo_url="https://github.com/test/another-repo.git",
            clone_url="https://github.com/test/another-repo.git",
            branch_name="main",
            provider="github"
        )
        
        test_exception = ParsingServiceError("Simulated parsing error in setup")
        # Simulate an error during parsing setup (e.g., in clone_or_copy_repository)
        self.mock_parse_helper.clone_or_copy_repository = mock.AsyncMock(side_effect=test_exception)
        
        # Action & Assert: Call the method and expect it to raise the exception
        # The original code raises HTTPException, but the span records the original error.
        # We expect ParsingServiceError to be recorded if it's one of the custom exceptions.
        # If it's a generic Exception, that would be recorded.
        # The method itself might wrap it in HTTPException for the FastAPI response.
        # For the span, we care about the exception that was recorded.
        
        with self.assertRaises(Exception) as context: # Catch a broader exception due to potential wrapping
            await self.parsing_service.parse_directory(
                repo_details=repo_details_request,
                user_id=user_id_param,
                user_email=user_email_param,
                project_id=project_id
            )
        
        # Assertions for telemetry
        self.mock_tracer.start_as_current_span.assert_called_once_with("repo.parse")
        
        actual_attributes = {call_args[0][0]: call_args[0][1] for call_args in self.mock_span.set_attribute.call_args_list}

        self.assertEqual(actual_attributes.get("project.id"), str(project_id))
        self.assertEqual(actual_attributes.get("repo.name"), repo_details_request.repo_name)
        self.assertEqual(actual_attributes.get("repo.url"), repo_details_request.clone_url)
        self.assertEqual(actual_attributes.get("parsing.status"), "failure")
        
        # Check that the recorded exception is the one we raised
        self.mock_span.record_exception.assert_called_once()
        recorded_exception = self.mock_span.record_exception.call_args[0][0]
        self.assertIsInstance(recorded_exception, ParsingServiceError) # Or the specific exception type raised
        self.assertEqual(str(recorded_exception), str(test_exception))


if __name__ == '__main__':
    asyncio.run(unittest.main())

# Note: If running directly, ensure the environment is set up for app imports.
# Typically, tests are run with pytest or python -m unittest discover.
