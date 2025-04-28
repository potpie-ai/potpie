import os
import pytest
from unittest.mock import Mock, patch
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService

@pytest.fixture
def mock_sql_db():
    return Mock()

# Mock Github Service
@pytest.fixture
def mock_github_service(mock_sql_db):
    service = Mock(spec=GithubService)
    service.get_repo.return_value = {"name": "test-repo", "id": 1}
    service.get_project_structure_async.return_value = {"files": [], "directories": []}
    service.get_file_content.return_value = "test content"
    return service

# Mock Local Repo Service
@pytest.fixture
def mock_local_repo_service(mock_sql_db):
    service = Mock(spec=LocalRepoService)
    service.get_repo.return_value = {"name": "test-repo", "id": 1}
    service.get_project_structure_async.return_value = {"files": [], "directories": []}
    service.get_file_content.return_value = "test content"
    return service

class TestCodeProviderService:
    # Test initialization in development mode
    def test_init_development_mode(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            service = CodeProviderService(mock_sql_db)
            assert isinstance(service.service_instance, LocalRepoService)

    # Test initialization in production mode
    def test_init_production_mode(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}):
            service = CodeProviderService(mock_sql_db)
            assert isinstance(service.service_instance, GithubService)

    @pytest.mark.parametrize("is_dev_mode,service_class", [
        ("enabled", LocalRepoService),
        ("disabled", GithubService)
    ])
    def test_get_repo(self, mock_sql_db, is_dev_mode, service_class):
        with patch.dict(os.environ, {"isDevelopmentMode": is_dev_mode}):
            service = CodeProviderService(mock_sql_db)
            repo_name = "test-repo"
            result = service.get_repo(repo_name)
            assert result == {"name": "test-repo", "id": 1}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_dev_mode,service_class", [
        ("enabled", LocalRepoService),
        ("disabled", GithubService)
    ])
    # Test get project structure in async mode
    async def test_get_project_structure_async(self, mock_sql_db, is_dev_mode, service_class):
        with patch.dict(os.environ, {"isDevelopmentMode": is_dev_mode}):
            service = CodeProviderService(mock_sql_db)
            project_id = "123"
            path = "test/path"
            result = await service.get_project_structure_async(project_id, path)
            assert result == {"files": [], "directories": []}

    @pytest.mark.parametrize("is_dev_mode,service_class", [
        ("enabled", LocalRepoService),
        ("disabled", GithubService)
    ])
    def test_get_file_content(self, mock_sql_db, is_dev_mode, service_class):
        with patch.dict(os.environ, {"isDevelopmentMode": is_dev_mode}):
            service = CodeProviderService(mock_sql_db)
            result = service.get_file_content(
                repo_name="test-repo",
                file_path="test/file.py",
                start_line=1,
                end_line=10,
                branch_name="main",
                project_id="123"
            )
            assert result == "test content"

    # Cover Edge Cases
    def test_get_repo_with_invalid_name(self, mock_sql_db, mock_github_service):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}):
            mock_github_service.get_repo.side_effect = Exception("Repository not found")
            service = CodeProviderService(mock_sql_db)
            service.service_instance = mock_github_service
            
            with pytest.raises(Exception) as exc_info:
                service.get_repo("invalid-repo")
            assert str(exc_info.value) == "Repository not found"

    @pytest.mark.asyncio
    async def test_get_project_structure_async_invalid_id(self, mock_sql_db, mock_github_service):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}):
            mock_github_service.get_project_structure_async.side_effect = Exception("Project not found")
            service = CodeProviderService(mock_sql_db)
            service.service_instance = mock_github_service
            
            with pytest.raises(Exception) as exc_info:
                await service.get_project_structure_async("invalid-id")
            assert str(exc_info.value) == "Project not found"

    # Cover Edge Cases
    def test_get_file_content_invalid_params(self, mock_sql_db, mock_github_service):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}):
            mock_github_service.get_file_content.side_effect = Exception("File not found")
            service = CodeProviderService(mock_sql_db)
            service.service_instance = mock_github_service
            
            with pytest.raises(Exception) as exc_info:
                service.get_file_content(
                    repo_name="test-repo",
                    file_path="invalid/path.py",
                    start_line=1,
                    end_line=10,
                    branch_name="main",
                    project_id="123"
                )
            assert str(exc_info.value) == "No file found" 