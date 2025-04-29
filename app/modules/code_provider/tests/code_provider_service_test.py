import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService

@pytest.fixture
def mock_sql_db():
    return Mock()

@pytest.fixture
def mock_project_manager():
    manager = Mock()
    manager.get_project_from_db_by_id_sync.return_value = {
        "repo_path": "/demo/repo/path",
        "id": "123"
    }
    manager.get_project_from_db_by_id.return_value = {
        "repo_path": "/demo/repo/path",
        "id": "123"
    }
    return manager

@pytest.fixture
def mock_github_service(mock_sql_db):
    with patch.dict(os.environ, {"GH_TOKEN_LIST": "demo-token"}):
        service = Mock(spec=GithubService)
        service.get_repo.return_value = {"name": "test-repo", "id": 1}
        service.get_project_structure_async.return_value = {"files": [], "directories": []}
        service.get_file_content.return_value = "test content"
        return service

@pytest.fixture
def mock_local_repo_service(mock_sql_db, mock_project_manager):
    service = Mock(spec=LocalRepoService)
    service.get_repo.return_value = MagicMock()
    service.get_project_structure_async.return_value = {"files": [], "directories": []}
    service.get_file_content.return_value = "test content"
    service.project_manager = mock_project_manager
    return service

class TestCodeProviderService:
    
    def test_init_development_mode(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            with patch("app.modules.code_provider.code_provider_service.LocalRepoService") as mock_local:
                mock_local.return_value = Mock(spec=LocalRepoService)
                service = CodeProviderService(mock_sql_db)
                assert isinstance(service.service_instance, LocalRepoService)

    def test_init_production_mode(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled", "GH_TOKEN_LIST": "demo-token"}):
            with patch("app.modules.code_provider.code_provider_service.GithubService") as mock_github:
                mock_github.return_value = Mock(spec=GithubService)
                service = CodeProviderService(mock_sql_db)
                assert isinstance(service.service_instance, GithubService)

    @pytest.mark.parametrize("is_dev_mode,service_class,mock_service", [
        ("enabled", LocalRepoService, "app.modules.code_provider.code_provider_service.LocalRepoService"),
        ("disabled", GithubService, "app.modules.code_provider.code_provider_service.GithubService")
    ])
    def test_get_repo(self, mock_sql_db, is_dev_mode, service_class, mock_service, mock_local_repo_service, mock_github_service):
        env_vars = {"isDevelopmentMode": is_dev_mode}
        if is_dev_mode == "disabled":
            env_vars["GH_TOKEN_LIST"] = "demo-token"
        
        with patch.dict(os.environ, env_vars):
            with patch(mock_service) as mock_service_class:
                mock_instance = mock_local_repo_service if is_dev_mode == "enabled" else mock_github_service
                mock_service_class.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                result = service.get_repo("test-repo")
                
                if is_dev_mode == "enabled":
                    assert mock_instance.get_repo.called
                else:
                    assert mock_instance.get_repo.called

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_dev_mode,service_class,mock_service", [
        ("enabled", LocalRepoService, "app.modules.code_provider.code_provider_service.LocalRepoService"),
        ("disabled", GithubService, "app.modules.code_provider.code_provider_service.GithubService")
    ])
    async def test_get_project_structure_async(self, mock_sql_db, is_dev_mode, service_class, mock_service, 
                                             mock_local_repo_service, mock_github_service):
        env_vars = {"isDevelopmentMode": is_dev_mode}
        if is_dev_mode == "disabled":
            env_vars["GH_TOKEN_LIST"] = "demo-token"
            
        with patch.dict(os.environ, env_vars):
            with patch(mock_service) as mock_service_class:
                mock_instance = mock_local_repo_service if is_dev_mode == "enabled" else mock_github_service
                mock_service_class.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                result = await service.get_project_structure_async("123", "test/path")
                
                if is_dev_mode == "enabled":
                    assert mock_instance.get_project_structure_async.called
                else:
                    assert mock_instance.get_project_structure_async.called

    @pytest.mark.parametrize("is_dev_mode,service_class,mock_service", [
        ("enabled", LocalRepoService, "app.modules.code_provider.code_provider_service.LocalRepoService"),
        ("disabled", GithubService, "app.modules.code_provider.code_provider_service.GithubService")
    ])
    def test_get_file_content(self, mock_sql_db, is_dev_mode, service_class, mock_service,
                            mock_local_repo_service, mock_github_service):
        env_vars = {"isDevelopmentMode": is_dev_mode}
        if is_dev_mode == "disabled":
            env_vars["GH_TOKEN_LIST"] = "demo-token"
            
        with patch.dict(os.environ, env_vars):
            with patch(mock_service) as mock_service_class:
                mock_instance = mock_local_repo_service if is_dev_mode == "enabled" else mock_github_service
                mock_service_class.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                result = service.get_file_content(
                    repo_name="test-repo",
                    file_path="test/file.py",
                    start_line=1,
                    end_line=10,
                    branch_name="main",
                    project_id="123"
                )
                
                if is_dev_mode == "enabled":
                    assert mock_instance.get_file_content.called
                else:
                    assert mock_instance.get_file_content.called

    def test_get_repo_with_invalid_name(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled", "GH_TOKEN_LIST": "demo-token"}):
            with patch("app.modules.code_provider.code_provider_service.GithubService") as mock_github:
                mock_instance = Mock(spec=GithubService)
                mock_instance.get_repo.side_effect = Exception("Repository not found")
                mock_github.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                with pytest.raises(Exception) as exc_info:
                    service.get_repo("invalid-repo")
                assert str(exc_info.value) == "Repository not found"

    @pytest.mark.asyncio
    async def test_get_project_structure_async_invalid_id(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled", "GH_TOKEN_LIST": "demo-token"}):
            with patch("app.modules.code_provider.code_provider_service.GithubService") as mock_github:
                mock_instance = Mock(spec=GithubService)
                mock_instance.get_project_structure_async.side_effect = Exception("Project not found")
                mock_github.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                with pytest.raises(Exception) as exc_info:
                    await service.get_project_structure_async("invalid-id")
                assert str(exc_info.value) == "Project not found"

    def test_get_file_content_invalid_params(self, mock_sql_db):
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled", "GH_TOKEN_LIST": "demo-token"}):
            with patch("app.modules.code_provider.code_provider_service.GithubService") as mock_github:
                mock_instance = Mock(spec=GithubService)
                mock_instance.get_file_content.side_effect = Exception("File not found")
                mock_github.return_value = mock_instance
                
                service = CodeProviderService(mock_sql_db)
                with pytest.raises(Exception) as exc_info:
                    service.get_file_content(
                        repo_name="test-repo",
                        file_path="invalid/path.py",
                        start_line=1,
                        end_line=10,
                        branch_name="main",
                        project_id="123"
                    )
                assert str(exc_info.value) == "File not found" 