import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from app.modules.code_provider.code_provider_service import CodeProviderService


@pytest.fixture
def mock_sql_db():
    """Provides a MagicMock for the database session."""
    return MagicMock()


@patch('app.modules.code_provider.code_provider_service.GithubService')
@patch('app.modules.code_provider.code_provider_service.LocalRepoService')
class TestCodeProviderServiceInitialization:
    """Tests the service initialization and mode switching logic."""

    def test_initialization_in_development_mode(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Test LocalRepoService instantiated when 'isDevelopmentMode' is 'enabled'
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            service = CodeProviderService(mock_sql_db)
            mock_LocalRepoService.assert_called_once_with(mock_sql_db)
            mock_GithubService.assert_not_called()
            assert service.service_instance == mock_LocalRepoService.return_value

    def test_initialization_in_production_mode(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Test GithubService instantiated when 'isDevelopmentMode' is not equal to 'enabled'
        """
        with patch.dict(os.environ, {}, clear=True):
            service = CodeProviderService(mock_sql_db)
            mock_GithubService.assert_called_once_with(mock_sql_db)
            mock_LocalRepoService.assert_not_called()
            assert service.service_instance == mock_GithubService.return_value

        mock_LocalRepoService.reset_mock()
        mock_GithubService.reset_mock()
        with patch.dict(os.environ, {"isDevelopmentMode": "disabled"}):
            service = CodeProviderService(mock_sql_db)
            mock_GithubService.assert_called_once_with(mock_sql_db)
            mock_LocalRepoService.assert_not_called()


@patch('app.modules.code_provider.code_provider_service.GithubService')
@patch('app.modules.code_provider.code_provider_service.LocalRepoService')
class TestCodeProviderServiceDelegation:
    """Tests that methods correctly delegate to the underlying service instance."""

    def test_get_repo_delegates_to_local_service_in_dev_mode(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call to LocalRepoService.get_repo.
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            mock_local_instance.get_repo.return_value = "local_repo_object"
            
            service = CodeProviderService(mock_sql_db)
            result = service.get_repo("path/to/my/repo")
            mock_local_instance.get_repo.assert_called_once_with("path/to/my/repo")
            mock_GithubService.return_value.get_repo.assert_not_called()
            assert result == "local_repo_object"

    def test_get_repo_delegates_to_github_service_in_prod_mode(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call to GithubService.get_repo.
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            mock_github_instance.get_repo.return_value = "github_repo_object"
            
            service = CodeProviderService(mock_sql_db)
            result = service.get_repo("owner/repo_name")
            mock_github_instance.get_repo.assert_called_once_with("owner/repo_name")
            mock_LocalRepoService.return_value.get_repo.assert_not_called()
            assert result == "github_repo_object"
    
    @pytest.mark.asyncio
    async def test_get_project_structure_async_delegates_to_local_service(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call to LocalRepoService.get_project_structure_async.
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            mock_local_instance.get_project_structure_async = AsyncMock(return_value="local_file_tree")
            
            service = CodeProviderService(mock_sql_db)
            result = await service.get_project_structure_async("proj_id_123", path="src")
            mock_local_instance.get_project_structure_async.assert_awaited_once_with("proj_id_123", "src")
            assert result == "local_file_tree"

    @pytest.mark.asyncio
    async def test_get_project_structure_async_delegates_to_github_service(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call to GithubService.get_project_structure_async.
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            mock_github_instance.get_project_structure_async = AsyncMock(return_value="github_file_tree")
            
            service = CodeProviderService(mock_sql_db)
            result = await service.get_project_structure_async("proj_id_456", path=None)
            mock_github_instance.get_project_structure_async.assert_awaited_once_with("proj_id_456", None)
            assert result == "github_file_tree"
    
    def test_get_file_content_delegates_to_local_service(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call with all arguments to LocalRepoService.
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            mock_local_instance.get_file_content.return_value = "local file content"
            
            service = CodeProviderService(mock_sql_db)
            result = service.get_file_content(
                repo_name="my_repo",
                file_path="src/main.py",
                start_line=10,
                end_line=20,
                branch_name="feature-branch",
                project_id="proj_local",
                commit_id="abc123"
            )
            mock_local_instance.get_file_content.assert_called_once_with(
                "my_repo", "src/main.py", 10, 20, "feature-branch", "proj_local", "abc123"
            )
            assert result == "local file content"

    def test_get_file_content_delegates_to_github_service(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Should delegate the call with all arguments to GithubService.
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            mock_github_instance.get_file_content.return_value = "github file content"
            
            service = CodeProviderService(mock_sql_db)
            result = service.get_file_content(
                repo_name="owner/repo",
                file_path="src/app.js",
                start_line=1,
                end_line=5,
                branch_name="main",
                project_id="proj_gh",
                commit_id="commit123"
            )
            mock_github_instance.get_file_content.assert_called_once_with(
                "owner/repo", "src/app.js", 1, 5, "main", "proj_gh", "commit123"
            )
            assert result == "github file content"

@patch('app.modules.code_provider.code_provider_service.GithubService')
@patch('app.modules.code_provider.code_provider_service.LocalRepoService')
class TestCodeProviderServiceExceptionHandling:
    """Tests that exceptions from underlying services are propagated correctly."""

    def test_get_repo_local_not_found(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """        
        Raise HTTP Exception if Repo does not exist in development
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            err = HTTPException(status_code=404, detail="Local repository at invalid_path not found")
            mock_local_instance.get_repo.side_effect = err
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_repo("invalid_path")
            assert excinfo.value.status_code == 404
            assert "not found" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_get_project_structure_local_project_not_found(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if project not found when get_project_structure_async is called in development mode
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            err = HTTPException(status_code=404, detail="Project not found")
            mock_local_instance.get_project_structure_async = AsyncMock(side_effect=err)
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                await service.get_project_structure_async("nonexistent_id")
            assert excinfo.value.status_code == 404
            assert excinfo.value.detail == "Project not found"
            
    @pytest.mark.asyncio
    async def test_get_project_structure_local_fetch_failure(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if error in fetching project structure in development mode
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            err = HTTPException(status_code=500, detail="Failed to fetch project structure: IO error")
            mock_local_instance.get_project_structure_async = AsyncMock(side_effect=err)
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                await service.get_project_structure_async("proj_id_123")
            assert excinfo.value.status_code == 500
            assert "Failed to fetch project structure" in excinfo.value.detail

    def test_get_file_content_local_project_not_found(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if project not found when get_file_content is called in development mode
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            err = HTTPException(status_code=404, detail="Project not found")
            mock_local_instance.get_file_content.side_effect = err
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_file_content("repo", "file.py", 1, 10, "main", "nonexistent_id", "commit123")
            assert excinfo.value.status_code == 404
            assert excinfo.value.detail == "Project not found"

    def test_get_file_content_local_processing_error(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if error processing file_content in development mode
        """
        with patch.dict(os.environ, {"isDevelopmentMode": "enabled"}):
            mock_local_instance = mock_LocalRepoService.return_value
            err = HTTPException(status_code=500, detail="Error processing file content: Permission denied")
            mock_local_instance.get_file_content.side_effect = err
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_file_content("repo", "file.py", 1, 10, "main", "proj", "commit123")
            assert excinfo.value.status_code == 500
            assert "Error processing file content" in excinfo.value.detail

    @pytest.mark.parametrize("token_list", ["", " ", ",", " , "])
    @patch('os.getenv')
    def test_init_github_no_tokens_raises_value_error(self, mock_getenv, mock_LocalRepoService, mock_GithubService, mock_sql_db, token_list):
        """
        Raise ValueError if 'GH_TOKEN_LIST' is empty, whitespace, or just commas in Production
        """
        mock_GithubService.side_effect = ValueError("GitHub token list is empty or not set in environment variables")
        mock_getenv.return_value = token_list
        with patch.dict(os.environ, {}, clear=True):
             with pytest.raises(ValueError, match="GitHub token list is empty or not set in environment variables"):
                CodeProviderService(mock_sql_db)

    def test_get_repo_github_inaccessible_exception(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if Repo is inaccessible in production
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            err_detail = "Repository owner/repo not found or inaccessible on GitHub"
            mock_github_instance.get_repo.side_effect = HTTPException(status_code=404, detail=err_detail)
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_repo("owner/repo")            
            assert excinfo.value.status_code == 404
            assert excinfo.value.detail == err_detail
            mock_github_instance.get_repo.assert_called_once_with("owner/repo")

    @pytest.mark.asyncio
    async def test_get_project_structure_github_project_not_found(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if Project_id is invalid in production
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            err = HTTPException(status_code=404, detail="Project not found")
            mock_github_instance.get_project_structure_async = AsyncMock(side_effect=err)

            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                await service.get_project_structure_async("invalid_id")
            assert excinfo.value.status_code == 404
            assert excinfo.value.detail == "Project not found"

    @pytest.mark.asyncio
    async def test_get_project_structure_github_fetch_failure(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception when unable to fetch project structure is inaccessible in production
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            err = HTTPException(status_code=500, detail="Failed to fetch project structure: a remote error")
            mock_github_instance.get_project_structure_async = AsyncMock(side_effect=err)
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                await service.get_project_structure_async("proj_id_123")
            assert excinfo.value.status_code == 500
            assert "Failed to fetch project structure" in excinfo.value.detail

    def test_get_file_content_github_public_failure(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if file is inaccessible in production
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            err_detail = "Repository or file not found or inaccessible: owner/repo/file.py"
            err = HTTPException(status_code=404, detail=err_detail)
            mock_github_instance.get_file_content.side_effect = err
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_file_content("owner/repo", "file.py", 1, 10, "main", "proj", "commit123")
            assert excinfo.value.status_code == 404
            assert excinfo.value.detail == err_detail

    def test_get_file_content_github_processing_error(self, mock_LocalRepoService, mock_GithubService, mock_sql_db):
        """
        Raise HTTP Exception if error in accessing file content in production
        """
        with patch.dict(os.environ, {}, clear=True):
            mock_github_instance = mock_GithubService.return_value
            err = HTTPException(status_code=500, detail="Error processing file content: bad encoding")
            mock_github_instance.get_file_content.side_effect = err
            
            service = CodeProviderService(mock_sql_db)
            with pytest.raises(HTTPException) as excinfo:
                service.get_file_content("owner/repo", "file.py", 1, 10, "main", "proj", "commit123")
            assert excinfo.value.status_code == 500
            assert "Error processing file content" in excinfo.value.detail