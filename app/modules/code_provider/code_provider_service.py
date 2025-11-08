import os
import logging
from typing import Optional

from app.modules.code_provider.provider_factory import CodeProviderFactory

try:
    from github.GithubException import BadCredentialsException
except ImportError:
    BadCredentialsException = None

logger = logging.getLogger(__name__)


class ProviderWrapper:
    """
    Wrapper to make ICodeProvider compatible with existing service interface.

    This wrapper uses CodeProviderFactory.create_provider_with_fallback() for all
    authentication, which handles the complete fallback chain (GitHub App → PAT → Unauthenticated).
    No additional fallback logic should be added here - let exceptions propagate to callers.
    """

    def __init__(self, sql_db=None):
        # Don't create provider here - create it per-request with proper auth
        self.sql_db = sql_db

    def get_repo(self, repo_name):
        """
        Get repository using the provider.
        Uses create_provider_with_fallback which handles all authentication methods
        including GitHub App, PAT pool, single PAT, and unauthenticated fallback.

        If a configured token is invalid (401), falls back to unauthenticated access
        for GitHub public repos as a last resort.
        """
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

        try:
            repo_info = provider.get_repository(repo_name)
        except Exception as e:
            # Check if this is a 401 error (bad credentials)
            is_401_error = (
                (BadCredentialsException and isinstance(e, BadCredentialsException))
                or "401" in str(e)
                or "Bad credentials" in str(e)
                or (hasattr(e, "status") and e.status == 401)
            )

            # Only fall back for GitHub provider on 401 errors
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if provider_type == "github" and is_401_error:
                logger.warning(
                    f"Configured authentication failed (401) for {repo_name}, "
                    f"falling back to unauthenticated access for public repo"
                )
                # Try unauthenticated as final fallback for public repos
                from app.modules.code_provider.github.github_provider import (
                    GitHubProvider,
                )

                unauth_provider = GitHubProvider()
                unauth_provider.set_unauthenticated_client()
                repo_info = unauth_provider.get_repository(repo_name)
                # Replace provider for subsequent operations on the MockRepo
                provider = unauth_provider
            else:
                # Not a 401 error, or not GitHub - propagate the error
                raise

        # Create a mock repository object that matches the expected interface
        class MockRepo:
            def __init__(self, repo_info, provider):
                self.full_name = repo_info["full_name"]
                self.owner = type("Owner", (), {"login": repo_info["owner"]})()
                self.default_branch = repo_info["default_branch"]
                self.private = repo_info["private"]
                self.description = repo_info["description"]
                self.language = repo_info["language"]
                self.html_url = repo_info["url"]
                self.size = repo_info.get("size", 0)
                self.stargazers_count = repo_info.get("stars", 0)
                self.forks_count = repo_info.get("forks", 0)
                self.watchers_count = repo_info.get("watchers", 0)
                self.open_issues_count = repo_info.get("open_issues", 0)
                self.created_at = repo_info.get("created_at")
                self.updated_at = repo_info.get("updated_at")

                # Handle None values for datetime fields
                if self.created_at is None:
                    from datetime import datetime

                    self.created_at = datetime.now()
                if self.updated_at is None:
                    from datetime import datetime

                    self.updated_at = datetime.now()
                self._provider = provider

            def get_languages(self):
                # Return a mock languages dict
                return {}

            def get_commits(self):
                # Return a mock commits object
                class MockCommits:
                    totalCount = 0

                return MockCommits()

            def get_contributors(self):
                # Return a mock contributors object
                class MockContributors:
                    totalCount = 0

                return MockContributors()

            def get_topics(self):
                # Return empty topics list
                return []

            def get_archive_link(self, format_type, ref):
                # Return archive link using provider
                import logging

                logger = logging.getLogger(__name__)

                logger.info(
                    f"ProviderWrapper: Getting archive link for repo '{self.full_name}', format: '{format_type}', ref: '{ref}'"
                )

                try:
                    # Use the provider's get_archive_link method if available
                    if hasattr(self._provider, "get_archive_link"):
                        archive_url = self._provider.get_archive_link(
                            self.full_name, format_type, ref
                        )
                        logger.info(
                            f"ProviderWrapper: Retrieved archive URL from provider: {archive_url}"
                        )
                        return archive_url
                    else:
                        # Fallback to manual URL construction
                        base_url = self._provider.get_api_base_url()

                        # Check if this is GitBucket (different URL format)
                        if (
                            hasattr(self._provider, "get_provider_name")
                            and self._provider.get_provider_name() == "gitbucket"
                        ):
                            # GitBucket uses a different URL format: http://hostname/owner/repo/archive/ref.format
                            # Remove /api/v3 from base URL if present
                            if base_url.endswith("/api/v3"):
                                base_url = base_url[:-7]  # Remove '/api/v3'

                            if format_type == "tarball":
                                archive_url = (
                                    f"{base_url}/{self.full_name}/archive/{ref}.tar.gz"
                                )
                            else:
                                archive_url = (
                                    f"{base_url}/{self.full_name}/archive/{ref}.zip"
                                )
                        else:
                            # Standard GitHub API format
                            if format_type == "tarball":
                                archive_url = (
                                    f"{base_url}/repos/{self.full_name}/tarball/{ref}"
                                )
                            else:
                                archive_url = (
                                    f"{base_url}/repos/{self.full_name}/zipball/{ref}"
                                )

                        logger.info(
                            f"ProviderWrapper: Generated archive URL (fallback): {archive_url}"
                        )
                        return archive_url
                except Exception as e:
                    logger.error(
                        f"ProviderWrapper: Error getting archive link for '{self.full_name}': {e}"
                    )
                    raise

            @property
            def provider(self):
                # Add provider property to MockRepo for compatibility
                return self._provider if hasattr(self, "_provider") else None

            def get_branch(self, branch_name):
                # Get branch info using provider
                branch_info = self._provider.get_branch(self.full_name, branch_name)

                class MockBranch:
                    def __init__(self, branch_info):
                        self.name = branch_info["name"]
                        self.commit = type(
                            "Commit", (), {"sha": branch_info["commit_sha"]}
                        )()
                        self.protected = branch_info["protected"]

                return MockBranch(branch_info)

        # Return the provider client and mock repo
        return provider.client, MockRepo(repo_info, provider)

    def get_file_content(
        self,
        repo_name,
        file_path,
        start_line,
        end_line,
        branch_name,
        project_id,
        commit_id,
    ):
        """
        Get file content using the provider with fallback authentication.

        If a configured token is invalid (401), falls back to unauthenticated access
        for GitHub public repos as a last resort.
        """
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

        try:
            return provider.get_file_content(
                repo_name=repo_name,
                file_path=file_path,
                ref=branch_name if not commit_id else commit_id,
                start_line=start_line,
                end_line=end_line,
            )
        except Exception as e:
            # Check if this is a 401 error (bad credentials)
            is_401_error = (
                (BadCredentialsException and isinstance(e, BadCredentialsException))
                or "401" in str(e)
                or "Bad credentials" in str(e)
                or (hasattr(e, "status") and e.status == 401)
            )

            # Only fall back for GitHub provider on 401 errors
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            if provider_type == "github" and is_401_error:
                logger.warning(
                    f"Configured authentication failed (401) for {repo_name}/{file_path}, "
                    f"falling back to unauthenticated access"
                )
                # Try unauthenticated as final fallback for public repos
                from app.modules.code_provider.github.github_provider import (
                    GitHubProvider,
                )

                unauth_provider = GitHubProvider()
                unauth_provider.set_unauthenticated_client()
                return unauth_provider.get_file_content(
                    repo_name=repo_name,
                    file_path=file_path,
                    ref=branch_name if not commit_id else commit_id,
                    start_line=start_line,
                    end_line=end_line,
                )
            else:
                # Not a 401 error, or not GitHub - propagate the error
                raise

    async def get_project_structure_async(self, project_id, path: Optional[str] = None):
        """Get project structure using the provider."""
        try:
            # Get the project details from the database using project_id
            from app.modules.projects.projects_service import ProjectService
            from fastapi import HTTPException

            project_manager = ProjectService(self.sql_db)

            project = await project_manager.get_project_from_db_by_id(project_id)
            if not project:
                logger.error(f"Project not found for project_id: {project_id}")
                return []

            # Extract repository path/name from project details
            # Prefer repo_path (for local repos) over project_name (for remote repos)
            repo_path = project.get("repo_path")
            repo_name = repo_path if repo_path else project.get("project_name")

            if not repo_name:
                logger.error(
                    f"Project {project_id} has no associated repository name or path"
                )
                return []

            logger.info(
                f"Retrieved repository {'path' if repo_path else 'name'} '{repo_name}' for project_id '{project_id}'"
            )

            # Auto-detect local paths (absolute paths, starting with ~, or valid directory)
            is_local_path = (
                os.path.isabs(repo_name)
                or repo_name.startswith(("~", "./", "../"))
                or os.path.isdir(os.path.expanduser(repo_name))
            )

            # Determine provider type to decide which implementation to use
            provider_type = os.getenv("CODE_PROVIDER", "github").lower()

            # For local repos detected by path, always use LocalProvider
            if is_local_path or provider_type == "local":
                provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
                # Use the provider to get repository structure
                structure = provider.get_repository_structure(
                    repo_name=repo_name, path=path or "", max_depth=4
                )
                return structure

            # For GitHub repos, use the old GithubService implementation which has better async handling,
            # caching, proper depth tracking, and returns formatted string output
            if provider_type == "github":
                from app.modules.code_provider.github.github_service import (
                    GithubService,
                )

                github_service = GithubService(self.sql_db)
                # Let HTTPException propagate (GithubService raises it for errors)
                return await github_service.get_project_structure_async(
                    project_id, path
                )

            # For other providers (local, GitBucket, etc.), use the provider-based approach
            provider = CodeProviderFactory.create_provider_with_fallback(repo_name)

            # Use the provider to get repository structure
            structure = provider.get_repository_structure(
                repo_name=repo_name, path=path or "", max_depth=4
            )

            return structure
        except HTTPException:
            # Re-raise HTTP exceptions from GithubService
            raise
        except Exception as e:
            logger.error(f"Failed to get project structure for {project_id}: {e}")
            return []


class CodeProviderService:
    def __init__(self, sql_db):
        self.sql_db = sql_db
        self.service_instance = self._get_service_instance()

    def _get_service_instance(self):
        # Always use ProviderWrapper for unified provider access
        # ProviderWrapper handles factory creation and authentication fallback
        # LocalProvider will be auto-selected for local paths
        return ProviderWrapper(self.sql_db)

    def get_repo(self, repo_name):
        return self.service_instance.get_repo(repo_name)

    async def get_project_structure_async(self, project_id, path: Optional[str] = None):
        return await self.service_instance.get_project_structure_async(project_id, path)

    def get_file_content(
        self,
        repo_name,
        file_path,
        start_line,
        end_line,
        branch_name,
        project_id,
        commit_id,
    ):
        return self.service_instance.get_file_content(
            repo_name,
            file_path,
            start_line,
            end_line,
            branch_name,
            project_id,
            commit_id,
        )
