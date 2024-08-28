from fastapi import Depends, HTTPException
from app.modules.auth.auth_service import AuthService
from app.modules.github.github_service import GithubService

class GithubController:
    
    @staticmethod
    def get_user_repos(user=Depends(AuthService.check_auth)):
        try:
            repos = GithubService.get_repos_for_user()
            repo_list = [
                {
                    "id": repo["id"],
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "private": repo["private"],
                    "url": repo["html_url"],
                    "owner": repo["owner"]["login"],
                }
                for repo in repos
            ]
            return {"repositories": repo_list}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch repositories: {str(e)}"
            )
    
    @staticmethod
    def get_branch_list(repo_name: str, user=Depends(AuthService.check_auth)):
        github_client = None
        if github_client is None:
            github_client, _, _, _ = GithubService.get_github_repo_details(repo_name)
        
        try:
            repo = github_client.get_repo(repo_name)
            branches = repo.get_branches()
            branch_list = [branch.name for branch in branches]
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Repository not found or error fetching branches: {str(e)}"
                ),
            )
        return {"branches": branch_list}
