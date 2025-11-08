import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from loguru import logger
import pandas as pd
from pandas import DataFrame


def _normalize_repo_url(url: str) -> str:
    """
    Normalize a repository URL by stripping whitespace and adding .git suffix if needed.

    Args:
        url: Repository URL to normalize

    Returns:
        Normalized URL ending with .git
    """
    url = url.strip()
    if not url.endswith(".git"):
        url += ".git"
    return url


def get_unique_repo_and_commits(csv_path: Path | str) -> dict[str, list[str]]:
    csv_file_path = Path(csv_path)
    required_columns = [
        "repo_url",
        "commit_id",
    ]
    df: DataFrame = pd.read_csv(csv_file_path, usecols=required_columns)  # pyright: ignore[reportUnknownMemberType]
    df.dropna(inplace=True)
    df["repo_url"] = df["repo_url"].apply(_normalize_repo_url)  # pyright: ignore[reportUnknownMemberType]
    repo_commit_dict = df.groupby("repo_url")["commit_id"].apply(set).to_dict()  # pyright: ignore[reportUnknownMemberType]
    return repo_commit_dict


def clone_bare_repository(repo_url: str, bare_repo_path: Path) -> tuple[str, str]:
    # Clone as bare repository
    if bare_repo_path.exists():
        logger.info(f"Repository already exists at {bare_repo_path}. Skipping cloning.")
    else:
        _ = subprocess.run(
            ["git", "clone", "--bare", repo_url, str(bare_repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    # Fetch all references
    _ = subprocess.run(
        ["git", "--git-dir", str(bare_repo_path), "fetch", "--all", "--tags"],
        check=True,
        capture_output=True,
        text=True,
    )

    return repo_url, str(bare_repo_path.absolute())


def create_worktree(
    bare_repo_path: Path, worktree_path: Path, commit_id: str
) -> tuple[Path, str]:
    # Remove existing worktree if it exists
    if worktree_path.exists():
        _ = subprocess.run(
            cwd=str(worktree_path.parent),
            args=["git", "worktree", "remove", "--force", str(worktree_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    # Create new worktree
    _ = subprocess.run(
        [
            "git",
            "--git-dir",
            str(bare_repo_path),
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            commit_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return worktree_path.absolute(), commit_id


def setup_all_worktrees(
    repo_dicts: dict[str, list[str]], base_directory: Path
) -> dict[tuple[str, str], Path]:
    successful_repos: list[str] = []
    worktree_map: dict[tuple[str, str], Path] = {}

    # Parallel bare repository cloning
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures_dict: dict[Future[tuple[str, str]], str] = {}

        for repo_url in repo_dicts.keys():
            repo_name = get_repo_name(repo_url)
            bare_repo_path = base_directory / f"{repo_name}.git"
            future = executor.submit(clone_bare_repository, repo_url, bare_repo_path)
            futures_dict[future] = repo_url

        for future in as_completed(futures_dict.keys()):
            try:
                repo_url, bare_repo_path = (
                    future.result()
                )  # This will raise if clone failed
                successful_repos.append(repo_url)
                logger.info("Successfully cloned bare repository.", repo_url=repo_url)
            except Exception as e:  # TODO: Specialize the exception later like CalledProcessError or OSError
                logger.error(
                    "Failed to clone bare repository",
                    error=e,
                    repo_url=futures_dict[future],
                )

    #  Parallel worktree creation
    with ThreadPoolExecutor(max_workers=6) as executor:
        worktree_futures: dict[Future[tuple[Path, str]], str] = {}

        for repo_url in successful_repos:
            repo_name = get_repo_name(repo_url)
            bare_repo_path = base_directory / f"{repo_name}.git"

            for commit_id in repo_dicts[repo_url]:
                worktree_path = bare_repo_path / commit_id
                worktree_path.parent.mkdir(exist_ok=True)

                future = executor.submit(
                    create_worktree,
                    bare_repo_path,
                    worktree_path,
                    commit_id,
                )
                worktree_futures[future] = repo_url

        # Collect worktree results
        for future in as_completed(worktree_futures):
            try:
                worktree_path, commit_id = future.result()
                repo_url = worktree_futures[future]
                worktree_map[(repo_url, commit_id)] = worktree_path
                logger.debug("Created worktree for commit {}", commit_id[:8])
            except Exception as e:
                logger.error("Failed to create worktree: {}", str(e))

    return worktree_map


def get_repo_name(repo_url: str) -> str:
    """
    Get the name of the repository from the URL.

    Args:
        repo_url: The URL of the repository. Expected to be in the format
            "https://<provider>.com/username/repository.git".

    Returns:
        The name of the repository.
    """
    repo_slug_split = repo_url.strip().split("/")[-2:]
    return "_".join(repo_slug_split)[:-4]


# To be used with atexit
def cleanup_downloaded_repos(base_directory: Path) -> None:
    """
    Clean up all downloaded repositories in the specified base directory.

    Args:
        base_directory: Base directory for repositories and worktrees
    """
    if base_directory.exists():
        logger.warning("Cleaning up all repositories in {}", base_directory)
        import shutil

        shutil.rmtree(base_directory)
        logger.success("Cleanup completed")
    else:
        logger.debug("No repositories to cleanup")


if __name__ == "__main__":
    logger.add(sys.stdout, format=" | {extra}")
    csv_path = Path("benchmark.csv")
    base_directory = Path("repos")
    base_directory.mkdir(parents=True, exist_ok=True)
    # csv_entries, worktree_map = setup_benchmark_repositories(csv_path, base_directory)
    repo_dict = get_unique_repo_and_commits(csv_path)
    # pprint(repo_dict)
    worktree_map = setup_all_worktrees(repo_dict, base_directory)
    print(worktree_map)
