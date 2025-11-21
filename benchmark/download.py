import concurrent.futures
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Tuple

from loguru import logger


def run_cmd(
    cmd: list,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = False,
    env: dict | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run to simplify calls."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        check=check,
        env=env,
        timeout=timeout,
    )


def clone_bare_repo(repo_url: str, bare_parent: Path) -> Path:
    """
    Ensure bare clone exists at bare_parent/<repo_name>.git
    Returns path to bare repo.
    """
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    bare_path = (bare_parent / f"{name}.git").resolve()
    bare_path.parent.mkdir(parents=True, exist_ok=True)

    if bare_path.exists():
        logger.debug("Bare repo already exists at {}.", bare_path)
        return bare_path

    logger.bind(path=bare_path).info("Cloning bare repository.")
    cmd = ["git", "clone", "--bare", repo_url, str(bare_path)]
    cp = run_cmd(cmd, capture_output=True)
    if cp.returncode != 0:
        logger.error("Failed to clone repo: {}", cp.stderr.strip())
        raise RuntimeError(f"Failed to clone {repo_url}: {cp.stderr.strip()}")
    logger.bind(path=bare_path).info("Successfully cloned bare repo.")
    return bare_path


def ensure_controller_for_bare(bare_repo_path: Path) -> Path:
    """
    For a bare repo at <X>.git, ensure controller at <X>.git.workcontroller exists:
      git clone --local --no-checkout <bare> <controller>
    Return path to controller repo.
    """
    controller = bare_repo_path.parent / (bare_repo_path.name + ".workcontroller")
    if controller.exists():
        logger.debug("Controller already exists at {}", controller)
        return controller.resolve()

    logger.debug("Creating controller repo at {}", controller)
    cmd = [
        "git",
        "clone",
        "--local",
        "--no-checkout",
        str(bare_repo_path),
        str(controller),
    ]
    cp = run_cmd(cmd, capture_output=True)
    if cp.returncode != 0:
        logger.error(
            "Failed to create controller for {}: {}", bare_repo_path, cp.stderr.strip()
        )
        raise RuntimeError(
            f"Failed to create controller for {bare_repo_path}: {cp.stderr.strip()}"
        )
    logger.debug(f"Successfully created controller at {controller}")
    return controller.resolve()


def controller_has_commit(controller_path: Path, commit_id: str) -> bool:
    """Return True if controller repo knows about commit_id."""
    try:
        cp = run_cmd(
            ["git", "rev-parse", "--verify", f"{commit_id}^{{commit}}"],
            cwd=controller_path,
            capture_output=True,
        )
        return cp.returncode == 0
    except Exception:
        return False


def fetch_all_into_controller(controller_path: Path) -> None:
    """Run git fetch --all --tags in the controller to ensure refs/objects are present."""
    logger.debug(f"Fetching all refs into controller at {controller_path}")
    cp = run_cmd(
        ["git", "fetch", "--all", "--tags"], cwd=controller_path, capture_output=True
    )
    if cp.returncode != 0:
        logger.error("git fetch failed for {}: {}", controller_path, cp.stderr.strip())
        raise RuntimeError(
            f"git fetch failed for {controller_path}: {cp.stderr.strip()}"
        )
    logger.debug("Successfully fetched refs for {}.", controller_path)


def add_worktree_for_commit(
    controller_path: Path, worktree_path: Path, commit_id: str
) -> None:
    """
    Run: git worktree add <worktree_path> <commit_id>
    If already exists, silently return.
    """
    if worktree_path.exists():
        # already created (idempotent behavior)
        logger.debug("Worktree already exists at {}", worktree_path)
        return

    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    logger.bind(worktree_path=worktree_path, commit_id=commit_id).debug(
        "Adding worktree"
    )
    cmd = ["git", "worktree", "add", str(worktree_path), commit_id]
    cp = run_cmd(cmd, cwd=controller_path, capture_output=True)
    if cp.returncode != 0:
        # Stderr for debugging
        logger.error(
            "Failed to add worktree at {}: {}", worktree_path, cp.stderr.strip()
        )
        raise RuntimeError(
            f"Failed to add worktree {worktree_path} at {commit_id}: {cp.stderr.strip()}"
        )
    logger.debug(f"Successfully added worktree at {worktree_path}")


def group_rows_by_repo(
    rows: Iterable[dict[str, str]],
    task: str = "qa",
) -> dict[str, list[Tuple[str, str]]]:
    """
    Group rows by repo URL.

    Args:
        rows: Iterable of row dictionaries
        task: "qa" or "codegen"

    Returns mapping: repo_url -> list of (commit_id, problem_id)
    """
    mapping: dict[str, list[Tuple[str, str]]] = {}
    for r in rows:
        repo_url_value = r.get("repo_url", "")
        is_url_format = repo_url_value.startswith(("http://", "https://"))
        
        if is_url_format:
            repo_url = r["repo_url"]
            commit = r["commit_id"]
            prob = r["problem_id"]
        else:
            repo_url = f"https://github.com/{r['repo']}"
            commit = r["base_commit"]
            prob = r["instance_id"]

        mapping.setdefault(repo_url, []).append((commit, prob))
    return mapping


def process_single_repo(
    repo_url: str,
    entries: list[Tuple[str, str]],
    base_dir: Path,
    batch_no: int,
    skip_if_exists: bool = True,
) -> tuple[dict[tuple[str, str], dict[tuple[str, int], Path]], dict[str, Any]]:
    """
    For a single repo_url with entries [(commit_id, problem_id), ...]:
      - clone bare repo (if missing) at base_dir/bare/<repo_name>.git
      - create controller repo
      - for each entry and for batch_index in 0..batch_no-1, create worktree
    Returns summary dict with counts and failures.
    """
    summary = {"repo_url": repo_url, "created": [], "skipped": [], "failed": []}
    repo_mapping: dict[tuple[str, str], dict[tuple[str, int], Path]] = defaultdict(dict)

    try:
        bare_parent = base_dir / "bare"
        with logger.contextualize(repo_url=repo_url):
            bare_repo = clone_bare_repo(repo_url, bare_parent)
            controller = ensure_controller_for_bare(bare_repo)

        # process each commit/problem pair
        for idx, (commit_id, problem_id) in enumerate(entries, 1):
            # sanitize parts for filesystem
            for batch_index in range(batch_no):
                wt_name = f"{commit_id}_{problem_id}_{batch_index}"
                # choose parent for worktrees: base_dir/worktrees/<repo_name>/<wt_name>
                repo_name = bare_repo.name
                worktree_parent = base_dir / "worktrees" / repo_name
                worktree_path = (worktree_parent / wt_name).resolve()

                if skip_if_exists and worktree_path.exists():
                    repo_mapping[(repo_url, commit_id)].update(
                        {(problem_id, batch_index): worktree_path}
                    )
                    summary["skipped"].append(str(worktree_path))
                    continue

                # ensure the controller knows the commit; otherwise fetch
                if not controller_has_commit(controller, commit_id):
                    logger.warning(
                        f"Commit {commit_id} not found in controller, fetching..."
                    )
                    try:
                        with logger.contextualize(repo_url=repo_url):
                            fetch_all_into_controller(controller)
                    except Exception as e:
                        logger.bind(repo_url=repo_url, commit_id=commit_id).error(
                            "Fetch failed for commit"
                        )
                        summary["failed"].append(
                            {
                                "worktree": str(worktree_path),
                                "commit": commit_id,
                                "error": f"fetch_failed: {e}",
                            }
                        )
                        continue  # go to next worktree

                    # check again
                    if not controller_has_commit(controller, commit_id):
                        logger.error(f"Commit {commit_id} still not found after fetch")
                        summary["failed"].append(
                            {
                                "worktree": str(worktree_path),
                                "commit": commit_id,
                                "error": "commit_not_found_after_fetch",
                            }
                        )
                        continue

                # add the worktree
                try:
                    with logger.contextualize(repo_url=repo_url, commit_id=commit_id):
                        add_worktree_for_commit(controller, worktree_path, commit_id)
                    repo_mapping[(repo_url, commit_id)].update(
                        {(problem_id, batch_index): worktree_path}
                    )
                    summary["created"].append(str(worktree_path))
                    logger.bind(
                        repo=repo_url, step="{}/{}".format(batch_index, batch_no)
                    ).debug("Created worktree: {}", worktree_path.name)
                except Exception as e:
                    logger.error(f"Failed to create worktree {worktree_path.name}: {e}")
                    summary["failed"].append(
                        {
                            "worktree": str(worktree_path),
                            "commit": commit_id,
                            "error": str(e),
                        }
                    )
    except Exception as e:
        logger.bind(repo_url=repo_url).error("Repo setup failed")
        summary["failed"].append({"repo_setup": repo_url, "error": str(e)})

    # commit_id: {(problem_id, commit_id): worktree}

    logger.bind(
        repo_url=repo_url,
        created=len(summary["created"]),
        skipped=len(summary["skipped"]),
        failed=len(summary["failed"]),
    ).info("Completed repo")
    return repo_mapping, summary


def prepare_worktrees(
    rows: Iterable[dict[str, Any]],
    base_dir: str,
    batch_no: int,
    max_workers: int = 8,
    skip_if_exists: bool = True,
    task: str = "qa",
) -> tuple[dict[tuple[str, str], dict[tuple[str, int], Path]], list[dict[str, Any]]]:
    """
    rows: iterable of dicts with keys 'repo_url','commit_id','problem_id' (URL format) or 'repo','base_commit','instance_id' (slug format)
    base_dir: root directory where we will create:
        - base_dir/bare/<repo_name>.git
        - base_dir/worktrees/<repo_name>/<worktree_name>
    batch_no: number of batch copies to create per (commit,problem)
    max_workers: number of threads for parallel repo processing
    task: Deprecated, kept for backward compatibility. Auto-detects format.
    Returns a list of per-repo summaries.
    """
    logger.bind(base_dir=base_dir, batch_no=batch_no, max_worker=max_workers).info(
        "Starting dataset processing"
    )
    base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    grouped = group_rows_by_repo(rows, task=task)
    summaries = []
    repo_map: dict[tuple[str, str], dict[tuple[str, int], Path]] = {}

    # Process different repos in parallel, but each repo is processed serially inside process_single_repo
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_single_repo, repo, grouped[repo], base, batch_no, skip_if_exists
            ): repo
            for repo in grouped
        }
        for fut in concurrent.futures.as_completed(futures):
            repo = futures[fut]
            try:
                _repo_map, summary = fut.result()
                repo_map.update(_repo_map)
            except Exception as e:
                logger.bind(repo=repo).error("Unhandled exception processing repo.")
                summary = {
                    "repo_url": repo,
                    "created": [],
                    "skipped": [],
                    "failed": [{"error": f"unhandled_exception: {e}"}],
                }
            summaries.append(summary)

    total_created = sum(len(s["created"]) for s in summaries)
    total_skipped = sum(len(s["skipped"]) for s in summaries)
    total_failed = sum(len(s["failed"]) for s in summaries)
    logger.bind(
        total_created=total_created,
        total_skipped=total_skipped,
        total_failed=total_failed,
    ).info("Dataset processing complete.")
    return repo_map, summaries
