"""
Benchmark orchestrator - runs evaluations for multiple tools concurrently.

Usage:
    python -m benchmark.run_benchmark --tools potpie codex claude_code
    python -m benchmark.run_benchmark --tools potpie --csv custom.csv
"""

import asyncio
import argparse
import logging
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional
import pandas as pd
from pathlib import Path
from datetime import datetime
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

from . import potpie
from .potpie import (
    save_benchmark_results,
    upload_results_to_phoenix,
    PotPieUserInfo,
    create_conv_and_send_msg,
    parse_single_repo,
    status_checker_worker,
)
from .download import get_unique_repo_and_commits, setup_all_worktrees
from .eval import correctness
from .cli import run_cli_tool_async, get_available_tools, verify_cli_tool_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set BASE_URL for potpie module functions
potpie.BASE_URL = "http://localhost:8001"


def validate_csv_file(csv_file: str) -> None:
    """Validate that CSV file exists and has required columns."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}\n"
            f"Please create the CSV file with columns: repo_url, commit_id, question, expected_answer"
        )
    
    # Check required columns
    try:
        df = pd.read_csv(csv_file, nrows=0)  # Just read headers
        required_columns = ["repo_url", "commit_id", "question", "expected_answer"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV file missing required columns: {missing_columns}\n"
                f"Required columns: {required_columns}"
            )
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading CSV file: {e}")


async def run_potpie_benchmark(
    csv_file: str,
    base_directory: Path,
) -> tuple[pd.DataFrame, list[str], list]:
    """
    Run PotPie benchmark evaluation.
    
    Returns:
        Tuple of (qa_df, answers, eval_results)
    """
    logger.info("Starting PotPie benchmark...")
    
    conn_info = PotPieUserInfo(
        user_id=os.environ["defaultUsername"],
        user_token=os.environ["INTERNAL_ADMIN_SECRET"],
    )

    qa_df = pd.read_csv(csv_file)
    repo_to_commits = get_unique_repo_and_commits(csv_file)
    worktree_map = setup_all_worktrees(repo_to_commits, base_directory)

    # Parse repos
    jobs = []
    for (_, commit_id), worktree_path in worktree_map.items():
        jobs.append(
            parse_single_repo(
                user_info=conn_info,
                commit_id=commit_id,
                repo_path=worktree_path,
            )
        )
    keys = list(worktree_map.keys())
    project_ids = await asyncio.gather(*jobs)
    repo_commit_to_project_id = dict(zip(keys, project_ids))

    ready_projects: set[str] = set()
    check_queue: asyncio.Queue[str] = asyncio.Queue()

    for pid in project_ids:
        await check_queue.put(pid)

    workers = [
        asyncio.create_task(
            status_checker_worker(conn_info, check_queue, ready_projects)
        )
        for _ in range(5)
    ]
    ready_projects = set(repo_commit_to_project_id.values())

    # Process questions
    qa_tasks = []
    for _, row in qa_df.iterrows():
        repo_url: str = row["repo_url"]
        commit_id: str = row["commit_id"]
        question: str = row["question"]
        project_id = repo_commit_to_project_id[(repo_url, commit_id)]
        qa_tasks.append(
            create_conv_and_send_msg(
                user_info=conn_info,
                project_id=project_id,
                ready_projects=ready_projects,
                msg=question,
            )
        )
    answers = await asyncio.gather(*qa_tasks)
    
    # Evaluate
    expected_answers = qa_df["expected_answer"].tolist()
    test_cases = [
        LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )
        for question, answer, expected_answer in zip(
            qa_df["question"], answers, expected_answers
        )
    ]

    # Suppress verbose output from deepeval (test case details, LLM outputs)
    # Only Metrics Summary will be shown
    with redirect_stdout(StringIO()):
        results = evaluate(
            metrics=[correctness],
            test_cases=test_cases,
        )

    test_results = results.test_results
    eval_results_for_save = []
    for test_result in test_results:
        metric_data = test_result.metrics_data[0]
        eval_results_for_save.append(metric_data)
    
    logger.info("PotPie benchmark completed")
    return qa_df, answers, eval_results_for_save


async def run_cli_tool_benchmark(
    tool_name: str,
    csv_file: str,
    base_directory: Path,
) -> tuple[pd.DataFrame, list[str], list]:
    """
    Run benchmark evaluation for a CLI tool.
    
    Returns:
        Tuple of (qa_df, answers, eval_results)
    """
    logger.info(f"Starting {tool_name} benchmark...")
    
    # Verify CLI tool is available before proceeding
    is_available, error_message = verify_cli_tool_available(tool_name)
    if not is_available:
        logger.error(f"{error_message}")
        raise RuntimeError(
            f"CLI tool '{tool_name}' is not properly configured. {error_message}"
        )
    
    logger.info(f"Verified {tool_name} CLI tool is available")
    
    qa_df = pd.read_csv(csv_file)
    repo_to_commits = get_unique_repo_and_commits(csv_file)
    worktree_map = setup_all_worktrees(repo_to_commits, base_directory)
    
    # Process questions through CLI tool
    answers = []
    for idx, (_, row) in enumerate(qa_df.iterrows()):
        repo_url: str = row["repo_url"]
        commit_id: str = row["commit_id"]
        question: str = row["question"]
        
        worktree_path = worktree_map.get((repo_url, commit_id))
        if not worktree_path:
            logger.error(f"No worktree found for {repo_url}@{commit_id[:8]}")
            answers.append(f"Error: No worktree found for {repo_url}@{commit_id[:8]}")
            continue
        
        logger.info(f"Processing question {idx + 1}/{len(qa_df)} for {tool_name}")
        
        try:
            answer = await run_cli_tool_async(
                tool_name=tool_name,
                prompt=question,
                workspace_path=str(worktree_path),
            )
            
            if answer.startswith("Error:"):
                logger.warning(f"CLI returned error for question {idx + 1}: {answer}")
                answers.append(answer)
            else:
                answers.append(answer)
                logger.info(f"Got response for question {idx + 1} ({len(answer)} chars)")
        except Exception as e:
            logger.error(f"Exception processing question {idx + 1}: {e}")
            answers.append(f"Error: {str(e)}")
    
    # Check if any answers contain errors - raise exception if so
    has_errors = any(answer.startswith("Error:") for answer in answers)
    if has_errors:
        error_count = sum(1 for answer in answers if answer.startswith("Error:"))
        error_examples = [answer for answer in answers if answer.startswith("Error:")][:3]
        error_details = "\n".join(f"  - {err}" for err in error_examples)
        raise RuntimeError(
            f"CLI tool '{tool_name}' failed for {error_count}/{len(answers)} questions. "
            f"Please fix the CLI tool installation/configuration before running evaluation.\n"
            f"Error examples:\n{error_details}"
        )
    
    # Evaluate
    expected_answers = qa_df["expected_answer"].tolist()
    test_cases = [
        LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )
        for question, answer, expected_answer in zip(
            qa_df["question"], answers, expected_answers
        )
    ]

    # Suppress verbose output from deepeval (test case details, LLM outputs)
    # Only Metrics Summary will be shown
    with redirect_stdout(StringIO()):
        results = evaluate(
            metrics=[correctness],
            test_cases=test_cases,
        )

    test_results = results.test_results
    eval_results_for_save = []
    for test_result in test_results:
        metric_data = test_result.metrics_data[0]
        eval_results_for_save.append(metric_data)
    
    logger.info(f"{tool_name} benchmark completed")
    return qa_df, answers, eval_results_for_save


async def save_and_upload_results(
    tool_name: str,
    qa_df: pd.DataFrame,
    answers: list[str],
    eval_results: list,
    upload_to_phoenix: bool = True,
) -> Path:
    """Save results to CSV and optionally upload to Phoenix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / f"{tool_name}_benchmark_results_{timestamp}.csv"
    save_benchmark_results(
        qa_df=qa_df,
        answers=answers,
        eval_results=eval_results,
        output_file=output_file,
    )
    logger.info(f"Results saved to {output_file}")

    if upload_to_phoenix:
        upload_results_to_phoenix(
            qa_df=qa_df,
            answers=answers,
            dataset_name=f"Benchmark Results",
            experiment_name=f"{tool_name} Benchmark {timestamp}",
            eval_results=eval_results,
        )
        logger.info(f"Results uploaded to Phoenix")
    
    return output_file


async def run_single_tool_benchmark(
    tool_name: str,
    csv_file: str,
    base_directory: Path,
    upload_to_phoenix: bool = True,
) -> Optional[Path]:
    """
    Run benchmark for a single tool and save results.
    
    Returns:
        Path to output file if successful, None if there was an error
    """
    try:
        if tool_name == "potpie":
            qa_df, answers, eval_results = await run_potpie_benchmark(
                csv_file, base_directory
            )
        elif tool_name in get_available_tools():
            qa_df, answers, eval_results = await run_cli_tool_benchmark(
                tool_name, csv_file, base_directory
            )
        else:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Available tools: ['potpie'] + {get_available_tools()}"
            )
        
        output_file = await save_and_upload_results(
            tool_name, qa_df, answers, eval_results, upload_to_phoenix
        )
        return output_file
    except RuntimeError as e:
        # CLI tool configuration or execution errors - don't save files or upload
        logger.error(f"{tool_name} benchmark failed: {e}")
        logger.error(f"No files generated or uploaded to Phoenix due to errors")
        return None
    except Exception as e:
        # Other unexpected errors
        logger.error(f"Unexpected error in {tool_name} benchmark: {e}", exc_info=True)
        logger.error(f"No files generated or uploaded to Phoenix due to errors")
        return None


async def main():
    """Main orchestrator function."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluations for multiple tools concurrently"
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        required=True,
        help="Tool names to run benchmarks for (e.g., potpie codex claude_code)",
    )
    parser.add_argument(
        "--csv",
        default="benchmark_sub.csv",
        help="Path to CSV file with questions (default: benchmark_sub.csv)",
    )
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Don't upload results to Phoenix",
    )
    
    args = parser.parse_args()
    
    # Validate CSV file
    try:
        validate_csv_file(args.csv)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}")
        return
    
    # Validate tool names
    available_tools = ["potpie"] + get_available_tools()
    invalid_tools = [t for t in args.tools if t not in available_tools]
    if invalid_tools:
        logger.error(
            f"Unknown tools: {invalid_tools}\n"
            f"Available tools: {available_tools}"
        )
        return
    
    logger.info(f"Running benchmarks for: {', '.join(args.tools)}")
    logger.info(f"Using CSV file: {args.csv}")
    
    base_directory = Path("repos")
    base_directory.mkdir(parents=True, exist_ok=True)
    
    # Run all benchmarks concurrently
    tasks = [
        run_single_tool_benchmark(
            tool_name=tool_name,
            csv_file=args.csv,
            base_directory=base_directory,
            upload_to_phoenix=not args.no_phoenix,
        )
        for tool_name in args.tools
    ]
    
    output_files = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None values (errors) and exceptions
    successful_files = []
    failed_tools = []
    
    for tool_name, result in zip(args.tools, output_files):
        if isinstance(result, Exception):
            logger.error(f"{tool_name} benchmark failed with exception: {result}")
            failed_tools.append(tool_name)
        elif result is None:
            failed_tools.append(tool_name)
        else:
            successful_files.append(result)
    
    if successful_files:
        logger.info("Completed benchmarks:")
        logger.info(f"Results saved to:")
        for output_file in successful_files:
            logger.info(f"   - {output_file}")
    
    if failed_tools:
        logger.error(f"Failed benchmarks: {', '.join(failed_tools)}")
        logger.error("No files were generated or uploaded to Phoenix for failed benchmarks")
        sys.exit(1)


if __name__ == "__main__":
    _ = load_dotenv()
    asyncio.run(main())

