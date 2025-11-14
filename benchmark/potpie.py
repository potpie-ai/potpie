import asyncio
import os
import aiohttp
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import logging
import pandas as pd
from collections.abc import Awaitable
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from datetime import datetime
from phoenix.client import Client

try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

from .download import get_unique_repo_and_commits, setup_all_worktrees
from .eval import correctness

def save_benchmark_results(
    qa_df: pd.DataFrame,
    answers: list[str],
    eval_results,
    output_file: Path,
):
    """
    Save benchmark results to a CSV file with all fields and evaluation metrics.
    
    Args:
        qa_df: Original DataFrame with repo_url, commit_id, question, expected_answer
        answers: List of generated answers
        eval_results: List of MetricData objects (or None) from DeepEval
        output_file: Path to save the results CSV
    """
    # Prepare results data
    results_data = []
    eval_results_list = list(eval_results) if eval_results else []
    
    # Ensure we have enough results (pad with None if needed)
    while len(eval_results_list) < len(qa_df):
        eval_results_list.append(None)
    
    for idx, (_, row) in enumerate(qa_df.iterrows()):
        result = eval_results_list[idx] if idx < len(eval_results_list) else None
        
        result_row = {
            "repo_url": row["repo_url"],
            "commit_id": row["commit_id"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "generated_answer": answers[idx] if idx < len(answers) else "",
            "timestamp": datetime.now().isoformat(),
        }
        
        # Extract evaluation metrics from the result (MetricData object)
        if result is not None:
            score = result.score
            success = result.success
            reason = result.reason
            metric_name = result.name
        else:
            score = None
            success = None
            reason = None
            metric_name = "correctness"
        
        result_row["eval_score"] = score
        result_row["eval_success"] = success
        result_row["eval_reason"] = reason
        result_row["eval_metric_name"] = metric_name
        
        results_data.append(result_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_file, index=False, encoding="utf-8")

def upload_results_to_phoenix(
    qa_df: pd.DataFrame,
    answers: list[str],
    eval_results,
    dataset_name: str = None,
    experiment_name: str = None,
) -> None:
    """Upload benchmark results to Phoenix with versioning."""
    try:
        client = Client()
        
        # Normalize eval results
        eval_results_list = (
            list(eval_results.values()) if isinstance(eval_results, dict)
            else list(eval_results) if hasattr(eval_results, '__iter__') and not isinstance(eval_results, (str, bytes))
            else [eval_results]
        )
        
        # Names
        base_dataset_name = dataset_name or "Benchmark Dataset"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or f"Benchmark Experiment {timestamp}"
        
        # Prepare new dataset
        dataset_df = pd.DataFrame({
            "question": qa_df["question"].tolist(),
            "expected_answer": qa_df["expected_answer"].tolist(),
            "repo_url": qa_df["repo_url"].tolist(),
            "commit_id": qa_df["commit_id"].tolist(),
            "example_id": range(len(qa_df))
        })
        
        # Find matching dataset or determine version
        dataset = None
        final_dataset_name = base_dataset_name
        
        try:
            print(f"Searching for existing dataset: '{base_dataset_name}'")
            datasets_list = list(client.datasets.list())
            print(f"Found {len(datasets_list)} total datasets in Phoenix")
            
            # Look for base name OR versioned datasets (e.g., "Benchmark Results v20251112_205958")
            matching_datasets = [
                ds for ds in datasets_list 
                if ds.get('name') == base_dataset_name or ds.get('name', '').startswith(f"{base_dataset_name} v")
            ]
            
            existing = None
            if matching_datasets:
                # Use the most recent one (sort by name, versions have timestamps)
                existing = sorted(matching_datasets, key=lambda x: x.get('name', ''), reverse=True)[0]
                existing_name = existing.get('name')
                print(f"Found {len(matching_datasets)} matching dataset(s), using most recent: '{existing_name}'")
            
            if existing:
                dataset_obj = client.datasets.get_dataset(dataset=existing)
                phoenix_df = dataset_obj.to_dataframe()
                
                # Extract questions from Phoenix format for comparison
                existing_questions = _extract_questions_from_phoenix(phoenix_df)
                new_questions = dataset_df["question"].tolist()

                # WORKAROUND: Phoenix sometimes duplicates the first row
                # Remove consecutive duplicates from existing questions
                existing_deduped = []
                for i, q in enumerate(existing_questions):
                    if i == 0 or q != existing_questions[i-1]:
                        existing_deduped.append(q)
                
                # Log deduplication if it occurred
                if len(existing_deduped) != len(existing_questions):
                    print(f"Phoenix bug workaround: Removed {len(existing_questions) - len(existing_deduped)} duplicate rows")
                
                # Normalize questions for comparison (strip whitespace)
                existing_normalized = [q.strip() for q in existing_deduped]
                new_normalized = [q.strip() for q in new_questions]
    
                print(f"Comparing: {len(existing_normalized)} existing vs {len(new_normalized)} new questions")
                
                # Simple content check: compare questions
                if existing_normalized == new_normalized:
                    print(f"Dataset unchanged, reusing '{existing_name}'")
                    dataset = dataset_obj
                else:
                    # Content changed - create versioned dataset
                    final_dataset_name = f"{base_dataset_name} v{timestamp}"
                    print(f"Dataset content changed, creating '{final_dataset_name}'")
                    # Show first difference for debugging
                    if len(existing_normalized) != len(new_normalized):
                        print(f"   Reason: Different sizes ({len(existing_normalized)} vs {len(new_normalized)})")
                    else:
                        for i, (old, new) in enumerate(zip(existing_normalized, new_normalized)):
                            if old != new:
                                print(f"   Reason: Question {i} differs")
                                print(f"   Old starts with: {old[:80]}...")
                                print(f"   New starts with: {new[:80]}...")
                                break
            else:
                print(f"No existing dataset found with name '{base_dataset_name}'")
        except Exception as e:
            print(f"Error checking existing dataset: {e}")
            import traceback
            traceback.print_exc()
        
        # Create new dataset if needed
        if dataset is None:
            logging.info(f"Creating dataset '{final_dataset_name}' ({len(dataset_df)} rows)")
            dataset = client.datasets.create_dataset(
                dataframe=dataset_df,
                name=final_dataset_name,
                input_keys=["question"],
                metadata_keys=["repo_url", "commit_id", "expected_answer", "example_id"]
            )
        
        # Task function
        def task(example):
            idx = int(example.metadata.get("example_id", 0))
            idx = max(0, min(idx, len(answers) - 1))
            return {"output": answers[idx] if answers else ""}
        
        # Evaluators
        def correctness_score(output, metadata=None):
            if not metadata: return 0.0
            idx = int(metadata.get("example_id", 0))
            result = eval_results_list[idx] if idx < len(eval_results_list) else None
            return float(result.score) if (result and hasattr(result, "score") and result.score is not None) else 0.0
        
        def success(output, metadata=None):
            if not metadata: return 0
            idx = int(metadata.get("example_id", 0))
            result = eval_results_list[idx] if idx < len(eval_results_list) else None
            return 1 if (result and hasattr(result, "success") and result.success) else 0
        
        def reason(output, metadata=None):
            if not metadata: return ""
            idx = int(metadata.get("example_id", 0))
            result = eval_results_list[idx] if idx < len(eval_results_list) else None
            return str(result.reason) if (result and hasattr(result, "reason") and result.reason) else ""
        
        # Run experiment
        experiment = client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=[correctness_score, success, reason],
            experiment_name=exp_name,
        )
        
        logging.info(f"Created experiment '{exp_name}' on '{dataset.name}'")
        
    except Exception as e:
        logging.error(f"Failed to upload to Phoenix: {e}", exc_info=True)
        raise


def _extract_questions_from_phoenix(phoenix_df: pd.DataFrame) -> list[str]:
    """Extract questions from Phoenix format for comparison."""
    questions = []
    for _, row in phoenix_df.iterrows():
        if 'input' in phoenix_df.columns:
            input_val = row['input']
            if isinstance(input_val, dict):
                questions.append(input_val.get('question', ''))
            else:
                questions.append(str(input_val))
        else:
            questions.append('')
    return questions


@dataclass
class PotPieUserInfo:
    user_id: str
    user_token: str


async def _get_parse_status(user_info: PotPieUserInfo, project_id: str):
    url = f"{BASE_URL}/api/v2/parsing-status/{project_id}"
    # Set the headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": user_info.user_token,
        "x-user-id": user_info.user_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            # Check for successful response
            if response.status == 200:
                data = await response.json()
                return str(data["status"])
            else:
                raise Exception(
                    f"Failed to get response: {await response.text()} {response.status}"
                )


async def status_checker_worker(
    user_info: PotPieUserInfo, check_queue: asyncio.Queue[str], ready_projects: set[str]
):
    """Worker that pulls from queue, checks status, updates set."""
    while True:
        project_id = await check_queue.get()

        try:
            status = await _get_parse_status(user_info, project_id)

            if status == "ready":
                ready_projects.add(project_id)
                logging.info("Project {} ready", project_id)
            else:
                await check_queue.put(project_id)
                await asyncio.sleep(5.0)
        except Exception as e:
            print(f"Error checking {project_id}: {e}")
            # Optionally re-queue on error
            await check_queue.put(project_id)
        finally:
            check_queue.task_done()


async def parse_single_repo(user_info: PotPieUserInfo, commit_id: str, repo_path: Path):
    url = f"{BASE_URL}/api/v2/parse"
    payload = {
        "commit_id": commit_id,
        "repo_path": str(repo_path.absolute()),
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": user_info.user_token,
        "x-user-id": user_info.user_id,
    }

    timeout = aiohttp.ClientTimeout(total=360)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, headers=headers, json=payload, timeout=timeout
        ) as response:
            # Check for successful response
            if response.status == 200:
                data = await response.json()
                project_id = str(data["project_id"])
                return project_id
            else:
                raise Exception(
                    f"Failed to get response: {await response.text()} {response.status}"
                )


async def create_conv_and_send_msg(
    user_info: PotPieUserInfo, project_id: str, ready_projects: set[str], msg: str
) -> str:
    while project_id not in ready_projects:
        await asyncio.sleep(0)

    url = f"{BASE_URL}/api/v2/project/{project_id}/message/"
    payload = {"content": msg, "agent_id": "codebase_qna_agent"}
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": user_info.user_token,
        "x-user-id": user_info.user_id,
    }
    timeout = aiohttp.ClientTimeout(total=360)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, headers=headers, json=payload, timeout=timeout
        ) as response:
            # Check for successful response
            if response.status == 200:
                data = await response.json()
                answer = data["message"]
                return answer
            else:
                raise Exception(
                    f"Failed to get response: {await response.text()} {response.status}"
                )


async def main():
    benchmark_file = "benchmark_sub.csv"
    base_directory = Path("repos")
    base_directory.mkdir(parents=True, exist_ok=True)

    conn_info = PotPieUserInfo(
        user_id=os.environ["defaultUsername"],
        user_token=os.environ["INTERNAL_ADMIN_SECRET"],
    )

    qa_df = pd.read_csv("benchmark_sub.csv")
    repo_to_commits = get_unique_repo_and_commits(benchmark_file)
    worktree_map = setup_all_worktrees(repo_to_commits, base_directory)

    # repo_commit_to_project_id = {
    #     (
    #         "https://github.com/cline/cline.git",
    #         "ba98b44504d81ea2a261a7a18bf894b4893579c3",
    #     ): "019a6a06-3dbf-720a-bdd3-eaf62c933d6f",
    #     (
    #         "https://github.com/google-gemini/gemini-cli.git",
    #         "f6499487132f82cc6b498d9e51c939d28ccc2c70",
    #     ): "019a6a06-2ef3-7493-8491-c66c6b62284c",
    #     (
    #         "https://github.com/huggingface/peft.git",
    #         "e82e72a110b41ae8a62b4af83fab2b9aab614193",
    #     ): "019a6a05-e7bc-77e5-ba78-9dfcaed0ef73",
    #     (
    #         "https://github.com/huggingface/trl.git",
    #         "aaed6c1600ff4f3e0ccc6b3b8183c98d26390491",
    #     ): "019a6a06-028f-7759-8e68-279a651ce453",
    #     (
    #         "https://github.com/microsoft/vscode.git",
    #         "5ed7107dcc8338b4b43e32b22ae76106d7bcc6b5",
    #     ): "019a6a05-f4f0-703f-8556-a10c68c865e3",
    #     (
    #         "https://github.com/n8n-io/n8n.git",
    #         "ffbcafa2074e410279bf551bbac083874c10d19e",
    #     ): "019a6a06-62f3-77ae-9bf6-f63197bbf1e6",
    #     (
    #         "https://github.com/puppeteer/puppeteer.git",
    #         "bec92441dd8401e70150282e1065f283a5e11a14",
    #     ): "019a6a06-202b-7e61-97e4-393454399028",
    #     (
    #         "https://github.com/tinygrad/tinygrad.git",
    #         "614783693e97f57166e9e97d447ea6a6388d1519",
    #     ): "019a6a06-4f4d-7652-828b-33154defb764",
    #     (
    #         "https://github.com/unslothai/unsloth.git",
    #         "1c0ad844f170f67c7cdf6f7a9465bafb0f9627df",
    #     ): "019a6a06-10d4-7edb-8cfe-0b1d8703ffe2",
    # }

    jobs: list[Awaitable[str]] = []
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
    # _ = await asyncio.gather(*workers)
    ready_projects = set(repo_commit_to_project_id.values())

    qa_tasks: list[Awaitable[str]] = []

    # TODO: Run each question in batch to average out LLM results for each question
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
    expected_answers = qa_df["expected_answer"].tolist()

    test_cases = [
        LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )
        for question, answer, expected_answer in zip(
            qa_df["question"], answers, expected_answers
        )
    ]

    results = evaluate(
        metrics=[correctness],
        test_cases=test_cases,
    )

    # DeepEval returns an EvaluationResult object with test_results attribute
    test_results = results.test_results

    # Extract MetricData from each TestResult
    # Each TestResult has metrics_data (list of MetricData objects)
    eval_results_for_save = []
    for test_result in test_results:
        # Get the first MetricData object (our correctness metric)
        metric_data = test_result.metrics_data[0]
        eval_results_for_save.append(metric_data)

    # Print first result for verification
    # if eval_results_for_save and eval_results_for_save[0]:
    #     print(f"\nExtracted MetricData: score={eval_results_for_save[0].score}, success={eval_results_for_save[0].success}, name={eval_results_for_save[0].name}")

    # Save results to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create benchmark_results folder if it doesn't exist
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"benchmark_results_{timestamp}.csv"
    save_benchmark_results(
        qa_df=qa_df,
        answers=answers,
        eval_results=eval_results_for_save,  # Pass extracted results
        output_file=output_file,
    )
    logging.info(f"Benchmark results saved to {output_file}")

    # Upload results to Phoenix 
    upload_results_to_phoenix(
        qa_df=qa_df,
        answers=answers,
        dataset_name=f"Benchmark Results",
        experiment_name=f"Benchmark Experiment {timestamp}",
        eval_results=eval_results_for_save,  # Pass extracted results
    )


if __name__ == "__main__":
    BASE_URL = "http://localhost:8001"
    _ = load_dotenv()
    asyncio.run(main())
