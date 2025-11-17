import argparse
import asyncio
import sys
from pathlib import Path

from datasets import load_dataset
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from loguru import logger

from .download import prepare_worktrees
from .metrics import correctness
from .potpie import get_all_st_answers as get_all_st_answers_potpie


def get_available_agents() -> list[str]:
    return ["potpie", "codex-cli"]


async def main():
    """Main evaluation function that accepts command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate tools on benchmark questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.evaluate --tools potpie
  python -m benchmark.evaluate --tools potpie --input custom_questions.csv
  python -m benchmark.evaluate --tools potpie --output results.csv
        """,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        choices=get_available_agents(),
        help="List of agents to evaluate",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="benchmark.csv",
        help="Input CSV file with benchmark questions (default: benchmark.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file for results (default: evaluation_results.csv)",
    )

    args = parser.parse_args()

    nbatch = 2

    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    assert all(agent in get_available_agents() for agent in args.agents), (
        "Invalid Agent(s): {}".format(
            ", ".join(set(args.agents) - set(get_available_agents()))
        )
    )

    logger.info("Evaluating tools: {}", ", ".join(args.agents))
    logger.info("Input file: {}", args.input)
    logger.info("Output file: {}", args.output)

    # TODO: make it work for huggingface repos
    problem_sets = load_dataset("csv", data_files=args.input, split="train")
    logger.bind(n_problems=len(problem_sets)).info(
        "Loaded problem sets from benchmark.csv"
    )

    repo_map, summary = prepare_worktrees(
        problem_sets, base_dir="/tmp/repos_batch", batch_no=nbatch, max_workers=6
    )
    # repo_dict = worktree_results

    # Don't pass expected answers to the tools
    problem_sets_without_expected_answers = problem_sets.remove_columns(
        "expected_answer"
    )

    result_awaitables = []
    for agent in args.agents:
        if agent == "potpie":
            result_awaitables.append(
                get_all_st_answers_potpie(
                    problem_sets_without_expected_answers, repo_map
                )
            )
        else:
            ...

    answers = await asyncio.gather(*result_awaitables)
    print(f"len answers : {len(answers[0])}")
    expected_answers = problem_sets.select_columns("expected_answer")
    questions = problem_sets.select_columns("question")
    questions_batched = [item["question"] for item in questions for _ in range(nbatch)]
    answers_flattened = [item for sublist in answers for item in sublist]

    expected_answers_batched = [
        item["expected_answer"] for item in expected_answers for _ in range(nbatch)
    ]
    test_cases = [
        LLMTestCase(
            input=question, actual_output=answer, expected_output=expected_answer
        )
        for question, answer, expected_answer in zip(
            questions_batched, answers_flattened, expected_answers_batched
        )
    ]

    results = evaluate(
        metrics=[correctness],
        test_cases=test_cases,
    )
    metrics = [test_result.success for test_result in results.test_results]
    print(metrics)

    # print(answers)


if __name__ == "__main__":
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level> - {extra}"
    )
    logger.remove()
    logger.add(sys.stderr, format=logger_format, level="INFO")
    _ = load_dotenv()
    asyncio.run(main())
