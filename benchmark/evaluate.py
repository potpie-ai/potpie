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
from .metrics import code_correctness, correctness
from .potpie import (
    get_all_codegen_answers as get_all_codegen_answers_potpie,
    get_all_st_answers as get_all_st_answers_potpie,
)


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
        "--task",
        type=str,
        choices=["qa", "codegen"],
        default="qa",
        help="Type of evaluation to perform (default: qa)",
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of batches per problem (default: 1)",
    )

    args = parser.parse_args()

    nbatch = args.batch_size

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
        problem_sets, base_dir="/tmp/repos_batch", batch_no=nbatch, max_workers=6, task=args.task
    )
    # repo_dict = worktree_results

    # Select metric and generation function based on task type
    if args.task == "codegen":
        metric = code_correctness
        expected_column = "patch"  # SWE-bench uses "patch" for expected output
        input_column = "problem_statement"
        
        # Don't pass expected output to the tools
        problem_sets_without_expected = problem_sets.remove_columns(expected_column)
        
        result_awaitables = []
        for agent in args.agents:
            if agent == "potpie":
                result_awaitables.append(
                    get_all_codegen_answers_potpie(
                        problem_sets_without_expected, repo_map
                    )
                )
            else:
                ...
    else:  # args.task == "qa"
        metric = correctness
        expected_column = "expected_answer"
        input_column = "question"
        
        # Don't pass expected answers to the tools
        problem_sets_without_expected = problem_sets.remove_columns(expected_column)
        
        result_awaitables = []
        for agent in args.agents:
            if agent == "potpie":
                result_awaitables.append(
                    get_all_st_answers_potpie(
                        problem_sets_without_expected, repo_map, task=args.task
                    )
                )
            else:
                ...

    answers = await asyncio.gather(*result_awaitables)
    print(f"len answers : {len(answers[0])}")
    expected_outputs = problem_sets.select_columns(expected_column)
    inputs = problem_sets.select_columns(input_column)
    inputs_batched = [item[input_column] for item in inputs for _ in range(nbatch)]
    answers_flattened = [item for sublist in answers for item in sublist]

    expected_outputs_batched = [
        item[expected_column] for item in expected_outputs for _ in range(nbatch)
    ]
    test_cases = [
        LLMTestCase(
            input=input_text, actual_output=answer, expected_output=expected_output
        )
        for input_text, answer, expected_output in zip(
            inputs_batched, answers_flattened, expected_outputs_batched
        )
    ]

    results = evaluate(
        metrics=[metric],
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
