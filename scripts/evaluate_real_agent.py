# ------------------------------------------------------------
# AI Response Evaluation Script
#
# What this script does:
# - Takes a prompt and generates multiple variations
# - Simulates an AI agent producing responses
# - Converts responses into semantic embeddings
# - Compares them against expected meaning
#
# Output Explanation:
# - Final Score (0 → 1):
#     Measures overall quality of responses
#     Higher = more correct, consistent, and stable
#
# - ⚠️ Issues detected:
#     Indicates responses that deviate significantly
#     from others (potential hallucinations / wrong answers)
#
# - Responses:
#     Each generated answer from the agent
#     ⚠️ marks responses flagged as problematic
#
# Goal:
# Provide a lightweight way to evaluate AI/LLM outputs
# without relying on external APIs or heavy infrastructure.
# ------------------------------------------------------------
import asyncio
import random
import numpy as np
import warnings

# suppress warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Mock agent (replace later if needed)
# -------------------------------
def run_model(prompt):
    responses = [
        "A function is a reusable block of code that performs a specific task.",
        "Functions allow reuse of logic and help structure programs.",
        "Recursion is when a function calls itself until a base condition is met.",
        "Overfitting occurs when a model memorizes training data instead of learning patterns.",
    ]
    return random.choice(responses)


# -------------------------------
# Prompt variations
# -------------------------------
def generate_variants(prompt):
    return [
        prompt,
        f"Explain simply: {prompt}",
        f"Describe briefly: {prompt}",
        f"What does this mean: {prompt}",
    ]


# -------------------------------
# Expected answers
# -------------------------------
EXPECTED = {
    "python function": "A function is a reusable block of code that performs a task.",
    "recursion": "Recursion is when a function calls itself until a base condition is met.",
    "overfitting": "Overfitting occurs when a model memorizes training data instead of generalizing patterns.",
}


def get_expected(prompt):
    prompt = prompt.lower()
    for key in EXPECTED:
        if key in prompt:
            return EXPECTED[key]
    return prompt


# -------------------------------
# Load embedding model
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(prompt, responses):
    embeddings = embedder.encode(responses)

    # correctness vs expected
    expected = get_expected(prompt)
    expected_emb = embedder.encode([expected])[0]
    correctness = cosine_similarity([expected_emb], embeddings)[0]

    # consistency
    sim_matrix = cosine_similarity(embeddings)
    consistency = (
        np.sum(sim_matrix) - len(responses)
    ) / (len(responses) * (len(responses) - 1))

    # variance
    variance = np.var(sim_matrix)

    # outliers
    avg_sim = np.mean(sim_matrix, axis=1)
    outliers = [i for i, v in enumerate(avg_sim) if v < np.mean(avg_sim) * 0.7]

    # final score (single metric)
    final_score = (
        0.5 * np.mean(correctness) +
        0.3 * consistency -
        0.2 * variance
    )

    return final_score, outliers


# -------------------------------
# Runner
# -------------------------------
async def evaluate_prompt(prompt):
    variants = generate_variants(prompt)
    responses = [run_model(v) for v in variants]

    final_score, outliers = evaluate(prompt, responses)

    print("\n" + "=" * 40)
    print(f"Prompt: {prompt}")
    print(f"Final Score: {round(final_score,3)}")

    if outliers:
        print(f"⚠️ Issues detected in responses: {outliers}")

    print("\nResponses:")
    for i, r in enumerate(responses):
        flag = "⚠️" if i in outliers else ""
        print(f"{flag} {r}")


# -------------------------------
# MAIN
# -------------------------------
async def main():
    prompts = [
        "Explain what a Python function is",
        "What is recursion?",
        "What is overfitting in machine learning?",
    ]

    for p in prompts:
        await evaluate_prompt(p)


if __name__ == "__main__":
    asyncio.run(main())