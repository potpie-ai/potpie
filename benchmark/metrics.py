from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

EVAL_CRITERIA = """
You are an expert evaluator. Your task is to assess how similar ANSWER A and ANSWER B are in meaning, technical content, reasoning, and conclusions.

You must output ONLY a JSON object with:
{
  "score": <number_between_0_and_1>,
  "reason": "<brief explanation>"
}

Where:
- 0.0 = The answers are completely different in content, reasoning, or conclusions.
- 1.0 = The answers are effectively the same in meaning and technical content, even if wording differs.

Evaluate similarity using the following criteria:

1. Semantic Similarity (40%)
   - Do both answers express the same ideas and conclusions?
   - Are they diagnosing the same root causes or reasoning paths?

2. Technical Detail Overlap (30%)
   - Do they mention the same technical components, parameters, or mechanisms?
   - Do they propose similar technical fixes or explanations?

3. Structural & Reasoning Similarity (20%)
   - Do they follow similar logic or steps?
   - Is the argument or workflow comparable?

4. Specificity Alignment (10%)
   - Are the level of detail, granularity, and examples comparable?

Scoring Rules:
- Base the score ONLY on similarity, NOT quality.
- Wording differences do NOT reduce similarity if the meaning is the same.
- Minor differences → 0.7–0.9
- Substantial but related differences → 0.3–0.6
- Completely different → 0.0–0.2

Your final output must contain:
- A numeric "score" between 0 and 1.
- A short "reason" describing the main factors affecting similarity.

Do NOT include any text outside the JSON object.
"""

CODE_EVAL_CRITERIA = """
You are a senior software engineer evaluating a code fix or implementation.
Your task is to assess the quality and correctness of the GENERATED CODE compared to the EXPECTED SOLUTION (or based on the PROBLEM DESCRIPTION).

Evaluate using the following criteria:
1. Functional Correctness (50%): Does the code solve the stated problem?
2. Code Quality & Best Practices (30%): Is the code idiomatic, safe, and maintainable?
3. Logic Similarity (20%): Does it follow a similar (or better) valid approach to the reference solution?

Output ONLY a JSON object with:
{
  "score": <number_between_0_and_1>,
  "reason": "<brief explanation>"
}

Where:
- 0.0 = The generated code fails to address the problem or is fundamentally incorrect.
- 1.0 = The generated code completely solves the problem with high quality and matches the reference approach or improves upon it.

Scoring Rules:
- Functional Correctness is paramount. If the code doesn't solve the problem, the score should be low (<0.5).
- Code Quality matters. Correct but messy code should be penalized slightly.
- Logic Similarity allows for different valid approaches, but they must solve the same root cause.

Your final output must contain:
- A numeric "score" between 0 and 1.
- A short "reason" describing the main factors affecting the score.

Do NOT include any text outside the JSON object.
"""

correctness = GEval(
    name="Correctness",
    criteria=EVAL_CRITERIA,
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    strict_mode=True,
)

code_correctness = GEval(
    name="CodeCorrectness",
    criteria=CODE_EVAL_CRITERIA,
    evaluation_params=[
        LLMTestCaseParams.INPUT,         
        LLMTestCaseParams.ACTUAL_OUTPUT,  
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    strict_mode=True,
)
