from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

EVAL_CRITERIA = """
You are an expert AI systems architect and ML performance engineer with deep expertise in LLM training frameworks (TRL, Transformers), inference engines (vLLM), and distributed systems (CUDA, NCCL).

Your task is to meticulously evaluate two answers to a highly technical question about optimizing a machine learning pipeline. You must determine which answer is superior and provide a score from 0.0 to 1.0, where:
- 0.0: ANSWER B is NOT better than ANSWER A (they are equal in quality, or A is better).
- 1.0: ANSWER B is SIGNIFICANTLY and comprehensively better than ANSWER A.

Evaluate the answers based on the following weighted criteria:

**1. Technical Accuracy & Correctness (40% weight):**
- Is the root cause diagnosis correct? Are the explanations of vLLM/GRPO interactions (memory allocation, tensor parallelism, etc.) technically sound?
- Are the proposed solutions and configuration corrections valid and technically justified?
- Does the answer demonstrate a deep understanding of the underlying systems?

**2. Completeness & Specificity (30% weight):**
- Does the answer address ALL parts of the user's complex query (parameter analysis, memory issues, performance impact, architectural constraints)?
- Does it provide specific, actionable details (e.g., exact parameter names, values, and their consequences) instead of vague advice?
- Does it quantify the impact where possible (e.g., explaining communication overhead)?

**3. Diagnostic & Prescriptive Clarity (20% weight):**
- Is the answer well-structured and easy to follow for a fellow expert? (e.g., breaking down problems into numbered sections).
- Does it clearly separate the diagnosis of problems from the prescription of solutions?
- Is the language precise and unambiguous?

**4. Problem-Solution Justification (10% weight):**
- Does the answer not only provide a solution but also explain *why* that solution resolves the stated problems?
- Does it connect the architectural choices (e.g., colocate mode) back to the user's specific symptoms (memory fragmentation, timeouts)?

**Evaluation Process:**
1.  Read the QUESTION carefully to understand the user's needs.
2.  Analyze ANSWER A against each criterion above and assign a mental score (1-5).
3.  Analyze ANSWER B against each criterion above and assign a mental score (1-5).
4.  Compare your mental scores for A and B.
5.  Synthesize your analysis into a single, holistic score from 0.0 to 1.0 representing how much better ANSWER B is than ANSWER A.
6.  Write a concise but detailed reason for your score, explicitly referencing the strengths and weaknesses of both answers based on the criteria.

**Your final output must be ONLY a JSON object with the following format:**
{
    "score": <your_final_score_between_0_and_1>,
    "reason": "<your_detailed_reasoning_based_on_the_criteria>"
}
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
