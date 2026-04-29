---
name: systematic-debugging
description: Structured diagnostic workflow for identifying root causes in prompt failures
trigger_hint: Use when the failure mode is unclear or multiple causes seem possible
---

# Systematic Debugging Skill

Use this structured workflow when diagnosing why an agent's prompt caused a failure.

## Step 1: Reproduce the Failure

Examine the exact sequence of events in the trace:
- What was the user's query?
- What tool calls did the agent make, in what order?
- What were the arguments to each tool call?
- What were the results of each tool call?
- What was the agent's final response?

Map out the actual execution path. Don't skip ahead to diagnosis.

## Step 2: Isolate the Deviation

Identify the FIRST point where the agent's behavior deviated from what a well-prompted agent would do:
- Did it call the wrong tool?
- Did it call the right tool with wrong arguments?
- Did it call a tool unnecessarily (redundant call)?
- Did it skip a tool it should have called?
- Did it misinterpret a tool's result?
- Did it ignore relevant information from a previous tool call?

Be specific: "Tool call #3 (search_colgrep) used a broad natural language query instead of keywords"

## Step 3: Trace Back to the Prompt

Now connect the deviation to the prompt:
- Read the current prompt section by section
- For each section, ask: "Does this section address the behavior that went wrong?"
- Identify one of three root cause types:

### Missing Instruction
The prompt simply doesn't tell the agent how to handle this situation.
- Example: No guidance on when to stop calling colgrep
- Fix: Add a new section with clear instructions

### Ambiguous Instruction
The prompt has relevant guidance but it's vague enough to be misinterpreted.
- Example: "Use colgrep to search the codebase" (doesn't specify how many times or when to stop)
- Fix: Make the instruction specific and actionable

### Conflicting Instruction
Two parts of the prompt give contradictory guidance.
- Example: "Be thorough in your search" vs implicit expectation of efficiency
- Fix: Resolve the conflict with a clear priority rule

## Step 4: Classify the Pattern

Categorize the failure so the fix addresses the pattern, not just the symptom:
- **Tool selection failure**: Agent doesn't know when to use which tool
- **Tool argument failure**: Agent calls the right tool but with wrong inputs
- **Redundancy failure**: Agent repeats actions without making progress
- **Sequencing failure**: Agent does things in the wrong order
- **Termination failure**: Agent doesn't know when to stop
- **Interpretation failure**: Agent misreads tool results

## Step 5: Propose Minimal Edit

Design the smallest prompt change that:
1. Addresses the ROOT CAUSE type from Step 3 (missing/ambiguous/conflicting)
2. Fixes the PATTERN from Step 4 (not just this one instance)
3. Doesn't break existing correct behavior
4. Is specific and actionable (not vague advice)

Good edits are:
- Concrete rules: "Never call colgrep more than 2 times per query"
- Decision trees: "If colgrep returns >5 results, use ripgrep to narrow. If <5, read directly."
- Anti-patterns: "Do NOT call colgrep with full sentences. Extract 2-3 keywords first."

Bad edits are:
- Vague: "Be more efficient with tool calls"
- Overly broad: Rewriting the entire tool usage section
- Symptoms-only: Adding a rule for this exact query without addressing the pattern
