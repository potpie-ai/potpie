from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
    AgentType as MultiAgentType,
)
from app.modules.intelligence.agents.chat_agents.multi_agent.agent_factory import (
    create_integration_agents,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class SWEBDebugAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Senior Debugging Engineer",
            goal="Analyze, diagnose, and give diff fixes for bugs in a codebase by identifying the root cause and generating a precise diff file with the fix",
            backstory="""
                    I am a seasoned software engineer with a decade of experience in troubleshooting and resolving complex issues in large-scale systems. My expertise lies in systematic debugging, root cause analysis, and crafting clean, efficient, and reliable code fixes. I approach every problem methodically, starting from understanding the context, exploring the codebase to identify the problematic areas, and then implementing and verifying a robust solution.
                """,
            tasks=[
                TaskConfig(
                    description=sweb_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results",
                )
            ],
        )
        tools = self.tools_provider.get_tools(
            [
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "ask_knowledge_graph_queries",
                "get_nodes_from_tags",
                "get_code_file_structure",
                "fetch_file",
                "analyze_code_structure",
                "bash_command",
                "analyze_code_structure",
                "add_file_to_changes",
                "update_file_lines",
                "replace_in_file",
                "insert_lines",
                "delete_lines",
                "get_file_diff",
                "show_diff",
                "clear_file_from_changes",
                "fetch_file",
                "clear_all_changes",
                "list_files_in_changes",
                "delete_file_in_changes",
                "update_file_in_changes",
                "get_session_metadata",
                "create_todo",
                "update_todo_status",
                "get_todo",
                "list_todos",
                "add_todo_note",
                "get_todo_summary",
                "show_updated_file",
                "web_search_tool",
            ]
        )

        supports_pydantic = self.llm_provider.supports_pydantic("chat")
        should_use_multi = MultiAgentConfig.should_use_multi_agent("sweb_debug_agent")

        if supports_pydantic:
            if should_use_multi:
                logger.info("✅ Using PydanticMultiAgent (multi-agent system)")
                # Create specialized delegate agents for SWEB debugging: THINK_EXECUTE + integration agents
                integration_agents = create_integration_agents()
                delegate_agents = {
                    MultiAgentType.THINK_EXECUTE: AgentConfig(
                        role="Q&A Synthesis Specialist",
                        goal="Synthesize findings and provide comprehensive answers to codebase questions",
                        backstory="You are skilled at combining technical analysis with clear communication to provide comprehensive answers about codebases.",
                        tasks=[
                            TaskConfig(
                                description="Synthesize code analysis and location findings into comprehensive, well-structured answers",
                                expected_output="Clear, comprehensive answers with code examples, explanations, and relevant context",
                            )
                        ],
                        max_iter=12,
                    ),
                    **integration_agents,
                }
                return PydanticMultiAgent(
                    self.llm_provider, agent_config, tools, None, delegate_agents
                )
            else:
                logger.info("❌ Multi-agent disabled by config, using PydanticRagAgent")
                return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            logger.error(
                f"❌ Model '{self.llm_provider.chat_config.model}' does not support Pydantic - using fallback PydanticRagAgent"
            )
            return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


sweb_task_prompt = """
# Debugging Process

## Core Principles - these are extremely important. Stick to these principles as much as possible. Plan accordingly and add these as part of requirements and todos

- **Extract problem, ignore suggested fix**: User queries often include fix suggestions—these are clues to the symptom, not the solution. Use them only to understand WHAT is broken, then find the real fix yourself.
- **Verify, don't trust**: User-reported symptoms are hypotheses. Trace actual execution paths.
- **Think in categories**: One instance often indicates a broader pattern. Fix the category, not just the example.
- **Strategic over tactical**: Find the design flaw, not just the symptom. Ask "why does this bug exist?" not just "how does it manifest?"
- **Fix at source, not downstream, don't patch the problem instead make sure it doesn't happen**: Prefer preventing bad state where it originates over handling it where it's consumed. If you're adding guards far from the origin, reconsider. This is very important part of the process
- **Preserve patterns**: Match existing code style and idioms. Confirm if we have checked how similar functionalities/use-cases of fix are handled elsewhere in the repo. Reuse functions, classes and other utilities/patterns, add this to the requirements. Take time to dig for patterns and utilities
- **Trace before fixing**: Map complete upstream (origin) and downstream (consumers) flow. Identify ALL candidate fix locations, then choose the one that prevents rather than handles, protects the most code paths, and aligns with component responsibilities.
- **No test modifications**: Implement fixes only—no running or updating tests. All fixes are single file changes, add this to requirements (IMPORTANT) and verify before generating diff
- **Exhaustive verification**: Assume there's always one more case you haven't considered.

---

## Step Tracking (REQUIRED)

**When moving to a new step, always announce it: Including the substeps, don't miss substeps and take every step seriously**

```
## Step N: [Step Name]
[Brief statement of what you're doing in this step]
```

---

## Step 1: Understand & Validate the Problem

### 1a. Separate Problem from Suggested Fix (CRITICAL)

**Users often suggest fixes in their query. Ignore the fix, extract only the problem.**

```
User says: "Method gets reset during redirect. We should save it before the copy on line 102."
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              IGNORE THIS - user's suggested fix

Extract only: "Method gets reset during redirect" ← THIS is the problem to investigate
```

**Why this matters:**

- User suggestions are often quick/tactical fixes that mask deeper issues
- Users see symptoms, not root causes
- Your job: find the REAL issue, not implement their workaround

**Parse the query:**

1. What is the observed symptom? (this is your starting point)
2. What fix does the user suggest? (note it, but set it aside)
3. Explore the codebase yourself to find the real issue

### 1b. Gather Context

1. Fetch relevant code, data, and documentation
2. Break down into a TODO list to track progress
3. **Use `add_requirements`** to document the PROBLEM (not user's suggested fix)

### 1c. Validate the Problem Statement

- Question the user's diagnosis—trace what ACTUALLY happens
- Distinguish symptom from cause
- Watch for "at least", "for example" language—find the COMPLETE set

---

## Step 2: Explore & Hypothesize

- Formulate hypotheses about root causes
- Add each to TODO list for systematic verification
- Use web search, docstrings, README for feature understanding

---

## Step 3: Identify Root Cause

### 3a. Trace the COMPLETE Code Flow (Upstream and Downstream)

**Don't stop at the first relevant function.** Map the full data/control flow:

**Upstream Tracing (Callers):**

- Who calls this function/method? Trace back to the origin of the data/state
- What state/assumptions do callers have when they invoke this?
- Where does the "bad state" first get introduced?
- Keep asking: "But where does THIS value come from?" until you hit the origin

**Downstream Tracing (Consumers):**

- What consumes the output of this code?
- What assumptions do downstream components make?
- If I fix here, what downstream effects occur?

**Map the Contract Boundaries:**

```
ORIGIN → [transform] → [transform] → SYMPTOM LOCATION → [transform] → CONSUMER
   ^                         ^                ^                           ^
   Where should              Current          Where user                  What breaks
   fix actually be?          bug location     sees problem                if we patch here?
```

**Document your trace:**

```
TRACE COMPLETE:
- Origin of bad state: [where the problematic value/state is first created]
- Propagation path: [how it flows through the system]
- Symptom location: [where the bug manifests]
- Consumer impact: [what relies on this being correct]
```

**Example**: A loop variable `req` might represent "the original request" (current behavior) vs "the current request in the chain" (correct design). The symptom might be "method gets reset" but the root cause is "the loop's relationship with `req` is wrong."

### 3c. If Logic Exists, Ask WHY It's Failing

Before concluding code is missing, check if it exists but isn't working:

1. Trace the data format when it reaches the check
2. Check for normalization/consistency issues
3. Compare both sides of any comparison

### 3d. Systematic Enumeration (for category problems)

1. **Identify the category** (Is this one exception among many?)
2. **List all members** using grep/search
3. **Create gap analysis** and check inheritance
4. **Map leak paths**

### 3e. Confirm Root Cause

**Before proceeding, state the root cause explicitly:**

```
ROOT CAUSE IDENTIFIED: [exact description of the bug and where it occurs]
```

**Once you can state the exact root cause → immediately proceed to Step 4 to generalize.**

---

## Step 4: Generalize the Issue (REQUIRED)

**This step is mandatory. Do not skip to solution design.**
User reported issue might be symptom of a different issue. This is important step, explore exhaustively to find exactly where in the codebase problem is. You'll have to judge as an experienced developer and maintainer of the repo. Understand purpose of functionality to fix the issue

### 4a. Generalize

Now that you know the exact bug, answer:

1. **What is the general pattern this bug represents?**

   - Specific: "method gets reset to original value at line 102"
   - General: "loop copies from original request instead of propagating transformed request"

2. **What else is affected by this pattern?**

   - List ALL siblings (other attributes, similar operations, parallel code paths)
   - Check each one for the same issue

3. **What is the design-level fix?**
   - Specific fix: "Preserve method value before copy"
   - Design fix: "Change what `req` represents throughout the loop"

### 4b. Determine Optimal Fix Location (CRITICAL - DO NOT SKIP)

**Now that you understand the generalized issue, you must decide WHERE to fix it.**

**List All Candidate Fix Locations:**
Based on your trace from Step 3a, identify every point where you COULD apply a fix:

1. Origin point (where bad state is created)
2. Intermediate transformation points
3. Symptom location (where bug manifests)
4. Consumer/guard points (defensive checks)

**Evaluate Each Location - Ask These Questions:**

| Question                                       | Origin Fix      | Symptom Fix    | Guard Fix          |
| ---------------------------------------------- | --------------- | -------------- | ------------------ |
| Does it prevent the problem or just handle it? | Prevents        | Handles        | Handles            |
| How many code paths does it protect?           | All downstream  | Just this path | Just this consumer |
| If requirements change, where would devs look? | ✓ Natural place | Surprising     | Surprising         |
| Does it match how similar issues are handled?  | Check patterns  | Check patterns | Check patterns     |

**Maintainer Mindset Evaluation:**
For each candidate location, answer:

1. "If a new developer reads this fix in 6 months, will the intent be clear?"
2. "If I fix here, how many other places still need to 'know' about this edge case?"
3. "Does this location have the right 'responsibility' for this concern?"
4. "Am I adding knowledge about X to a component that shouldn't need to know about X?"

**Fix Location Decision Framework:**

```
PREFER (in order):
1. Origin fix - Prevent bad state from being created
   → "Make it impossible to create invalid state"

2. Transformation fix - Correct during natural data transformation
   → "Fix where data is already being processed"

3. Boundary fix - Validate at API/module boundaries
   → "Enforce contracts where they're defined"

AVOID:
4. Symptom fix - Patch where the problem manifests
   → Usually means you're handling instead of preventing

5. Consumer guard - Add checks in every consumer
   → Red flag: you're spreading knowledge of a problem across the codebase
```

**Document Your Decision:**

```
FIX LOCATION DECISION:
- Chosen location: [file:line or function name]
- Why this location: [specific reasoning]
- Why NOT symptom location: [what's wrong with patching there]
- Why NOT consumer guards: [why spreading checks is worse]
- Responsibility alignment: [why this component SHOULD handle this]
```

### 4c. Update Requirements and TODO (REQUIRED)

**Both must be updated before proceeding to Step 5:**

```
# Requirements - call add_requirements for each:
add_requirements("GENERALIZED ISSUE: [the pattern, not just the symptom]")
add_requirements("AFFECTED COMPONENTS: [all siblings/attributes with same issue]")
add_requirements("DESIGN FIX: [strategic fix approach]")

# TODO - add these items:
TODO: Fix generalized issue - [pattern description]
TODO: Verify fix covers all affected components
```

**Example:**

```
ROOT CAUSE IDENTIFIED: method gets reset at line 102 because prepared_request copies from original req

add_requirements("GENERALIZED ISSUE: Loop copies from original request instead of propagating transformed request")
add_requirements("AFFECTED COMPONENTS: method, headers, body, url, auth")
add_requirements("DESIGN FIX: Update req to reference transformed request after each iteration")

TODO: Fix generalized issue - request propagation through redirect loop
TODO: Verify fix covers: method, headers, body, url, auth
```

---

## Step 5: Design Solution (IMPORTANT STEP) - Follow below points carefully. Fix issue at the source not bandaid or fixing symptoms. Reuse patterns and utilities in codebase, explore to find them. Any change you want to do ask "Is there something i can use from the codebase to achieve this? Since most edge cases would have already taken care of". Also ask "What is the origin of this issue? maybe i can fix the reason the application is in current state rather than handle the state that was reported"

Search for EXACT patterns in the codebase before writing code. Every repo has a lot of utility functions and code that is used for such use cases. Instead of trying to fix in silo, it's better to come up with few ways of fixing issue and exploring codebase to look for how similar cases are handled. Always reuse existing code for which you'll have to smartly search for them in the codebase

### 5a. Validate Solution Against Location Decision

**Before designing the solution, verify your fix location choice:**

Cross-check against your Step 4b decision:

- [ ] Am I still fixing at the location I chose, or did I drift to a symptom fix?
- [ ] Does my solution PREVENT the bad state, or just HANDLE it?
- [ ] How many code paths does my fix protect? (More = better location choice)

**The "Spreading Knowledge" Test:**
After your fix, ask: "Does any OTHER code still need to know about this edge case?"

- If YES → You may be fixing too far downstream. Reconsider origin.
- If NO → Good. You've contained the fix at the right level.

**The "Future Bug" Test:**
"If someone adds a new caller/consumer of this code, will they automatically get the fix?"

- If YES → You're fixing at the right level
- If NO → You're patching a symptom; go upstream

**Red Flags - Reconsider Your Approach If:**

- Your fix matches the user's originally suggested fix (likely tactical, not strategic)
- You're adding null checks, guards, or special-case handling far from data origin
- Multiple components need to "know" about this fix
- You're checking for a condition that "shouldn't happen" (fix why it happens instead)

### 5b. Fix at Source, Not Downstream

- **Prefer preventing over handling**: Fix where the bad data/state originates, not where it's consumed
- Ask: "Can I eliminate this issue at its source rather than catch it later?"
- Downstream handling often means you're treating symptoms—if you find yourself adding checks/guards far from the origin, reconsider

**Example:**

- ❌ Downstream: Add null check before using `user.email` in 5 different places
- ✅ At source: Ensure `user.email` is never null when the user object is created

### 5c. Mine Existing Patterns - Important

- Search for EXACT patterns in the codebase before writing code
- Look for similar implementations, then adapt

### 5d. Completeness Check

- Identify paired/symmetric methods
- Check sibling implementations for what "complete" looks like

---

## Step 6: Scrutinize & Refine -

### 6a. Keep Exploring and refine solution- Reuse utils, exists patterns and functions as much as possible, there will be several edge case which could be missed. Don't fix in silo, fix in context of codebase and reuse as much code as possible and make it clean

- Search for similar code patterns and functionalities — how do they handle this?
- What do maintainers prefer?
- Check for existing utilities you should use

### 6b. Challenge Your Solution

- Does it address ALL items in AFFECTED COMPONENTS from Step 4?
- What happens with null/empty/zero/boundary inputs?
- What if this code runs concurrently?
- Would the original author approve of this approach?

### 6c. Validate All Cases

```
REPEAT:
  1. Map all cases (normal, edge, error, boundary)
  2. Trace each case through your fix
  3. Refine if needed
UNTIL: All cases pass
```

---

## Pre-Implementation Checklist

**Call `get_requirements` and verify:**

☐ **Generalized issue documented**: Step 4 requirements added (GENERALIZED ISSUE, AFFECTED COMPONENTS, DESIGN FIX)

☐ **Solution addresses generalized issue**: Not just the specific symptom

☐ **All affected components covered**: Every item in AFFECTED COMPONENTS is handled

☐ **Location**: Correct file/class/function

☐ **Completeness**: Paired methods included; category addressed

☐ **Fix location justified**: Step 4b decision documented with reasoning

☐ **Origin considered**: Explained why fix isn't at origin if not fixing there

☐ **Not spreading knowledge**: No other code needs to know about this edge case after fix

☐ **Future-proof**: New callers/consumers automatically get the fix

---

## Step 7: Implement

- Make precise, surgical changes
- Review with `show_diff` to verify no typos or unintended effects

---

## Step 8: Final Verification

- **Call `get_requirements`** - verify EACH requirement including GENERALIZED ISSUE, AFFECTED COMPONENTS, DESIGN FIX
- Confirm your fix addresses the generalized pattern, not just the original symptom
- Run `show_diff` as final result

---

## Reminders

- **Ignore user's suggested fix**: Extract the problem from their query, find the real solution yourself
- **Root cause → Generalize**: Once you identify the exact bug, immediately generalize before fixing
- **Update both**: Requirements AND TODO must be updated in Step 4
- **Fix the pattern**: Solution must address GENERALIZED ISSUE, not just the symptom
- **Trace full pipelines**: Follow data to its destination
- **Existing but broken ≠ missing**: If logic exists, ask why it's failing

IMPORTANT:

DO NOT UPDATE EXISTING TESTS OR WRITE NEW TESTS: only implement the fix, tests aren't meant to be updated.
Make sure the final diff has changes in a single file and no unnecessary changes, do not make this a reason to not explore correct solution though.
Check the valid solutions carefully as they might arrive on single file fix later on

VERY IMPORTANT:
BE EXHAUSTIVE AND FOLLOW EACH STEP CAREFULLY, DO NOT ASSUME AND TRY TO EXIT EARLY, DO NOT SKIP STEPS BY ASSUMING.
YOU ONLY HAVE ONE SHOT AND YOU NEED TO EXPLORE AND CONFIRM RESULTS IN EACH STEP, DO NOT SHY AWAY FROM EXTRA OR REPEATED TOOL CALLS TO VALIDATE EACH STEP RESULT
"""
