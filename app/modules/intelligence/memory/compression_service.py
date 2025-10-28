"""
Context compression service for long-running agent conversations.
Uses AI to create technical summaries of agent execution history.
"""

import logging
from typing import List, Optional
from pydantic_ai.messages import ModelMessage, TextPart

logger = logging.getLogger(__name__)


class CompressionService:
    """
    Service for compressing message history into a "Briefing Document".
    This document contains both a high-level Execution Journal and a low-level Technical Scratchpad,
    designed for a Supervisor/Delegate agent architecture.
    Uses the provider service's call_llm method for LLM operations.
    """
    
    def __init__(self, llm_provider, tools=None):
        """
        Initialize compression service with the provider service.
        
        Args:
            llm_provider: Provider service instance (used for LLM calls).
            tools: Optional list of tools (kept for API compatibility, not used).
        """
        self.llm_provider = llm_provider
        self.tools = tools or []

    async def compress_message_history(
        self, 
        messages: List[ModelMessage], 
        original_user_query: str,
        project_id: Optional[str] = None,
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Create a rich "Briefing Document" summary of the agent's history.
        
        Retries on failure. Returns None if all retries are exhausted.
        
        Args:
            messages: List of ModelMessage objects from the agent's execution.
            original_user_query: The user's original high-level request.
            project_id: Optional project ID for context.
            max_retries: Number of retry attempts (default: 2).
            
        Returns:
            A summary string if successful, or None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"🗜️  Compressing {len(messages)} messages "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                
                history_text = self._format_messages(messages)
                
                user_prompt = self._create_summary_prompt(
                    history_text=history_text,
                    original_query=original_user_query,
                    project_id=project_id
                )
                
                llm_messages = [
                    {
                        "role": "system",
                        "content": "You are a Technical Scribe for a multi-agent AI system. Your task is to read a verbose execution history and create a concise but technically detailed 'Briefing Document' for the next agent."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
                
                # Use the provider service's call_llm method
                summary = await self.llm_provider.call_llm(
                    messages=llm_messages,
                    config_type="inference"  # Use inference config for compression tasks
                )
                
                logger.info(
                    f"✅ Compression complete: {len(messages)} messages → {len(summary)} chars"
                )
                
                return summary
                
            except Exception as e:
                logger.error(f"❌ Compression attempt {attempt + 1} failed: {e}", exc_info=True)
                if attempt == max_retries - 1:
                    logger.error("❌ All compression retries exhausted. Returning None.")
                    return None
                logger.info(f"🔄 Retrying compression (attempt {attempt + 2}/{max_retries})...")
        
        return None

    def _format_messages(self, messages: List[ModelMessage]) -> str:
        """
        Extract and format text content from messages.
        """
        formatted = []
        total_chars = 0
        MAX_TOTAL_CHARS = 400000  # ~100K token safety limit

        for i, msg in enumerate(messages):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content = part.content
                        if total_chars + len(content) > MAX_TOTAL_CHARS:
                            remaining = MAX_TOTAL_CHARS - total_chars
                            if remaining > 100:
                                content = content[:remaining] + "\n... (history truncated)"
                                formatted.append(f"**Step {i+1}**:\n{content}\n")
                            break
                        
                        formatted.append(f"**Step {i+1}**:\n{content}\n")
                        total_chars += len(content)
                
                if total_chars >= MAX_TOTAL_CHARS:
                    logger.warning(f"⚠️  History for summarizer was truncated at {MAX_TOTAL_CHARS} chars.")
                    break
        
        return "\n".join(formatted) if formatted else "[No text content found]"
    
    def _create_summary_prompt(
        self,
        history_text: str,
        original_query: str,
        project_id: Optional[str],
    ) -> str:
        """
        Create the "Briefing Document" prompt, which includes both a journal and a scratchpad.
        """
        project_context = f"\nProject ID: {project_id}" if project_id else ""
        
        return rf"""You are a Technical Scribe for a multi-agent AI system. Your task is to read a verbose execution history and create a concise but technically detailed "Briefing Document" for the next agent.

**User's Original High-Level Goal**:
"{original_query}"{project_context}

---
**Full Execution History You Must Summarize**:
{history_text}
---

**YOUR TASK: GENERATE A BRIEFING DOCUMENT WITH THE FOLLOWING SECTIONS**

## 1. Master Plan Snapshot
- **Goal:** Provide the strategic context.
- **Action:** Find the **LAST** "📋 Current Todo List" in the history and reproduce it exactly.

## 2. BANNED ACTIONS (Critical Anti-Loop Section)
- **Goal:** List ALL work that was ALREADY DONE. These actions are BANNED in the next cycle.
- **Action:** Create TWO lists:

### 2A. BANNED TOOL CALLS (Do NOT call these tools with these arguments)
For each tool that was called, write in this EXACT format:
```
🚫 fetch_file("prisma/schema.prisma") - THIS FILE WAS ALREADY READ
🚫 AskKnowledgeGraphQueries about "Stripe integration" - THIS WAS ALREADY ANSWERED
🚫 get_file_content("src/app/api/route.ts") - THIS FILE WAS ALREADY EXAMINED
```

### 2B. BANNED PHRASES (Do NOT start responses with these)
```
🚫 "Let me first explore..." - EXPLORATION IS COMPLETE
🚫 "Let me examine..." - EXAMINATION IS COMPLETE  
🚫 "I'll help you implement... Let me first..." - NO! Continue from where you left off
🚫 "Let's look at..." - YOU ALREADY LOOKED
🚫 "Let's check..." - YOU ALREADY CHECKED
```

### 2C. WORK ALREADY COMPLETED
- ✅ ALREADY EXPLORED: [list what was explored]
- ✅ ALREADY EXAMINED: [list what was examined]
- ✅ CODE ALREADY WRITTEN: [list code that was generated]
- ✅ FILES ALREADY READ: [list with full paths]

**FORMAT REQUIREMENT:** 
- Each banned tool call must include the EXACT tool name and argument
- Each banned phrase must be a direct quote
- Make it IMPOSSIBLE to accidentally repeat these actions

## 3. Technical Scratchpad (Known Facts)
- **Goal:** Provide the essential technical context and discoveries that are CONFIRMED and should be TRUSTED.
- **Action:** Extract the following details that were DISCOVERED (not guessed):
    - **🛑 CORE TECHNOLOGY STACK:** Language, Framework, ORM, etc. (Mandatory)
    - **File Paths ALREADY READ:** List all files that were opened/read
    - **Key Imports ALREADY SEEN:** e.g., `import Stripe from 'stripe';`
    - **Function Signatures ALREADY FOUND:** e.g., `async function processPayment(amount: int)`
    - **Critical Logic ALREADY UNDERSTOOD:** Business logic that was analyzed
    - **Environment Variables ALREADY IDENTIFIED:** Any env vars found
    - **Database Models ALREADY EXAMINED:** Key fields and relationships seen
    - **API Endpoints ALREADY REVIEWED:** Routes, methods, and formats discovered

## 4. What To Do NEXT (Not Repeat)
- **Goal:** Provide the NEXT STEP only - not what was already done.
- **Action:** 
    - State what the current phase is (e.g., "Exploration phase COMPLETE. Design phase starting.")
    - State the ONE next concrete action (e.g., "Create database schema", NOT "Explore database schema")
    - If stuck in exploration loop, state: "STOP exploring. Start implementing: [specific task]"

---
**EXAMPLE OUTPUT:**

## 1. Master Plan Snapshot
-   🔄 ⚡ **Analyze System** - in_progress
-   ⏳ ⚡ **Design Schema** - pending

## 2. BANNED ACTIONS (Critical Anti-Loop Section)

### 2A. BANNED TOOL CALLS
```
🚫 fetch_file("prisma/schema.prisma") - THIS FILE WAS ALREADY READ
🚫 get_file_content("src/actions/stripe.ts") - THIS FILE WAS ALREADY EXAMINED
🚫 AskKnowledgeGraphQueries about "database schema" - THIS WAS ALREADY ANSWERED
🚫 AskKnowledgeGraphQueries about "Stripe integration" - THIS WAS ALREADY ANSWERED
```

### 2B. BANNED PHRASES
```
🚫 "Let me first explore the codebase..." - EXPLORATION IS COMPLETE
🚫 "Let me examine the Prisma schema..." - EXAMINATION IS COMPLETE
🚫 "I'll help you implement... Let me first..." - NO! Continue from where you left off
```

### 2C. WORK ALREADY COMPLETED
-   ✅ ALREADY EXPLORED: The codebase structure, reservation patterns
-   ✅ ALREADY EXAMINED: Database schema, Stripe integration
-   ✅ CODE ALREADY WRITTEN: Prisma schema for ParkingSlot and ParkingReservation models
-   ✅ FILES ALREADY READ: prisma/schema.prisma, src/actions/stripe.ts

## 3. Technical Scratchpad (Known Facts)
-   **🛑 CORE TECHNOLOGY STACK**:
    -   **Language**: TypeScript
    -   **ORM**: Prisma
    -   **Payment**: Stripe
-   **File Paths ALREADY READ**: `prisma/schema.prisma`, `src/actions/stripe.ts`
-   **Database Models ALREADY EXAMINED**: `User` (id, email), `Restaurant` (id, name)
-   **Key Imports ALREADY SEEN**: `import Stripe from 'stripe';`

## 4. What To Do NEXT (Not Repeat)
-   **Current Phase:** Exploration phase COMPLETE. Design phase starting.
-   **Next Action:** Create database schema for `ParkingSlot` and `ParkingReservation` models
-   **DO NOT:** Re-explore or re-examine files already listed in section 2
---

**NOW, GENERATE THE BRIEFING DOCUMENT BASED ON THE PROVIDED HISTORY.**
"""