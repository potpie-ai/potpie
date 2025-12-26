import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from letta_client import Letta
import litellm
from app.modules.intelligence.memory.memory_interface import (
    MemoryInterface,
    MemorySearchResponse,
    MemorySearchResult,
)

logger = logging.getLogger(__name__)


class LettaService(MemoryInterface):
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv(
            "LETTA_SERVER_URL", "http://localhost:8283"
        )
        self.llm_config = llm_config or self._get_default_llm_config()

        self.client = Letta(base_url=self.base_url)

        self._agent_cache: Dict[str, str] = {}

        logger.info(f"Initialized LettaService with server at {self.base_url}")

    def _get_default_llm_config(self) -> dict:
        return {
            "model": os.getenv("LETTA_MODEL", "openai/gpt-4o-mini"),
            "embedding": os.getenv("LETTA_EMBEDDING", "openai/text-embedding-ada-002"),
        }

    def _get_extraction_prompt(self) -> str:
        return f"""Extract meaningful facts, preferences, and events from developer conversations.

## What to Extract

**DO extract:**
- Coding style preferences (naming conventions, formatting, patterns)
- Tool and framework preferences (editors, ORMs, libraries, databases)
- Technical skills and knowledge
- Project architecture and tech stack decisions
- Implementation actions (bugs fixed, features added, refactorings)
- Personal information relevant to work (name, timezone, preferred languages)
- Development workflow preferences

**DON'T extract:**
- Greetings and pleasantries ("hi", "thanks", "bye")
- Procedural conversation ("let me check", "looking at the code")
- Temporary/transient information ("loading...", "processing...")
- Generic statements without specific information ("code looks good")

## Format Rules

- Be specific and concise
- Use third person ("Uses X", "Prefers Y", not "I use X")
- For time-based events, include absolute dates: today is {datetime.now().strftime("%Y-%m-%d")}
- Return empty array if no relevant information

## Examples

**Input:** Hi, I'm Alice. I prefer using camelCase for variable names.
**Output:** {{"facts": ["Name is Alice", "Prefers camelCase for variable names"]}}

**Input:** This TypeScript project uses Prisma ORM with PostgreSQL database.
**Output:** {{"facts": ["Project uses TypeScript", "Project uses Prisma ORM", "Project database is PostgreSQL"]}}

**Input:** I fixed the authentication bug yesterday and deployed version 2.0 to production.
**Output:** {{"facts": ["Fixed authentication bug on {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}", "Deployed version 2.0 to production on {datetime.now().strftime('%Y-%m-%d')}"]}}

**Input:** Let me check the code structure.
**Output:** {{"facts": []}}

**Input:** I always use 4-space indentation and prefer functional programming patterns.
**Output:** {{"facts": ["Uses 4-space indentation", "Prefers functional programming patterns"]}}

**Input:** The weather is nice today.
**Output:** {{"facts": []}}

Return only the facts in JSON format with a "facts" array."""

    def _get_agent_name(self, user_id: str, project_id: Optional[str] = None) -> str:
        if project_id:
            return f"potpie-user-{user_id}-project-{project_id}"
        return f"potpie-user-{user_id}"

    async def _get_or_create_agent(
        self,
        user_id: str,
        project_id: Optional[str] = None,
    ) -> str:
        cache_key = f"{user_id}:{project_id}" if project_id else user_id

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent_name = self._get_agent_name(user_id, project_id)

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            agents_list = await loop.run_in_executor(
                None, lambda: self.client.agents.list()
            )

            for agent in agents_list:
                if hasattr(agent, "name") and agent.name == agent_name:
                    agent_id = agent.id
                    self._agent_cache[cache_key] = agent_id
                    logger.info(
                        f"[letta] Found existing agent: agent_id={agent_id}, user_id={user_id}, project_id={project_id}"
                    )
                    return agent_id

            agent = await loop.run_in_executor(
                None,
                lambda: self.client.agents.create(
                    name=agent_name,
                    model=self.llm_config["model"],
                    embedding=self.llm_config["embedding"],
                    metadata={"user_id": user_id, "project_id": project_id or "global"},
                ),
            )

            agent_id = agent.id
            self._agent_cache[cache_key] = agent_id

            logger.info(
                f"[letta] Created agent: agent_id={agent_id}, "
                f"user_id={user_id}, project_id={project_id}"
            )

            return agent_id

        except Exception as e:
            logger.error(f"Error creating Letta agent: {e}", exc_info=True)
            raise

    def _categorize_memory(
        self, memory_text: str, project_id: Optional[str]
    ) -> Dict[str, str]:
        memory_lower = memory_text.lower()

        episodic_indicators = [
            "fixed",
            "deployed",
            "implemented",
            "added",
            "removed",
            "refactored",
            "created",
            "updated",
            "deleted",
            "merged",
            "on 20",
            "yesterday",
            "today",
            "last week",
            "last month",
            "completed",
            "finished",
        ]

        is_episodic = any(
            indicator in memory_lower for indicator in episodic_indicators
        )
        memory_type = "episodic" if is_episodic else "semantic"

        project_indicators = [
            "project uses",
            "project has",
            "project is",
            "database is",
            "api is",
            "this project",
            "codebase uses",
            "repository uses",
            "stack includes",
            "prisma",
            "typeorm",
            "sequelize",
            "postgresql",
            "mysql",
            "mongodb",
            "nestjs",
            "fastapi",
            "django",
            "express",
            "react",
            "vue",
            "angular",
        ]

        user_indicators = [
            "name is",
            "prefers",
            "likes",
            "always uses",
            "favorite",
            "camelcase",
            "snake_case",
            "indentation",
            "timezone",
            "work hours",
            "coding style",
            "workflow",
        ]

        has_project_indicator = any(
            indicator in memory_lower for indicator in project_indicators
        )
        has_user_indicator = any(
            indicator in memory_lower for indicator in user_indicators
        )

        if has_user_indicator and not has_project_indicator:
            scope = "user"
        elif has_project_indicator or (project_id and not has_user_indicator):
            scope = "project"
        else:
            scope = "project" if project_id else "user"

        return {"scope": scope, "memory_type": memory_type}

    async def _extract_facts_with_llm(self, message: str) -> List[str]:
        try:
            import asyncio

            loop = asyncio.get_event_loop()

            prompt = self._get_extraction_prompt()
            model = self.llm_config.get("model", "gpt-4o-mini")

            response = await loop.run_in_executor(
                None,
                lambda: litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": message},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                ),
            )

            content = response.choices[0].message.content

            try:
                import json_repair

                parsed = json_repair.loads(content)
                facts = parsed.get("facts", [])
                return facts if isinstance(facts, list) else []
            except Exception as parse_error:
                logger.warning(
                    f"[letta] Failed to parse LLM response: {parse_error}, content: {content[:200]}"
                )
                return []

        except Exception as e:
            logger.error(f"[letta] Error extracting facts with LLM: {e}", exc_info=True)
            return []

    async def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            logger.info(
                f"[letta add] Adding memories: user_id={user_id}, "
                f"project_id={project_id}, messages_count={len(messages)}"
            )

            latest_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    latest_user_message = msg.get("content")
                    break

            if not latest_user_message:
                logger.warning("[letta add] No user message found in messages list")
                return {"results": []}

            facts = await self._extract_facts_with_llm(latest_user_message)

            if not facts:
                logger.info("[letta add] No facts extracted from message")
                return {"results": []}

            logger.info(f"[letta add] Extracted {len(facts)} facts: {facts}")

            import asyncio

            loop = asyncio.get_event_loop()

            stored_memories = []

            for fact in facts:
                categorization = self._categorize_memory(fact, project_id)
                scope = categorization["scope"]
                memory_type = categorization["memory_type"]

                # Store in the appropriate agent based on scope
                if scope == "user":
                    # User-scoped memories go to user-level agent (no project_id)
                    target_agent_id = await self._get_or_create_agent(user_id, None)
                    target_project_id = None
                else:
                    # Project-scoped memories go to project-specific agent
                    target_agent_id = await self._get_or_create_agent(
                        user_id, project_id
                    )
                    target_project_id = project_id

                passage = await loop.run_in_executor(
                    None,
                    lambda f=fact,
                    aid=target_agent_id: self.client.agents.passages.create(
                        agent_id=aid, text=f
                    ),
                )

                stored_memories.append(
                    {
                        "memory": fact,
                        "metadata": {
                            "user_id": user_id,
                            "project_id": target_project_id,
                            "passage_id": str(passage.id)
                            if hasattr(passage, "id")
                            else None,
                            "scope": scope,
                            "memory_type": memory_type,
                            **(metadata or {}),
                        },
                    }
                )

                logger.info(
                    f"[letta add] Stored fact: scope={scope}, "
                    f"type={memory_type}, agent={target_agent_id}, content='{fact[:60]}...'"
                )

            logger.info(
                f"[letta add] ✅ Stored {len(stored_memories)} memories as passages"
            )

            return {"results": stored_memories}

        except Exception as e:
            logger.error(f"Error adding messages to Letta memory: {e}", exc_info=True)
            raise

    async def search(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 5,
        include_user_preferences: bool = True,
        memory_type: Optional[str] = None,
    ) -> MemorySearchResponse:
        try:
            logger.info(
                f"[letta search] Searching: query='{query[:50]}...', "
                f"user_id={user_id}, project_id={project_id}, limit={limit}"
            )

            results = []

            if project_id:
                project_agent_id = await self._get_or_create_agent(user_id, project_id)
                project_results = await self._search_agent_memories(
                    agent_id=project_agent_id,
                    query=query,
                    limit=limit,
                    user_id=user_id,
                    project_id=project_id,
                )
                results.extend(project_results)

            if include_user_preferences or not project_id:
                user_agent_id = await self._get_or_create_agent(user_id, None)
                user_results = await self._search_agent_memories(
                    agent_id=user_agent_id,
                    query=query,
                    limit=limit,
                    user_id=user_id,
                    project_id=None,
                )
                results.extend(user_results)

            seen = set()
            unique_results = []
            for r in results:
                mem_id = r.metadata.get("memory_id") or r.memory
                if mem_id not in seen:
                    seen.add(mem_id)
                    unique_results.append(r)
                    if len(unique_results) >= limit:
                        break

            if memory_type:
                unique_results = [
                    r
                    for r in unique_results
                    if r.metadata.get("memory_type") == memory_type
                ]

            logger.info(
                f"[letta search] Returning {len(unique_results)} memories "
                f"(project={sum(1 for m in unique_results if m.metadata.get('project_id'))}, "
                f"user={sum(1 for m in unique_results if not m.metadata.get('project_id'))})"
            )

            return MemorySearchResponse(
                results=unique_results, total=len(unique_results)
            )

        except Exception as e:
            logger.error(f"Error searching Letta memories: {e}", exc_info=True)
            return MemorySearchResponse(results=[], total=0)

    async def _search_agent_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        try:
            import asyncio

            loop = asyncio.get_event_loop()

            search_result = await loop.run_in_executor(
                None,
                lambda: self.client.agents.passages.search(
                    agent_id=agent_id, query=query, top_k=limit
                ),
            )

            memories = []
            if hasattr(search_result, "results"):
                for result in search_result.results:
                    passage_text = getattr(result, "content", "")
                    passage_id = getattr(result, "id", "")

                    # Re-categorize to get scope and memory_type
                    categorization = self._categorize_memory(passage_text, project_id)

                    memories.append(
                        MemorySearchResult(
                            memory=passage_text,
                            metadata={
                                "passage_id": passage_id,
                                "user_id": user_id,
                                "project_id": project_id,
                                "scope": categorization["scope"],
                                "memory_type": categorization["memory_type"],
                            },
                            score=None,
                        )
                    )

            return memories

        except Exception as e:
            logger.error(f"Error searching agent memories: {e}", exc_info=True)
            return []

    async def get_all_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> MemorySearchResponse:
        try:
            logger.info(
                f"[letta get_all] user_id={user_id}, project_id={project_id}, limit={limit}"
            )

            agent_id = await self._get_or_create_agent(user_id, project_id)

            import asyncio

            loop = asyncio.get_event_loop()

            passages_page = await loop.run_in_executor(
                None, lambda: self.client.agents.passages.list(agent_id=agent_id)
            )

            passages = list(passages_page)

            memories = []
            for passage in passages:
                passage_text = getattr(passage, "text", "")

                # Re-categorize to get scope and memory_type
                categorization = self._categorize_memory(passage_text, project_id)

                memories.append(
                    MemorySearchResult(
                        memory=passage_text,
                        metadata={
                            "passage_id": getattr(passage, "id", ""),
                            "user_id": user_id,
                            "project_id": project_id,
                            "scope": categorization["scope"],
                            "memory_type": categorization["memory_type"],
                        },
                        score=None,
                    )
                )

            if limit and len(memories) > limit:
                memories = memories[:limit]

            logger.info(f"[letta get_all] Returning {len(memories)} memories")

            return MemorySearchResponse(results=memories, total=len(memories))

        except Exception as e:
            logger.error(f"Error getting all Letta memories: {e}", exc_info=True)
            return MemorySearchResponse(results=[], total=0)

    async def delete(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> bool:
        try:
            agent_id = await self._get_or_create_agent(user_id, project_id)

            import asyncio

            loop = asyncio.get_event_loop()

            if memory_ids:
                # Delete specific passages by ID
                deleted_count = 0
                for passage_id in memory_ids:
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda pid=passage_id: self.client.agents.passages.delete(
                                agent_id=agent_id, passage_id=pid
                            ),
                        )
                        deleted_count += 1
                        logger.info(f"[letta delete] Deleted passage {passage_id}")
                    except Exception as e:
                        logger.warning(
                            f"[letta delete] Failed to delete passage {passage_id}: {e}"
                        )

                logger.info(
                    f"[letta delete] Successfully deleted {deleted_count}/{len(memory_ids)} passages"
                )
                return deleted_count > 0
            else:
                # Delete all passages for the agent
                # First, get all passages
                passages_page = await loop.run_in_executor(
                    None, lambda: self.client.agents.passages.list(agent_id=agent_id)
                )
                passages = list(passages_page)

                deleted_count = 0
                for passage in passages:
                    passage_id = getattr(passage, "id", None)
                    if passage_id:
                        try:
                            await loop.run_in_executor(
                                None,
                                lambda pid=passage_id: self.client.agents.passages.delete(
                                    agent_id=agent_id, passage_id=pid
                                ),
                            )
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(
                                f"[letta delete] Failed to delete passage {passage_id}: {e}"
                            )

                logger.info(
                    f"[letta delete] Deleted all {deleted_count} passages for agent {agent_id}"
                )
                return deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting Letta memories: {e}", exc_info=True)
            return False

    def close(self) -> None:
        self._agent_cache.clear()
        logger.info("Closed LettaService connections")
