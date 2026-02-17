import asyncio
import re
import threading
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    GetNodeNeighboursFromNodeIdTool,
)
from app.modules.intelligence.tools.impact_analysis.impact_analysis_config import (
    IMPACT_ANALYSIS_CONFIG,
    canonicalize_identifier,
    expand_identifier_variants,
    is_allowed_xml_path,
    is_xml_file,
    normalize_repo_relative_path,
    to_repo_relative_output_path,
)
from app.modules.intelligence.tools.impact_analysis.impact_analysis_schema import (
    Ambiguity,
    BlockedByScope,
    Evidence,
    ImpactAnalysisRequest,
    ImpactAnalysisResponse,
    ImpactTraceAnalysisInput,
    RecommendedTest,
    TracePath,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    KnowledgeGraphQueryTool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    GetCodeFromProbableNodeNameTool,
)
from app.modules.search.search_service import SearchService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImpactTraceAnalysisTool:
    name = "impact_trace_analysis"
    description = (
        "Analyze impact for a changed file/function using repository evidence and KG hints. "
        "Follow the chain Function -> XML mapping -> AutomationId/Name/ControlName/Accessibility -> "
        "UI trigger (FlaUI/StepDefinitions) -> Robot/SpecFlow tests, enforce XML scope constraints, "
        "and return minimum test recommendations with confidence and traceable evidence."
    )

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.search_service = SearchService(sql_db)
        self._kg_tool: Optional[KnowledgeGraphQueryTool] = None
        self._neighbour_tool: Optional[GetNodeNeighboursFromNodeIdTool] = None
        self._probable_node_tool: Optional[GetCodeFromProbableNodeNameTool] = None

    async def arun(
        self,
        project_id: str,
        changed_file: str,
        function_name: Optional[str] = None,
        module_hint: Optional[str] = None,
        strict_mode: bool = True,
        change_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.run,
            project_id,
            changed_file,
            function_name,
            module_hint,
            strict_mode,
            change_notes,
        )

    def run(
        self,
        project_id: str,
        changed_file: str,
        function_name: Optional[str] = None,
        module_hint: Optional[str] = None,
        strict_mode: bool = True,
        change_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = ImpactAnalysisRequest(
            changed_file=changed_file,
            function_name=function_name,
            module_hint=module_hint,
            strict_mode=strict_mode,
            change_notes=change_notes,
        )
        response = self._execute_analysis(project_id, request)
        return response.model_dump()

    def _execute_analysis(
        self, project_id: str, request: ImpactAnalysisRequest
    ) -> ImpactAnalysisResponse:
        response = ImpactAnalysisResponse()

        try:
            normalized_changed_file = normalize_repo_relative_path(request.changed_file)
        except ValueError as exc:
            response.ambiguities.append(
                Ambiguity(
                    field="changed_file",
                    message=str(exc),
                    candidates=[],
                )
            )
            return response

        function_name = (request.function_name or "").strip()
        file_only_mode = not function_name
        primary_symbol = (
            normalized_changed_file
            if file_only_mode
            else f"{normalized_changed_file}:{function_name}"
        )

        evidence_by_key: Dict[tuple[str, str, str], str] = {}
        evidence_index: Dict[str, Evidence] = {}
        kg_evidence_ids: List[str] = []

        if not file_only_mode:
            node_candidates = self._resolve_function_candidates(
                project_id,
                normalized_changed_file,
                function_name,
                request.module_hint,
            )

            if not node_candidates:
                response.ambiguities.append(
                    Ambiguity(
                        field="function_name",
                        message=(
                            "Could not deterministically locate the changed function in indexed code. "
                            "Confirm the file path/function name or provide module_hint."
                        ),
                        candidates=[],
                    )
                )
                return response

            if len(node_candidates) > 1:
                response.ambiguities.append(
                    Ambiguity(
                        field="function_name",
                        message=(
                            "Multiple candidates match this function. Provide module_hint if you want a single deterministic trace."
                        ),
                        candidates=[
                            f"{candidate.get('file_path')}:{candidate.get('name')}"
                            for candidate in node_candidates[:5]
                        ],
                    )
                )

            primary_candidate = node_candidates[0]
            primary_node_id = primary_candidate.get("node_id")

            declaration_evidence_id = self._add_evidence(
                response,
                evidence_by_key,
                evidence_index,
                evidence_type="function_declaration",
                file_path=primary_candidate.get("file_path") or normalized_changed_file,
                detail=(
                    f"Resolved changed function candidate '{function_name}' in {primary_candidate.get('file_path') or normalized_changed_file}."
                ),
                source="search_index",
                confidence="high",
                matched_text=primary_candidate.get("content"),
            )

            response.trace_paths.append(
                TracePath(
                    source=primary_symbol,
                    target=f"{primary_candidate.get('file_path')}:{primary_candidate.get('name')}",
                    relation="declares",
                    confidence="high",
                    evidence_ids=[declaration_evidence_id],
                )
            )

            kg_entries = self._query_knowledge_graph(
                project_id,
                function_name,
                [primary_node_id] if primary_node_id else [],
            )

            for kg_entry in kg_entries:
                file_path = to_repo_relative_output_path(
                    kg_entry.get("file_path", ""), project_id
                )
                if not file_path:
                    continue

                evidence_id = self._add_evidence(
                    response,
                    evidence_by_key,
                    evidence_index,
                    evidence_type="kg_reference",
                    file_path=file_path,
                    detail=kg_entry.get("query") or "KG-discovered related node",
                    source="knowledge_graph",
                    confidence="medium",
                    matched_text=kg_entry.get("docstring"),
                )
                kg_evidence_ids.append(evidence_id)
                response.trace_paths.append(
                    TracePath(
                        source=primary_symbol,
                        target=file_path,
                        relation="related_via_kg",
                        confidence="medium",
                        evidence_ids=[evidence_id],
                    )
                )

            neighbour_entries = self._expand_neighbours(
                project_id,
                [primary_node_id] if primary_node_id else [],
            )

            for neighbour in neighbour_entries:
                neighbour_name = neighbour.get("name") or neighbour.get("node_id") or "unknown"
                evidence_id = self._add_evidence(
                    response,
                    evidence_by_key,
                    evidence_index,
                    evidence_type="code_neighbour",
                    file_path=normalized_changed_file,
                    detail=f"Neighbour discovered: {neighbour_name}",
                    source="graph_neighbours",
                    confidence="medium",
                    matched_text=neighbour.get("docstring"),
                )
                response.trace_paths.append(
                    TracePath(
                        source=primary_symbol,
                        target=neighbour_name,
                        relation="graph_neighbour",
                        confidence="medium",
                        evidence_ids=[evidence_id],
                    )
                )

            discovered_identifiers = self._collect_identifier_tokens(
                function_name=function_name,
                change_notes=request.change_notes,
                module_hint=request.module_hint,
            )

            for identifier_payload in discovered_identifiers:
                original_identifier = identifier_payload["original"]
                variant = identifier_payload["variant"]
                normalized_identifier = identifier_payload["normalized"]
                search_hits = self._search_codebase(project_id, variant)

                for search_hit in search_hits:
                    candidate_path = to_repo_relative_output_path(
                        search_hit.get("file_path", ""), project_id
                    )
                    if not candidate_path:
                        continue

                    if is_xml_file(candidate_path):
                        if IMPACT_ANALYSIS_CONFIG.reject_xml_outside_scope and not is_allowed_xml_path(
                            candidate_path
                        ):
                            self._add_blocked_scope(
                                response,
                                file_path=candidate_path,
                                matched_identifier=original_identifier,
                                reason=(
                                    "XML candidate is outside allowed impact-analysis scopes."
                                ),
                            )
                            continue

                        evidence_id = self._add_evidence(
                            response,
                            evidence_by_key,
                            evidence_index,
                            evidence_type="xml_mapping",
                            file_path=candidate_path,
                            detail=f"Scoped XML mapping matched identifier '{variant}'.",
                            source="repository_search",
                            confidence="medium",
                            matched_text=search_hit.get("content"),
                            original_identifier=original_identifier,
                            normalized_identifier=normalized_identifier,
                        )
                        response.trace_paths.append(
                            TracePath(
                                source=primary_symbol,
                                target=candidate_path,
                                relation="maps_via_xml",
                                confidence="medium",
                                evidence_ids=[evidence_id],
                            )
                        )
                    elif self._is_ui_reference_file(candidate_path):
                        self._add_evidence(
                            response,
                            evidence_by_key,
                            evidence_index,
                            evidence_type="ui_reference",
                            file_path=candidate_path,
                            detail=f"UI reference matched identifier '{variant}'.",
                            source="repository_search",
                            confidence="medium",
                            matched_text=search_hit.get("content"),
                            original_identifier=original_identifier,
                            normalized_identifier=normalized_identifier,
                        )

        recommended_tests = self._recommend_tests(
            project_id=project_id,
            function_name=function_name or "",
            changed_file=normalized_changed_file,
            module_hint=request.module_hint,
            evidence_index=evidence_index,
            response=response,
            evidence_by_key=evidence_by_key,
            kg_evidence_ids=kg_evidence_ids,
            file_only_mode=file_only_mode,
        )

        if request.strict_mode:
            recommended_tests = [
                test for test in recommended_tests if test.confidence != "low"
            ]

        response.recommended_tests = recommended_tests

        if not response.recommended_tests:
            response.ambiguities.append(
                Ambiguity(
                    field="recommended_tests",
                    message=(
                        "No deterministic tests were found with sufficient evidence. "
                        "Provide module_hint or additional change_notes for tighter matching."
                    ),
                    candidates=[],
                )
            )

        return response

    def _resolve_function_candidates(
        self,
        project_id: str,
        changed_file: str,
        function_name: str,
        module_hint: Optional[str],
    ) -> List[Dict[str, Any]]:
        scoped_query = f"{changed_file} {function_name}"
        candidates = self._search_codebase(project_id, scoped_query)

        if module_hint:
            candidates.extend(
                self._search_codebase(
                    project_id,
                    f"{changed_file} {module_hint} {function_name}",
                )
            )

        if not candidates:
            candidates = self._search_codebase(project_id, function_name)

        scored: List[tuple[int, Dict[str, Any]]] = []
        changed_file_lower = changed_file.lower()
        function_name_lower = function_name.lower()

        for candidate in candidates:
            file_path = to_repo_relative_output_path(candidate.get("file_path", ""), project_id)
            name = str(candidate.get("name", ""))
            content = str(candidate.get("content", ""))

            score = 0
            if file_path.lower().endswith(changed_file_lower):
                score += 5
            if name.lower() == function_name_lower:
                score += 5
            if function_name_lower in content.lower():
                score += 2
            if changed_file_lower in file_path.lower():
                score += 2

            if score <= 0:
                continue

            resolved = dict(candidate)
            resolved["file_path"] = file_path
            scored.append((score, resolved))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            deduped: List[Dict[str, Any]] = []
            seen = set()
            for _score, candidate in scored:
                key = (candidate.get("node_id"), candidate.get("file_path"), candidate.get("name"))
                if key in seen:
                    continue
                deduped.append(candidate)
                seen.add(key)
            return deduped[:5]

        probable_node_candidates = self._resolve_probable_node_candidates(
            project_id,
            changed_file,
            function_name,
        )
        return probable_node_candidates

    def _resolve_probable_node_candidates(
        self, project_id: str, changed_file: str, function_name: str
    ) -> List[Dict[str, Any]]:
        tool = self._get_probable_node_tool()
        if not tool:
            return []

        probable_name = f"{changed_file}:{function_name}"
        try:
            results = tool.get_code_from_probable_node_name(project_id, [probable_name])
        except Exception:
            logger.exception("impact_trace_analysis probable-node resolution failed")
            return []

        candidates = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            if result.get("error"):
                continue
            candidates.append(
                {
                    "node_id": result.get("node_id"),
                    "name": function_name,
                    "file_path": to_repo_relative_output_path(
                        result.get("relative_file_path", ""), project_id
                    ),
                    "content": result.get("code_content", ""),
                }
            )
        return candidates

    def _query_knowledge_graph(
        self, project_id: str, function_name: str, node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        tool = self._get_kg_tool()
        if not tool:
            return []

        queries = [
            f"Where is {function_name} used?",
            f"What tests relate to {function_name}?",
        ]

        try:
            raw_results = tool.run(queries=queries, project_id=project_id, node_ids=node_ids)
        except Exception:
            logger.exception("impact_trace_analysis knowledge-graph query failed")
            return []

        flattened: List[Dict[str, Any]] = []
        for query, query_results in zip(queries, raw_results or []):
            for entry in query_results or []:
                if isinstance(entry, dict):
                    payload = dict(entry)
                else:
                    payload = {
                        "node_id": getattr(entry, "node_id", None),
                        "file_path": getattr(entry, "file_path", None),
                        "docstring": getattr(entry, "docstring", None),
                    }
                payload["query"] = query
                flattened.append(payload)
        return flattened

    def _expand_neighbours(
        self, project_id: str, node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not node_ids:
            return []

        tool = self._get_neighbour_tool()
        if not tool:
            return []

        try:
            result = tool.run(project_id=project_id, node_ids=node_ids)
        except Exception:
            logger.exception("impact_trace_analysis neighbour expansion failed")
            return []

        neighbors = result.get("neighbors") if isinstance(result, dict) else None
        if isinstance(neighbors, list):
            return [entry for entry in neighbors if isinstance(entry, dict)]
        return []

    def _collect_identifier_tokens(
        self,
        function_name: str,
        change_notes: Optional[str],
        module_hint: Optional[str],
    ) -> List[Dict[str, str]]:
        seed_identifiers = {
            "ControlName",
            "Accessibility",
            "automationIdentifiers",
        }
        if function_name:
            seed_identifiers.add(function_name)
        if module_hint:
            seed_identifiers.add(module_hint)

        if change_notes:
            for token in change_notes.replace("\n", " ").split(" "):
                token = token.strip(" ,.;:()[]{}")
                if len(token) >= 3:
                    seed_identifiers.add(token)

        payloads: List[Dict[str, str]] = []
        seen = set()
        for identifier in seed_identifiers:
            for variant in expand_identifier_variants(identifier):
                normalized = canonicalize_identifier(variant)
                key = (identifier, variant, normalized)
                if key in seen:
                    continue
                seen.add(key)
                payloads.append(
                    {
                        "original": identifier,
                        "variant": variant,
                        "normalized": normalized,
                    }
                )
        return payloads

    def _recommend_tests(
        self,
        project_id: str,
        function_name: str,
        changed_file: str,
        module_hint: Optional[str],
        evidence_index: Dict[str, Evidence],
        response: ImpactAnalysisResponse,
        evidence_by_key: Dict[tuple[str, str, str], str],
        kg_evidence_ids: List[str],
        file_only_mode: bool = False,
    ) -> List[RecommendedTest]:
        test_candidates: Dict[str, Dict[str, Any]] = {}

        for query in self._test_queries(
            function_name, changed_file, module_hint, file_only_mode
        ):
            for search_hit in self._search_codebase(project_id, query):
                file_path = to_repo_relative_output_path(
                    search_hit.get("file_path", ""), project_id
                )
                if not self._looks_like_test_path(file_path):
                    continue

                ref_target = function_name or changed_file
                evidence_id = self._add_evidence(
                    response,
                    evidence_by_key,
                    evidence_index,
                    evidence_type="test_reference",
                    file_path=file_path,
                    detail=(
                        f"Repository search query '{query}' matched test referencing '{ref_target}'."
                    ),
                    source="repository_search",
                    confidence="high",
                    matched_text=search_hit.get("content"),
                )

                test_names_from_hit = self._extract_test_names_from_search_hit(
                    file_path, search_hit
                )

                existing = test_candidates.get(file_path)
                confidence = self._derive_test_confidence(
                    file_path=file_path,
                    content=str(search_hit.get("content", "")),
                    function_name=function_name or changed_file.split("/")[-1],
                    source="search",
                )
                reason = (
                    "Direct repository reference to changed file"
                    if file_only_mode
                    else "Direct repository reference to changed function"
                )
                if existing:
                    existing["evidence_ids"].add(evidence_id)
                    existing["confidence"] = self._pick_stronger_confidence(
                        existing["confidence"], confidence
                    )
                    existing["test_names"].update(test_names_from_hit)
                else:
                    test_candidates[file_path] = {
                        "confidence": confidence,
                        "reason": reason,
                        "evidence_ids": {evidence_id},
                        "test_names": set(test_names_from_hit),
                    }

        # Promote KG-linked test files when present.
        for evidence_id in kg_evidence_ids:
            evidence = evidence_index.get(evidence_id)
            if not evidence:
                continue
            if not self._looks_like_test_path(evidence.file_path):
                continue

            existing = test_candidates.get(evidence.file_path)
            if existing:
                existing["evidence_ids"].add(evidence_id)
                existing["confidence"] = self._pick_stronger_confidence(
                    existing["confidence"], "medium"
                )
            else:
                test_candidates[evidence.file_path] = {
                    "confidence": "medium",
                    "reason": "Knowledge-graph linked test file",
                    "evidence_ids": {evidence_id},
                    "test_names": set(),
                }

        # Add low-confidence repository-aligned heuristics when deterministic mappings are sparse.
        for heuristic_test in self._heuristic_test_paths(changed_file):
            if heuristic_test in test_candidates:
                continue
            test_candidates[heuristic_test] = {
                "confidence": "low",
                "reason": (
                    "Heuristic repository path aligned to Robot/SpecFlow/FlaUI layout; "
                    "requires manual verification."
                ),
                "evidence_ids": set(),
                "test_names": set(),
            }

        recommended_tests: List[RecommendedTest] = []
        for file_path, payload in test_candidates.items():
            confidence = payload["confidence"]
            evidence_ids = sorted(payload["evidence_ids"])
            test_names: set[str] = payload.get("test_names") or set()
            if not evidence_ids:
                # Add explicit heuristic evidence only when no deterministic evidence is available.
                heuristic_evidence_id = self._add_evidence(
                    response,
                    evidence_by_key,
                    evidence_index,
                    evidence_type="heuristic_test_mapping",
                    file_path=file_path,
                    detail=(
                        "Generated from changed file naming convention; requires manual verification."
                    ),
                    source="heuristic",
                    confidence="low",
                )
                evidence_ids = [heuristic_evidence_id]

            test_ids = self._build_test_ids(file_path, test_names)

            recommended_tests.append(
                RecommendedTest(
                    name=file_path.split("/")[-1],
                    file_path=file_path,
                    test_ids=test_ids,
                    confidence=confidence,
                    reason=payload["reason"],
                    evidence_ids=evidence_ids,
                )
            )

            trace_source = changed_file if file_only_mode else f"{changed_file}:{function_name}"
            response.trace_paths.append(
                TracePath(
                    source=trace_source,
                    target=file_path,
                    relation="validated_by_test",
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                )
            )

        # Keep ordering deterministic.
        recommended_tests.sort(
            key=lambda item: (
                self._confidence_rank(item.confidence),
                item.file_path,
            )
        )
        return recommended_tests

    def _search_codebase(self, project_id: str, query: str) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        try:
            return self._run_async(self.search_service.search_codebase(project_id, query))
        except Exception:
            logger.exception(
                "impact_trace_analysis search query failed",
                project_id=project_id,
                query=query,
            )
            return []

    def _add_evidence(
        self,
        response: ImpactAnalysisResponse,
        evidence_by_key: Dict[tuple[str, str, str], str],
        evidence_index: Dict[str, Evidence],
        evidence_type: str,
        file_path: str,
        detail: str,
        source: str,
        confidence: str,
        matched_text: Optional[str] = None,
        original_identifier: Optional[str] = None,
        normalized_identifier: Optional[str] = None,
    ) -> str:
        relative_file_path = to_repo_relative_output_path(file_path)
        key = (evidence_type, relative_file_path, detail)
        if key in evidence_by_key:
            return evidence_by_key[key]

        evidence_id = f"e{len(response.evidence) + 1}"
        evidence = Evidence(
            id=evidence_id,
            type=evidence_type,
            file_path=relative_file_path,
            detail=detail,
            source=source,
            confidence=confidence,
            matched_text=matched_text,
            original_identifier=original_identifier,
            normalized_identifier=normalized_identifier,
        )
        response.evidence.append(evidence)
        evidence_by_key[key] = evidence_id
        evidence_index[evidence_id] = evidence
        return evidence_id

    def _add_blocked_scope(
        self,
        response: ImpactAnalysisResponse,
        file_path: str,
        reason: str,
        matched_identifier: Optional[str] = None,
    ) -> None:
        relative_path = to_repo_relative_output_path(file_path)
        exists = any(
            blocked.file_path == relative_path and blocked.reason == reason
            for blocked in response.blocked_by_scope
        )
        if exists:
            return
        response.blocked_by_scope.append(
            BlockedByScope(
                file_path=relative_path,
                reason=reason,
                matched_identifier=matched_identifier,
            )
        )

    def _get_kg_tool(self) -> Optional[KnowledgeGraphQueryTool]:
        if self._kg_tool is not None:
            return self._kg_tool
        try:
            self._kg_tool = KnowledgeGraphQueryTool(self.sql_db, self.user_id)
        except Exception:
            logger.exception("Failed to initialize KnowledgeGraphQueryTool")
            self._kg_tool = None
        return self._kg_tool

    def _get_neighbour_tool(self) -> Optional[GetNodeNeighboursFromNodeIdTool]:
        if self._neighbour_tool is not None:
            return self._neighbour_tool
        try:
            self._neighbour_tool = GetNodeNeighboursFromNodeIdTool(self.sql_db)
        except Exception:
            logger.exception("Failed to initialize GetNodeNeighboursFromNodeIdTool")
            self._neighbour_tool = None
        return self._neighbour_tool

    def _get_probable_node_tool(self) -> Optional[GetCodeFromProbableNodeNameTool]:
        if self._probable_node_tool is not None:
            return self._probable_node_tool
        try:
            self._probable_node_tool = GetCodeFromProbableNodeNameTool(
                self.sql_db, self.user_id
            )
        except Exception:
            logger.exception("Failed to initialize GetCodeFromProbableNodeNameTool")
            self._probable_node_tool = None
        return self._probable_node_tool

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}

        def _runner():
            try:
                result_holder["result"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")

    @staticmethod
    def _looks_like_test_path(file_path: str) -> bool:
        if not file_path:
            return False
        lowered = file_path.lower()
        return (
            "/test" in lowered
            or "/tests/" in lowered
            or "/testcode/" in lowered
            or lowered.endswith("_test.py")
            or lowered.endswith(".spec.ts")
            or lowered.endswith(".spec.js")
            or lowered.endswith(".test.ts")
            or lowered.endswith(".test.js")
            or lowered.endswith(".robot")
            or lowered.endswith(".feature")
            or lowered.endswith("tests.cs")
            or "robottestcaselayer" in lowered
            or "vmm.functional.tests" in lowered
            or ("stepdefinition" in lowered and "flaui" in lowered and lowered.endswith(".cs"))
        )

    @staticmethod
    def _is_ui_reference_file(file_path: str) -> bool:
        lowered = file_path.lower()
        return lowered.endswith((".xaml", ".cs", ".csproj", ".json", ".config"))

    @staticmethod
    def _derive_test_confidence(
        file_path: str,
        content: str,
        function_name: str,
        source: str,
    ) -> str:
        lowered_path = (file_path or "").lower()
        lowered_content = (content or "").lower()
        lowered_function = function_name.lower()

        if source == "search" and lowered_function in lowered_content:
            return "high"
        if lowered_function in lowered_path:
            return "high"
        return "medium"

    @staticmethod
    def _pick_stronger_confidence(current: str, incoming: str) -> str:
        order = {"high": 3, "medium": 2, "low": 1}
        return incoming if order.get(incoming, 0) > order.get(current, 0) else current

    @staticmethod
    def _extract_test_names_from_search_hit(
        file_path: str, search_hit: Dict[str, Any]
    ) -> List[str]:
        """Extract runnable test identifiers from a search hit (name + content)."""
        names: set[str] = set()
        lowered_path = file_path.lower()
        hit_name = search_hit.get("name", "")
        if hit_name and (
            hit_name.startswith("test_")
            or hit_name.startswith("Test")
            or "test" in hit_name.lower()
        ):
            names.add(hit_name)

        content = str(search_hit.get("content", ""))
        # Python: def test_*, async def test_*
        for match in re.finditer(
            r"(?:async\s+)?def\s+(test_\w+)\s*\(", content, re.IGNORECASE
        ):
            names.add(match.group(1))
        # Python: class Test* with methods
        for match in re.finditer(
            r"class\s+(Test\w+)\s*[:(]", content
        ):
            names.add(match.group(1))
        # JavaScript/TypeScript: it(, test(, describe(
        for match in re.finditer(
            r"(?:it|test|describe)\s*\(\s*[\"']([^\"']+)[\"']", content
        ):
            names.add(match.group(1))

        # Robot/FlaUI keywords.
        for match in re.finditer(r"\b(Step\w+_FlaUI)\b", content):
            names.add(match.group(1))

        # SpecFlow binding attributes.
        for match in re.finditer(r'\[(?:Given|When|Then)\s*\(@"([^"]+)"\)\]', content):
            names.add(match.group(1))
        for match in re.finditer(r'\[(?:Given|When|Then)\s*\("([^"]+)"\)\]', content):
            names.add(match.group(1))

        # Feature scenario names.
        if lowered_path.endswith(".feature"):
            for match in re.finditer(
                r"(?im)^\s*Scenario(?: Outline)?:\s*(.+?)\s*$", content
            ):
                names.add(match.group(1).strip())

        return sorted(names)

    @staticmethod
    def _build_test_ids(file_path: str, test_names: set[str]) -> List[str]:
        """Build runnable test IDs across pytest, Robot/FlaUI, and SpecFlow styles."""
        lowered_path = file_path.lower()
        if not test_names:
            return [file_path]

        if lowered_path.endswith(".robot"):
            return sorted(f"{file_path}::keyword:{name}" for name in test_names)

        if lowered_path.endswith(".feature"):
            return sorted(f"{file_path}::scenario:{name}" for name in test_names)

        if lowered_path.endswith(".cs") and any(
            name.startswith("Step") and name.endswith("_FlaUI") for name in test_names
        ):
            return sorted(f"{file_path}::step:{name}" for name in test_names)

        return sorted(f"{file_path}::{name}" for name in test_names)

    @staticmethod
    def _test_queries(
        function_name: str,
        changed_file: str,
        module_hint: Optional[str],
        file_only_mode: bool = False,
    ) -> Iterable[str]:
        changed_file_name = changed_file.split("/")[-1]
        stem = changed_file_name.rsplit(".", 1)[0] if "." in changed_file_name else changed_file_name

        queries: List[str] = []
        if function_name:
            queries.extend(
                [
                    f"{function_name} test",
                    f"{function_name} tests",
                    f"{function_name} .robot",
                    f"{function_name} .feature",
                    f"{function_name} Step_FlaUI",
                    f"{function_name} automationIdentifiers",
                ]
            )
        queries.extend(
            [
                f"{changed_file_name} test",
                f"{stem} test",
                f"{changed_file_name} .robot",
                f"{changed_file_name} .feature",
                f"{stem} Step_FlaUI",
            ]
        )
        if module_hint:
            queries.extend(
                [
                    f"{module_hint} test",
                    f"{module_hint} .robot",
                    f"{module_hint} .feature",
                    f"{module_hint} Step_FlaUI",
                ]
            )

        context_blob = f"{changed_file} {module_hint or ''}".lower()
        if "primarydisplay" in context_blob:
            queries.extend(
                [
                    "PrimaryDisplayControls.xml",
                    "syncFusionDataGrid_Metadata.xml",
                    "ControlName Accessibility",
                ]
            )

        seen = set()
        for query in queries:
            if query and query not in seen:
                seen.add(query)
                yield query

    @staticmethod
    def _heuristic_test_paths(changed_file: str) -> List[str]:
        if not changed_file:
            return []

        parts = changed_file.split("/")
        file_name = parts[-1]
        stem = file_name.rsplit(".", 1)[0]

        if ImpactTraceAnalysisTool._looks_like_test_path(changed_file):
            return [changed_file]

        pascal_stem = ImpactTraceAnalysisTool._to_pascal_case(stem)
        candidates = [
            f"TestCode/RobotTestcaseLayer/Regression/{stem}.robot",
            f"_Release/Platform/VMM/VMM/VMM.Functional.Tests/{stem}.feature",
            (
                "TestCode/FlaUITaskLayer/PrimaryDisplayUI/Modules/PrimaryDisplay/"
                f"PrimeDisp{pascal_stem}.cs"
            ),
        ]

        if stem.lower().startswith("primedisp"):
            candidates.append(
                "TestCode/FlaUITaskLayer/PrimaryDisplayUI/Modules/PrimaryDisplay/"
                f"{file_name}"
            )

        if len(parts) > 1:
            directory = "/".join(parts[:-1])
            candidates.append(f"tests/{directory}/test_{stem}.py")
        candidates.append(f"tests/test_{stem}.py")

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    @staticmethod
    def _to_pascal_case(value: str) -> str:
        parts = re.split(r"[^a-zA-Z0-9]+", value)
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    @staticmethod
    def _confidence_rank(confidence: str) -> int:
        ranks = {"high": 0, "medium": 1, "low": 2}
        return ranks.get(confidence, 99)


def get_impact_trace_analysis_tool(sql_db: Session, user_id: str) -> StructuredTool:
    tool = ImpactTraceAnalysisTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool.arun,
        func=tool.run,
        name="impact_trace_analysis",
        description=tool.description,
        args_schema=ImpactTraceAnalysisInput,
    )
