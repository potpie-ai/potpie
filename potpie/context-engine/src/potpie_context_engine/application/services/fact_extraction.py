"""Opt-in document claim extraction, resolution, and guarded graph writes."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import unicodedata
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from typing import Any, Callable

from potpie_context_engine.adapters.outbound.local_paths import default_home
from potpie_context_engine.adapters.outbound.resources.fact_inbox import FactInbox
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.core.ontology import EDGE_TYPES, ENTITY_TYPES
from potpie_context_engine.core.identity import IdentityClass
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.graph_service import GraphEntitySearchRequest
from potpie_context_engine.core.ports.resource_store import format_resource_id
from potpie_context_engine.core.semantic_mutations import SemanticMutationRequest

_AUTO_KEY_TYPES = frozenset(
    {"Service", "Environment", "DataStore", "Cluster", "Feature", "Team"}
)


def _ontology_prompt() -> str:
    entities = ", ".join(
        sorted(name for name, spec in ENTITY_TYPES.items() if spec.public)
    )
    edges = "; ".join(
        f"{name}: {spec.description}"
        for name, spec in EDGE_TYPES.items()
        if spec.public and name not in {"SECTION_OF", "ELEMENT_OF", "VERIFIED"}
    )
    return f"Entity types: {entities}. Allowed predicates: {edges}."


def _configured_llm(text: str) -> dict[str, Any]:
    """Call an explicitly selected user key or a local Ollama endpoint."""
    provider = os.environ.get("POTPIE_FACTS_LLM_PROVIDER", "").strip().lower()
    key = os.environ.get("POTPIE_FACTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not provider:
        provider = "openai" if key else "ollama"
    if provider not in {"openai", "ollama"}:
        raise ValueError("POTPIE_FACTS_LLM_PROVIDER must be openai or ollama")
    if provider == "openai" and not key:
        raise ValueError(
            "set POTPIE_FACTS_LLM_API_KEY or OPENAI_API_KEY for extraction"
        )
    base = os.environ.get("POTPIE_FACTS_LLM_BASE_URL") or (
        "https://api.openai.com/v1"
        if provider == "openai"
        else "http://127.0.0.1:11434"
    )
    model = os.environ.get("POTPIE_FACTS_LLM_MODEL") or (
        "gpt-4o-mini" if provider == "openai" else "llama3.1"
    )
    system = (
        "Extract only explicit subject-predicate-object facts supported by an exact "
        'quote in this chunk. Return JSON {"facts":[{"subject":string,"subject_type":string,'
        '"predicate":string,"object":string,"object_type":string,'
        '"quote":string,"confidence":number,"page":integer|null}]}. '
        "Use at most five facts. If uncertain, return an empty list. "
        + _ontology_prompt()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text[:8000]},
    ]
    if provider == "ollama":
        url = base.rstrip("/") + "/api/chat"
        body = {"model": model, "messages": messages, "stream": False, "format": "json"}
    else:
        url = base.rstrip("/") + "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    headers = {"Content-Type": "application/json"}
    if key and provider == "openai":
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = json.load(response)
    content = (
        raw.get("message", {}).get("content", "")
        if provider == "ollama"
        else raw["choices"][0]["message"]["content"]
    )
    if not isinstance(content, str):
        raise ValueError("LLM returned no JSON text")
    result = json.loads(content)
    if result == []:
        return {"facts": []}
    if not isinstance(result, dict) or not isinstance(result.get("facts"), list):
        raise ValueError("LLM response must contain a facts array")
    return result


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _slug(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:80]


def _model_signature() -> str:
    """Identify extraction behavior without storing a credential."""
    provider = os.environ.get("POTPIE_FACTS_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = (
            "openai"
            if os.environ.get("POTPIE_FACTS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            else "ollama"
        )
    base = os.environ.get("POTPIE_FACTS_LLM_BASE_URL") or (
        "https://api.openai.com/v1"
        if provider == "openai"
        else "http://127.0.0.1:11434"
    )
    model = os.environ.get("POTPIE_FACTS_LLM_MODEL") or (
        "gpt-4o-mini" if provider == "openai" else "llama3.1"
    )
    identity = f"facts-v1\0{provider}\0{base.rstrip('/')}\0{model}"
    return hashlib.sha256(identity.encode()).hexdigest()


@dataclass(slots=True)
class FactExtractor:
    facade: ResourceFacade
    inbox: FactInbox | None = None
    llm: Callable[[str], dict[str, Any]] = _configured_llm
    confidence_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.inbox is None:
            self.inbox = FactInbox(
                home=getattr(self.facade.store, "home", default_home())
            )

    def run(self, *, pot_id: str, doc: str) -> dict[str, Any]:
        manifest = self.facade.current_manifest(pot_id=pot_id, slug=doc)
        self.inbox.close_superseded(pot_id=pot_id, doc=doc, revision=manifest.revision)
        verified = 0
        proposed = 0
        suppressed = 0
        errors: list[str] = []
        model_signature = _model_signature() if self.llm is _configured_llm else None
        for section in manifest.sections:
            for ref in section.chunks:
                uri = format_resource_id(
                    doc, section.slug, ref.seq, revision=manifest.revision
                )
                chunk = self.facade.get(pot_id=pot_id, resource_ids=(uri,))
                if not isinstance(chunk, tuple) or not chunk:
                    continue
                text_hash = hashlib.sha256(chunk[0].text.encode()).hexdigest()
                if model_signature and self.inbox.chunk_done(
                    pot_id=pot_id,
                    doc=doc,
                    revision=manifest.revision,
                    chunk_uri=uri,
                    model_signature=model_signature,
                    text_hash=text_hash,
                ):
                    continue
                try:
                    result = self.llm(chunk[0].text)
                except HTTPError as exc:
                    if exc.code == 429:
                        errors.append(
                            "fact extraction stopped: LLM quota or rate limit reached "
                            "(HTTP 429); retry after the provider allows more requests"
                        )
                        break
                    if 500 <= exc.code <= 599:
                        errors.append(
                            f"fact extraction stopped: LLM provider temporarily unavailable "
                            f"(HTTP {exc.code}); retry later"
                        )
                        break
                    errors.append(f"{uri}: {exc}")
                    continue
                except Exception as exc:
                    errors.append(f"{uri}: {exc}")
                    continue
                facts = result.get("facts", [])
                errors_before = len(errors)
                for raw in facts[:5]:
                    if not isinstance(raw, dict):
                        continue
                    fact = self._fact(raw, uri=uri, page=ref.page)
                    try:
                        reason, subject_key, object_key = self._gate(
                            fact,
                            chunk_text=chunk[0].text,
                            pot_id=pot_id,
                            available_pages=chunk[0].page_numbers,
                            source_kind=manifest.source_kind,
                        )
                    except Exception as exc:
                        reason, subject_key, object_key = (
                            f"graph check unavailable: {exc}",
                            None,
                            None,
                        )
                    item = self.inbox.propose(
                        pot_id=pot_id,
                        doc=doc,
                        revision=manifest.revision,
                        payload=fact,
                        reason=reason or "high confidence; awaiting graph write",
                    )
                    if not item["inserted"] or item["status"] != "pending":
                        suppressed += 1
                        continue
                    if reason:
                        proposed += 1
                        continue
                    try:
                        committed = self._commit(
                            pot_id=pot_id,
                            fact=fact,
                            subject_key=subject_key,
                            object_key=object_key,
                            item_id=item["id"],
                        )
                    except Exception as exc:
                        errors.append(f"{uri}: graph write failed: {exc}")
                        committed = False
                    if committed:
                        self.inbox.transition(
                            pot_id=pot_id, item_id=item["id"], status="verified"
                        )
                        verified += 1
                    else:
                        proposed += 1
                if model_signature and len(errors) == errors_before:
                    self.inbox.mark_chunk_done(
                        pot_id=pot_id,
                        doc=doc,
                        revision=manifest.revision,
                        chunk_uri=uri,
                        model_signature=model_signature,
                        text_hash=text_hash,
                    )
            if errors and errors[-1].startswith("fact extraction stopped:"):
                break
        return {
            "verified": verified,
            "proposed": proposed,
            "suppressed": suppressed,
            "error": "; ".join(errors[:3]) if errors else None,
        }

    @staticmethod
    def _fact(raw: dict[str, Any], *, uri: str, page: int | None) -> dict[str, Any]:
        fact = {
            key: str(raw.get(key) or "").strip()[:500]
            for key in (
                "subject",
                "subject_type",
                "predicate",
                "object",
                "object_type",
                "quote",
            )
        }
        try:
            fact["confidence"] = float(raw.get("confidence", 0))
        except (ValueError, TypeError):
            fact["confidence"] = 0.0
        if not math.isfinite(fact["confidence"]):
            fact["confidence"] = 0.0
        fact["chunk_uri"] = uri
        reported_page = raw.get("page")
        fact["page"] = (
            reported_page
            if isinstance(reported_page, int)
            and not isinstance(reported_page, bool)
            and reported_page > 0
            else page
        )
        return fact

    def _resolve(
        self, *, pot_id: str, name: str, entity_type: str
    ) -> tuple[str | None, str | None]:
        spec = ENTITY_TYPES.get(entity_type)
        if spec is None or not spec.public or not name or self.facade.graph is None:
            return None, "unknown entity type or missing name"
        result = self.facade.graph.search_entities(
            GraphEntitySearchRequest(
                pot_id=pot_id,
                query=name,
                type=entity_type,
                limit=10,
            )
        )
        exact = [
            entity
            for entity in result.entities
            if entity_type in entity.labels
            and _normalized(entity.name or "") == _normalized(name)
        ]
        if len(exact) == 1:
            return exact[0].key, None
        if len(exact) > 1 or result.entities:
            return None, "ambiguous existing entity identity"
        if (
            spec.identity_class != IdentityClass.SLUG_ALIAS
            or entity_type not in _AUTO_KEY_TYPES
        ):
            return None, "entity needs an authoritative identity"
        slug = _slug(name)
        if not slug:
            return None, "entity name has no usable identity"
        return f"{spec.key_prefix}:{slug}", None

    def _gate(
        self,
        fact: dict[str, Any],
        *,
        chunk_text: str,
        pot_id: str,
        available_pages: tuple[int, ...] = (),
        source_kind: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        predicate = fact["predicate"]
        edge = EDGE_TYPES.get(predicate)
        if (
            edge is None
            or not edge.public
            or predicate in {"SECTION_OF", "ELEMENT_OF", "VERIFIED"}
        ):
            return "predicate is outside the supported ontology", None, None
        subject_type, object_type = fact["subject_type"], fact["object_type"]
        if not edge.allows((subject_type,), (object_type,)):
            return (
                "ontology does not allow these entity types for this relation",
                None,
                None,
            )
        if (
            not 0 <= fact["confidence"] <= 1
            or fact["confidence"] < self.confidence_threshold
        ):
            return "low extraction confidence", None, None
        if not fact["quote"] or fact["quote"] not in chunk_text:
            return "evidence quote is absent from cited chunk", None, None
        if source_kind == "image" or (
            source_kind == "pdf"
            and fact["quote"] not in "\n\n".join(
                block
                for block in chunk_text.split("\n\n")
                if not re.match(r"^Figure \d+\b", block)
            )
        ):
            return "visual or OCR evidence requires review", None, None
        if fact["page"] is not None and fact["page"] not in available_pages:
            return "cited page is absent from chunk provenance", None, None
        subject_key, subject_reason = self._resolve(
            pot_id=pot_id,
            name=fact["subject"],
            entity_type=subject_type,
        )
        object_key, object_reason = self._resolve(
            pot_id=pot_id,
            name=fact["object"],
            entity_type=object_type,
        )
        if subject_reason or object_reason:
            return subject_reason or object_reason, None, None
        if subject_key == object_key:
            return "self relation needs review", None, None
        if self.facade.claims is None:
            return "cannot check existing graph claims for conflicts", None, None
        claims = self.facade.claims.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id, subject_key_in=(subject_key,), predicate_in=(predicate,)
            )
        )
        if edge.singleton and any(row.object_key != object_key for row in claims):
            return "conflicts with an existing graph claim", None, None
        return None, subject_key, object_key

    def _commit(
        self,
        *,
        pot_id: str,
        fact: dict[str, Any],
        subject_key: str,
        object_key: str,
        item_id: str,
    ) -> bool:
        evidence = {
            "source_ref": fact["chunk_uri"],
            "metadata": {
                "quote_sha256": hashlib.sha256(
                    fact["quote"].encode("utf-8")
                ).hexdigest(),
                "page": fact["page"],
            },
        }
        payload = {
            "pot_id": pot_id,
            "idempotency_key": f"document-fact:{item_id}",
            "created_by": {
                "surface": "cli",
                "harness": "document-fact-extraction",
            },
            "operations": [
                {
                    "op": "link_entities",
                    "subject": {
                        "key": subject_key,
                        "type": fact["subject_type"],
                        "name": fact["subject"],
                    },
                    "predicate": fact["predicate"],
                    "object": {
                        "key": object_key,
                        "type": fact["object_type"],
                        "name": fact["object"],
                    },
                    "truth": "source_observation",
                    "confidence": fact["confidence"],
                    "description": (
                        f"{fact['subject']} {fact['predicate'].lower().replace('_', ' ')} "
                        f"{fact['object']}"
                    )[:500],
                    "evidence": [evidence],
                }
            ],
        }
        request = SemanticMutationRequest.parse(
            payload, approved_by="document-fact-extraction"
        )
        result = self.facade.graph.mutate(request)
        if not result.ok or result.status != "applied":
            return False
        if self.facade.claims is not None and result.claim_keys:
            live = self.facade.claims.find_claims(
                ClaimQueryFilter(
                    pot_id=pot_id,
                    claim_key_in=tuple(result.claim_keys),
                )
            )
            if not set(result.claim_keys).issubset({row.claim_key for row in live}):
                return False
        return True

    def approve(
        self,
        *,
        pot_id: str,
        item_id: str,
        subject_key: str | None = None,
        object_key: str | None = None,
    ) -> dict[str, Any]:
        item = self.inbox.get(pot_id=pot_id, item_id=item_id)
        if item is None or item["status"] != "pending":
            raise ValueError("pending fact candidate not found")
        fact = item["fact"]
        manifest = self.facade.current_manifest(pot_id=pot_id, slug=item["doc"])
        if manifest.revision != item["revision"]:
            self.inbox.transition(
                pot_id=pot_id, item_id=item_id, status="closed", reason="superseded"
            )
            raise ValueError("candidate belongs to a superseded document revision")
        chunk = self.facade.get(pot_id=pot_id, resource_ids=(fact["chunk_uri"],))
        if not isinstance(chunk, tuple) or not chunk:
            raise ValueError("cited chunk is unavailable")
        edge = EDGE_TYPES.get(fact["predicate"])
        if (
            edge is None
            or not edge.public
            or not edge.allows((fact["subject_type"],), (fact["object_type"],))
        ):
            raise ValueError("fact does not match the ontology")
        if fact["quote"] not in chunk[0].text:
            raise ValueError("evidence quote is absent from cited chunk")
        if fact["page"] is not None and fact["page"] not in chunk[0].page_numbers:
            raise ValueError("cited page is absent from chunk provenance")
        if subject_key is None:
            subject_key, subject_reason = self._resolve(
                pot_id=pot_id, name=fact["subject"], entity_type=fact["subject_type"]
            )
            if subject_reason:
                raise ValueError(subject_reason)
        if object_key is None:
            object_key, object_reason = self._resolve(
                pot_id=pot_id, name=fact["object"], entity_type=fact["object_type"]
            )
            if object_reason:
                raise ValueError(object_reason)
        for key, entity_type in (
            (subject_key, fact["subject_type"]),
            (object_key, fact["object_type"]),
        ):
            if not key or not key.startswith(
                ENTITY_TYPES[entity_type].key_prefix + ":"
            ):
                raise ValueError("selected entity key does not match its ontology type")
        if not self._commit(
            pot_id=pot_id,
            fact=fact,
            subject_key=subject_key,
            object_key=object_key,
            item_id=item_id,
        ):
            raise ValueError(
                "graph mutation was not applied; candidate remains pending"
            )
        self.inbox.transition(pot_id=pot_id, item_id=item_id, status="verified")
        return self.inbox.get(pot_id=pot_id, item_id=item_id)
