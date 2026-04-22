"""
Qdrant Hybrid Indexing Service

# THREE EMBEDDING TYPES PER NODE
- dense:   384-dim (all-MiniLM-L6-v2)  → semantic similarity
- bm25:    sparse lexical scores        → keyword matching
- colbert: 48-dim × N tokens           → late interaction (code-specialized)

# PIPELINE
1. PARSE  → RepoMap.create_graph() → nx graph with nodes (file/func/class/interface)
2. INDEX  → index_nodes_to_qdrant()  → upserts to Qdrant hybrid collection
3. SEARCH → (TBD: dense prefetch → bm25 prefetch → colbert rerank)

# QUICKSTART

    from qdrant_client import QdrantClient
    from app.modules.parsing.knowledge_graph.qdrant_indexing_service import (
        create_hybrid_collection,
        index_nodes_to_qdrant,
    )

    client = QdrantClient(host="localhost", port=6333)
    collection = "hybrid_myproject"

    # Create collection schema (one-time)
    create_hybrid_collection(client, collection)

    # Index nodes (after RepoMap.create_graph())
    # nodes = list from nx_graph.nodes(data=True))
    count, bm25_tokens, colbert_tokens = index_nodes_to_qdrant(
        client, collection, nodes
    )

# MANUAL (if you need individual steps)

    from app.modules.parsing.knowledge_graph.qdrant_indexing_service import (
        build_dense_embeddings,
        build_bm25_vectors,
        build_colbert_embeddings,
        upsert_hybrid_points,
    )

    dense   = build_dense_embeddings(nodes)    # {node_id: [384 floats]}
    bm25    = build_bm25_vectors(nodes)       # {node_id: {token: score}}
    colbert = build_colbert_embeddings(nodes)  # {node_id: [[48 dims] × N tokens]}

    upsert_hybrid_points(client, collection, nodes, dense, bm25, colbert)

# SEARCH (coming soon)
# Dense prefetch → BM25 prefetch → ColBERT rerank pipeline TBD
"""

from __future__ import annotations

import importlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DENSE_MODEL = "all-MiniLM-L6-v2"
DENSE_DIM = 384  # standard for all-MiniLM-L6-v2

COLBERT_MODEL = "lightonai/LateOn-Code-edge"
COLBERT_DIM = 48  # per-token embedding dimension for LateOn-Code-edge
COLBERT_MAX_TOKENS = 300  # document-side token limit (query-side = 48)

BM25_K1 = 1.5
BM25_B = 0.75

# Node types to index (FILE nodes are excluded from indexing)
INDEXABLE_NODE_TYPES = {"FUNCTION", "CLASS", "INTERFACE"}

_VOCAB_DIR = Path.home() / ".potpie" / "qdrant_vocabs"


def _get_vocab_path(collection_name: str) -> Path:
    return _VOCAB_DIR / f"{collection_name}_bm25_vocabulary.json"


def _load_token_vocabulary(collection_name: str) -> Optional[List[str]]:
    vocab_path = _get_vocab_path(collection_name)
    if vocab_path.exists():
        try:
            with open(vocab_path) as f:
                data = json.load(f)
            return data.get("token_vocabulary")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load vocabulary from {vocab_path}: {e}")
    return None


def _save_token_vocabulary(collection_name: str, token_vocabulary: List[str]) -> None:
    _VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = _get_vocab_path(collection_name)
    with open(vocab_path, "w") as f:
        json.dump({"token_vocabulary": token_vocabulary}, f)


def _delete_token_vocabulary(collection_name: str) -> None:
    vocab_path = _get_vocab_path(collection_name)
    try:
        vocab_path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Failed to delete vocabulary file {vocab_path}: {e}")


def _build_staging_collection_name(collection_name: str) -> str:
    return f"{collection_name}__rebuild_{uuid.uuid4().hex}"


def _get_alias_target(client: Any, alias_name: str) -> Optional[str]:
    aliases = client.get_aliases().aliases
    for alias in aliases:
        if alias.alias_name == alias_name:
            return alias.collection_name
    return None


def _activate_collection_alias(
    client: Any,
    alias_name: str,
    new_collection_name: str,
) -> Optional[str]:
    from qdrant_client import models  # type: ignore

    previous_collection = _get_alias_target(client, alias_name)
    operations: List[Any] = []

    if previous_collection is not None:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=alias_name)
            )
        )

    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=new_collection_name,
                alias_name=alias_name,
            )
        )
    )

    client.update_collection_aliases(operations)
    return previous_collection


def delete_hybrid_collection(
    client: Any,
    collection_name: str,
    alias_name: Optional[str] = None,
) -> None:
    from qdrant_client import models  # type: ignore

    alias_target = _get_alias_target(client, alias_name) if alias_name else None

    if alias_name and alias_target is not None:
        client.update_collection_aliases(
            [
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            ]
        )

    collections_to_delete = {collection_name}
    if alias_target is not None:
        collections_to_delete.add(alias_target)

    for target_collection in collections_to_delete:
        if client.collection_exists(target_collection):
            client.delete_collection(target_collection)
        _delete_token_vocabulary(target_collection)


# ---------------------------------------------------------------------------
# Text builders  (different views per modality)
# ---------------------------------------------------------------------------


def build_dense_text(node: Dict[str, Any]) -> str:
    """
    Dense view: type + name + file_path.
    Compact summary-focused text for semantic similarity.
    """
    parts = [
        node.get("type", ""),
        node.get("name", ""),
        node.get("file", ""),
    ]
    return " ".join(parts)


def build_bm25_text(node: Dict[str, Any]) -> str:
    """
    BM25 view: name + file_path + raw code.
    Keyword/lexical text for BM25 matching.
    """
    parts = [
        node.get("name", ""),
        node.get("file", ""),
        node.get("text", "") or "",
    ]
    return "\n".join(parts)


def build_colbert_text(node: Dict[str, Any]) -> str:
    """
    ColBERT view: name + type + file_path + trimmed code.
    Trimmed code keeps essential tokens for late-interaction matching.
    """
    code = node.get("text", "") or ""
    # Strip docstrings / comments to reduce noise
    code = _strip_comments_and_docstrings(code)
    parts = [
        node.get("name", ""),
        node.get("type", ""),
        node.get("file", ""),
        code,
    ]
    return "\n".join(parts)


def _strip_comments_and_docstrings(code: str) -> str:
    """Remove docstrings and line comments to reduce ColBERT noise."""
    # Remove triple-quoted strings (docstrings)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    # Remove line comments
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"#.*", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# Dense embedding
# ---------------------------------------------------------------------------

_dense_model: Optional[SentenceTransformer] = None


def _get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense embedding model: {DENSE_MODEL}")
        _dense_model = SentenceTransformer(DENSE_MODEL, device="cpu")
        logger.info(f"Dense model loaded: {DENSE_MODEL}")
    return _dense_model


def build_dense_embeddings(
    nodes: List[Dict[str, Any]],
    text_field: str = "_dense_text",
) -> Dict[str, List[float]]:
    """
    Encode a list of graph nodes into dense vectors.

    Args:
        nodes: List of node dicts from create_graph() (nx graph node data dicts).
        text_field: Key to store the built text on each node dict (in-place).

    Returns:
        Dict mapping node_id (str) -> 384-dim dense vector list.
    """
    filtered = []
    for node in nodes:
        if node.get("type", "") not in INDEXABLE_NODE_TYPES:
            continue
        text = build_dense_text(node)
        node[text_field] = text
        filtered.append(node)

    if not filtered:
        return {}

    model = _get_dense_model()
    texts = [n[text_field] for n in filtered]

    logger.info(f"Encoding {len(texts)} dense vectors with {DENSE_MODEL}")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32).tolist()

    return {n["node_id"]: emb for n, emb in zip(filtered, embeddings)}


# ---------------------------------------------------------------------------
# BM25 sparse vectors
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + alphanumeric tokenization."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _compute_bm25_scores(
    corpus: List[str],
    avg_dl: float,
    doc_lens: List[int],
) -> List[Dict[str, float]]:
    """
    Compute per-document BM25 scores for all term-doc pairs.

    Returns list of dict: doc_idx -> {token -> bm25_score}
    """
    N = len(corpus)
    term_doc_freqs: Dict[str, int] = {}
    term_doc_scores: Dict[int, Dict[str, float]] = {i: {} for i in range(N)}

    for doc_idx, doc in enumerate(corpus):
        tokens = _tokenize(doc)
        unique_terms = set(tokens)
        for term in unique_terms:
            term_doc_freqs[term] = term_doc_freqs.get(term, 0) + 1

    for doc_idx, doc in enumerate(corpus):
        tokens = _tokenize(doc)
        dl = doc_lens[doc_idx]
        term_freqs: Dict[str, int] = {}
        for t in tokens:
            term_freqs[t] = term_freqs.get(t, 0) + 1

        for term, tf in term_freqs.items():
            df = term_doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            score = idf * numerator / denominator
            term_doc_scores[doc_idx][term] = score

    return [term_doc_scores[i] for i in range(N)]


def build_bm25_vectors(
    nodes: List[Dict[str, Any]],
    text_field: str = "_bm25_text",
) -> Dict[str, Any]:
    filtered = []
    for node in nodes:
        if node.get("type", "") not in INDEXABLE_NODE_TYPES:
            continue
        text = build_bm25_text(node)
        node[text_field] = text
        filtered.append(node)

    if not filtered:
        return {}

    texts = [n[text_field] for n in filtered]
    doc_lens = [len(_tokenize(t)) for t in texts]
    avg_dl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0

    all_tokens: set[str] = set()
    for doc in texts:
        all_tokens.update(_tokenize(doc))
    sorted_tokens = sorted(all_tokens)

    bm25_scores = _compute_bm25_scores(texts, avg_dl, doc_lens)

    result: Dict[str, Any] = {
        n["node_id"]: scores for n, scores in zip(filtered, bm25_scores)
    }
    result["__token_vocabulary__"] = sorted_tokens
    return result


# ---------------------------------------------------------------------------
# ColBERT embeddings
# ---------------------------------------------------------------------------

_colbert_model: Optional[Any] = None  # typed as Any; loaded via transformers
_colbert_tokenizer: Optional[Any] = None


def _normalize_colbert_multivector(
    token_vectors: Any, *, node_id: str
) -> List[List[float]]:
    arr = np.asarray(token_vectors, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"ColBERT embeddings for node '{node_id}' must be 2D; got shape {arr.shape}"
        )

    if arr.shape[1] != COLBERT_DIM:
        raise ValueError(
            f"ColBERT embedding dimension mismatch for node '{node_id}': "
            f"expected {COLBERT_DIM}, got {arr.shape[1]}"
        )

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (arr / norms).tolist()


def _get_colbert_model_and_tokenizer():
    """
    Load ColBERT model and tokenizer lazily.
    Uses the LateOn-Code-edge model (17M params, code-specialized).

    Falls back to transformers AutoModel / AutoTokenizer only when PyLate itself
    is not installed. Other PyLate import/runtime issues are raised explicitly.
    """
    global _colbert_model, _colbert_tokenizer
    if _colbert_model is None:
        try:
            pylate_models = importlib.import_module("pylate.models")
        except ModuleNotFoundError as exc:
            if exc.name != "pylate":
                raise

            from transformers import AutoModel, AutoTokenizer

            logger.info(
                "PyLate is not installed; falling back to transformers for %s",
                COLBERT_MODEL,
            )
            _colbert_tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL)
            _colbert_model = AutoModel.from_pretrained(COLBERT_MODEL)
            logger.info("ColBERT model loaded via transformers")
        else:
            logger.info(
                "Loading ColBERT model via pylate models.ColBERT: %s",
                COLBERT_MODEL,
            )
            _colbert_model = pylate_models.ColBERT(model_name_or_path=COLBERT_MODEL)
            _colbert_tokenizer = getattr(_colbert_model, "tokenizer", None)
            logger.info("ColBERT model loaded via pylate")
    return _colbert_model, _colbert_tokenizer


def build_colbert_embeddings(
    nodes: List[Dict[str, Any]],
    text_field: str = "_colbert_text",
) -> Dict[str, List[List[float]]]:
    """
    Encode a list of graph nodes into per-token ColBERT embeddings (multivector).

    Each document becomes a list of [COLBERT_DIM] vectors (one per token).

    Returns:
        Dict mapping node_id -> List of per-token vectors.
        Each value is List[List[float]] of shape [seq_len, COLBERT_DIM].
    """
    # Filter to indexable types
    filtered = []
    for node in nodes:
        if node.get("type", "") not in INDEXABLE_NODE_TYPES:
            continue
        text = build_colbert_text(node)
        node[text_field] = text
        filtered.append(node)

    if not filtered:
        return {}

    model, tokenizer = _get_colbert_model_and_tokenizer()
    assert model is not None
    model.eval()
    texts = [n[text_field] for n in filtered]

    logger.info(f"Encoding {len(texts)} ColBERT documents with {COLBERT_MODEL}")

    results: Dict[str, List[List[float]]] = {}

    # Process in batches to avoid OOM
    batch_size = 32
    for i in range(0, len(filtered), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_nodes = filtered[i : i + batch_size]

        if hasattr(model, "encode"):
            batch_embeddings = model.encode(  # type: ignore[attr-defined]
                batch_texts,
                batch_size=len(batch_texts),
                is_query=False,
                show_progress_bar=False,
            )

            for node, token_vecs in zip(batch_nodes, batch_embeddings):
                results[node["node_id"]] = _normalize_colbert_multivector(
                    token_vecs,
                    node_id=node["node_id"],
                )
            continue

        if tokenizer is None:
            raise RuntimeError(
                "Transformers ColBERT fallback requires a tokenizer, but none was loaded"
            )

        # Tokenize with [D] document prefix (ColBERT convention)
        inputs = tokenizer(  # type: ignore
            ["[D] " + t for t in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=COLBERT_MAX_TOKENS,
        )

        with torch.no_grad():
            outputs = model(**inputs)  # type: ignore[misc]
            hidden = outputs.last_hidden_state.detach().cpu().float().numpy()

        attention_mask = inputs["attention_mask"].cpu().numpy()

        for j, node in enumerate(batch_nodes):
            mask = attention_mask[j]
            token_vecs = hidden[j][mask == 1]
            results[node["node_id"]] = _normalize_colbert_multivector(
                token_vecs,
                node_id=node["node_id"],
            )

    logger.info(f"ColBERT encoded {len(results)} documents")
    return results


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------


def create_hybrid_collection(
    client: Any,
    collection_name: str,
    dense_dim: int = DENSE_DIM,
    colbert_dim: int = COLBERT_DIM,
    recreate: bool = False,
) -> None:
    """
    Create (or recreate) a Qdrant collection with three named vector fields:
      - dense_vector:  cosine, 384-dim dense
      - colbert_multivector: cosine, 48-dim multivector with MAX_SIM comparator
      - bm25_sparse: sparse BM25 (managed externally; stored as sparse_vector)

    Args:
        client: QdrantClient instance
        collection_name: target collection name
        dense_dim: dense vector dimension (default 384)
        colbert_dim: ColBERT per-token dimension (default 48)
        recreate: if True, delete existing collection first
    """
    from qdrant_client import models  # type: ignore

    existing = client.collection_exists(collection_name)
    if existing:
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(
                f"Collection '{collection_name}' already exists; skipping creation."
            )
            return

    logger.info(f"Creating Qdrant collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense_vector": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            ),
            "colbert_multivector": models.VectorParams(
                size=colbert_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
        sparse_vectors_config={
            "bm25_sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(f"Collection '{collection_name}' created successfully.")


def upsert_hybrid_points(
    client: Any,
    collection_name: str,
    nodes: List[Dict[str, Any]],
    dense_vectors: Dict[str, List[float]],
    bm25_vectors: Dict[str, Dict[str, float]],
    colbert_vectors: Dict[str, List[List[float]]],
    batch_size: int = 100,
) -> int:
    """
    Upsert one Qdrant point per node with three vector representations.

    Args:
        client: QdrantClient instance
        collection_name: target collection name
        nodes: list of node dicts from create_graph()
        dense_vectors: node_id -> 384-dim dense vector
        bm25_vectors: node_id -> {token: bm25_score}
        colbert_vectors: node_id -> [[f32; COLBERT_DIM]; seq_len]
        batch_size: points per upsert batch

    Returns:
        Total number of points upserted.
    """
    from qdrant_client import models  # type: ignore

    stored_vocab = bm25_vectors.get("__token_vocabulary__")
    if stored_vocab:
        token_vocabulary = stored_vocab
    else:
        token_vocabulary = _load_token_vocabulary(collection_name)
    if token_vocabulary is None:
        all_tokens: set[str] = set()
        for node_id, scores in bm25_vectors.items():
            if node_id == "__token_vocabulary__":
                continue
            all_tokens.update(scores.keys())
        token_vocabulary = sorted(all_tokens)
    token_to_idx = {t: i for i, t in enumerate(token_vocabulary)}

    points_to_upsert = []
    for node in nodes:
        node_id = node.get("node_id", "")
        if node.get("type", "") not in INDEXABLE_NODE_TYPES:
            continue
        if node_id not in dense_vectors:
            continue

        # Build sparse BM25 vector
        bm25_scores = bm25_vectors.get(node_id, {})
        if bm25_scores:
            indices, values = [], []
            for token, score in bm25_scores.items():
                if token in token_to_idx and score > 0:
                    indices.append(token_to_idx[token])
                    values.append(score)
            sparse_vec = models.SparseVector(indices=indices, values=values)
        else:
            sparse_vec = models.SparseVector(indices=[], values=[])

        colbert_multivector = colbert_vectors.get(node_id, [])
        if colbert_multivector:
            colbert_multivector = _normalize_colbert_multivector(
                colbert_multivector,
                node_id=node_id,
            )

        point = models.PointStruct(
            id=str(uuid.UUID(bytes=bytes.fromhex(node_id))),
            vector={
                "dense_vector": dense_vectors[node_id],
                "colbert_multivector": colbert_multivector,
                "bm25_sparse": sparse_vec,
            },
            payload={
                "node_id": node_id,
                "name": node.get("name", ""),
                "file_path": node.get("file", ""),
                "type": node.get("type", ""),
                "start_line": node.get("line", -1),
                "end_line": node.get("end_line", -1),
                # NOTE: we do NOT store the full text in Qdrant payload.
                # Fetch source text from Neo4j by node_id when needed.
            },
        )
        points_to_upsert.append(point)

    total_upserted = 0
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total_upserted += len(batch)
        logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

    logger.info(f"Total upserted: {total_upserted} points into '{collection_name}'")
    return total_upserted


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------


def index_nodes_to_qdrant(
    client: Any,
    collection_name: str,
    nodes: List[Dict[str, Any]],
    recreate_collection: bool = False,
    alias_name: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Full pipeline: build all three embedding types and upsert to Qdrant.

    Args:
        client: QdrantClient instance
        collection_name: target Qdrant collection
        nodes: list of node dicts from create_graph()
        recreate_collection: whether to recreate the collection schema
        alias_name: optional stable alias name for staged rebuilds

    Returns:
        Tuple of (num_nodes_indexed, num_bm25_tokens, num_colbert_dims)
    """
    target_collection_name = collection_name
    previous_collection_name: Optional[str] = None

    if alias_name and recreate_collection:
        target_collection_name = _build_staging_collection_name(collection_name)
        create_hybrid_collection(client, target_collection_name, recreate=True)
        previous_collection_name = _get_alias_target(client, alias_name)
        if previous_collection_name is None and client.collection_exists(
            collection_name
        ):
            previous_collection_name = collection_name
    else:
        # 1. Ensure collection exists
        create_hybrid_collection(
            client, target_collection_name, recreate=recreate_collection
        )

    try:
        dense_vecs = build_dense_embeddings(nodes)
        bm25_vecs = build_bm25_vectors(nodes)
        colbert_vecs = build_colbert_embeddings(nodes)

        if "__token_vocabulary__" in bm25_vecs:
            _save_token_vocabulary(
                target_collection_name, bm25_vecs["__token_vocabulary__"]
            )

        count = upsert_hybrid_points(
            client,
            target_collection_name,
            nodes,
            dense_vecs,
            bm25_vecs,
            colbert_vecs,
        )

        if alias_name and recreate_collection:
            _ = _activate_collection_alias(client, alias_name, target_collection_name)
            if (
                previous_collection_name is not None
                and previous_collection_name != target_collection_name
                and client.collection_exists(previous_collection_name)
            ):
                client.delete_collection(previous_collection_name)
                _delete_token_vocabulary(previous_collection_name)
    except Exception:
        if alias_name and recreate_collection:
            if client.collection_exists(target_collection_name):
                client.delete_collection(target_collection_name)
            if _get_vocab_path(target_collection_name).exists():
                _delete_token_vocabulary(target_collection_name)
        raise

    n_bm25_tokens = sum(
        len(scores)
        for node_id, scores in bm25_vecs.items()
        if node_id != "__token_vocabulary__"
    )
    n_colbert_toks = (
        sum(len(vecs) for vecs in colbert_vecs.values()) if colbert_vecs else 0
    )

    return count, n_bm25_tokens, n_colbert_toks
