"""Compatibility import for the former sqlite_hybrid profile.

Existing callers now use Qdrant for vectors and retain their SQLite text and
embedding cache. No SQLite extension is loaded.
"""

from .qdrant_hybrid import EMBED_BATCH, QdrantHybridResourceIndex, pack_vector

SqliteHybridResourceIndex = QdrantHybridResourceIndex

__all__ = ["EMBED_BATCH", "SqliteHybridResourceIndex", "pack_vector"]
