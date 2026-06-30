# Parsing

Extract code graphs from repositories.

## Rust

```rust
use parsing_rs::{extract_graph, GraphPayload, NodePayload, RelationshipPayload};

let graph: GraphPayload = extract_graph("/path/to/repo");

// Nodes: FILE, CLASS, INTERFACE, FUNCTION
for node in graph.nodes {
    println!("{}: {} @ {}:{}", node.node_type, node.name, node.file, node.line);
}

// Relationships: CONTAINS, REFERENCES
for rel in graph.relationships {
    println!("{} → {}", rel.source_id, rel.target_id);
}
```

## Python

```python
from parsing.py_graph import create_graph
import networkx as nx

G = create_graph("/path/to/repo")  # Returns NetworkX MultiDiGraph
```

## Node Types

- `FILE` - Source file
- `CLASS` / `INTERFACE` - Type definitions
- `FUNCTION` - Methods and functions

## Relationship Types

- `CONTAINS` - FILE → CLASS/FUNCTION
- `REFERENCES` - FUNCTION → FUNCTION/CLASS (calls, usage)

## Supported Languages

Python, Rust, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP, C#, Elixir, OCaml, Elisp, Elm, QL

## Build

```bash
cd app/src/parsing && maturin develop
```

## Note

This crate creates **graphs only**. Embeddings and search are handled by the parent `feat/colbert` module (uses Qdrant + Neo4j).

## FFF Workspace Search (Lexical)

The FFF search surface indexes a checked-out workspace folder on demand and provides in-memory path/content search.

```rust
use parsing_rs::{build_workspace_index, search_files};

let index = build_workspace_index("/path/to/workspace").expect("workspace should be indexed");
let files = index.search_files("auth", 5);

assert!(!files.is_empty());
```

```python
import parsing_rs

index = parsing_rs.build_workspace_index("/path/to/workspace")
print(index.file_count(), index.content_file_count())
print([(r.path, r.score) for r in index.search_files("auth", 5)])
```

This is lexical, in-memory search only. It intentionally does not do semantic matching.
Pre-sandbox `list_files` behavior (`potpie/parsing/src/git.rs`) is unchanged.
