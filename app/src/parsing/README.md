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
