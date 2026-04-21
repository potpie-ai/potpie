mod code_index;
mod parse;
mod tag_extract;

pub use code_index::{create_code_indexes, get_text_files, process_files_parallel, CodeFile};
pub use parse::{index, process_file};
pub use tag_extract::{extract_graph, extract_tags, GraphPayload, NodePayload, RelationshipPayload, TagPayload};

#[pyo3::prelude::pymodule]
mod parsing_rs {
    use pyo3::prelude::*;

    #[pyo3::pyclass]
    #[derive(Clone, Debug, PartialEq)]
    pub struct TagPayload {
        #[pyo3(get, set)]
        pub kind: String,
        #[pyo3(get, set)]
        pub tag_type: String,
        #[pyo3(get, set)]
        pub name: String,
        #[pyo3(get, set)]
        pub line: u32,
        #[pyo3(get, set)]
        pub end_line: u32,
    }

    #[pyo3::pyclass]
    #[derive(Clone, Debug, PartialEq)]
    pub struct NodePayload {
        #[pyo3(get, set)]
        pub id: String,
        #[pyo3(get, set)]
        pub node_type: String,
        #[pyo3(get, set)]
        pub file: String,
        #[pyo3(get, set)]
        pub line: u32,
        #[pyo3(get, set)]
        pub end_line: u32,
        #[pyo3(get, set)]
        pub name: String,
        #[pyo3(get, set)]
        pub class_name: Option<String>,
        #[pyo3(get, set)]
        pub text: Option<String>,
    }

    #[pyo3::pyclass]
    #[derive(Clone, Debug, PartialEq)]
    pub struct RelationshipPayload {
        #[pyo3(get, set)]
        pub source_id: String,
        #[pyo3(get, set)]
        pub target_id: String,
        #[pyo3(get, set)]
        pub relationship_type: String,
        #[pyo3(get, set)]
        pub ident: Option<String>,
        #[pyo3(get, set)]
        pub ref_line: Option<u32>,
        #[pyo3(get, set)]
        pub end_ref_line: Option<u32>,
    }

    #[pyo3::pyclass]
    #[derive(Clone, Debug, PartialEq)]
    pub struct GraphPayload {
        #[pyo3(get, set)]
        pub nodes: Vec<NodePayload>,
        #[pyo3(get, set)]
        pub relationships: Vec<RelationshipPayload>,
    }

    use crate::tag_extract::extract_graph as extract_graph_rs;

    #[pyfunction]
    fn extract_graph(repo_dir: &str) -> GraphPayload {
        let internal = extract_graph_rs(repo_dir);
        let nodes = internal.nodes.into_iter().map(|n| NodePayload {
            id: n.id,
            node_type: n.node_type,
            file: n.file,
            line: n.line,
            end_line: n.end_line,
            name: n.name,
            class_name: n.class_name,
            text: n.text,
        }).collect();
        let relationships = internal.relationships.into_iter().map(|r| RelationshipPayload {
            source_id: r.source_id,
            target_id: r.target_id,
            relationship_type: r.relationship_type,
            ident: r.ident,
            ref_line: r.ref_line,
            end_ref_line: r.end_ref_line,
        }).collect();
        GraphPayload { nodes, relationships }
    }
}
