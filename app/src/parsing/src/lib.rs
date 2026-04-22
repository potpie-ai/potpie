mod code_index;
mod parse;
mod tag_extract;

pub use code_index::{create_code_indexes, get_text_files, process_files_parallel, CodeFile};
pub use parse::{index, process_file};
pub use tag_extract::{extract_graph, extract_tags};

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
        #[pyo3(get, set)]
        pub byte_start: usize,
        #[pyo3(get, set)]
        pub byte_end: usize,
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
        #[pyo3(get)]
        pub nodes: Vec<NodePayload>,
        #[pyo3(get)]
        pub relationships: Vec<RelationshipPayload>,
    }

    impl From<crate::tag_extract::TagPayload> for TagPayload {
        fn from(other: crate::tag_extract::TagPayload) -> Self {
            TagPayload {
                kind: other.kind,
                tag_type: other.tag_type,
                name: other.name,
                line: other.line,
                end_line: other.end_line,
                byte_start: other.byte_start,
                byte_end: other.byte_end,
            }
        }
    }

    impl From<crate::tag_extract::NodePayload> for NodePayload {
        fn from(other: crate::tag_extract::NodePayload) -> Self {
            NodePayload {
                id: other.id,
                node_type: other.node_type,
                file: other.file,
                line: other.line,
                end_line: other.end_line,
                name: other.name,
                class_name: other.class_name,
                text: other.text,
            }
        }
    }

    impl From<crate::tag_extract::RelationshipPayload> for RelationshipPayload {
        fn from(other: crate::tag_extract::RelationshipPayload) -> Self {
            RelationshipPayload {
                source_id: other.source_id,
                target_id: other.target_id,
                relationship_type: other.relationship_type,
                ident: other.ident,
                ref_line: other.ref_line,
                end_ref_line: other.end_ref_line,
            }
        }
    }

    impl From<crate::tag_extract::GraphPayload> for GraphPayload {
        fn from(other: crate::tag_extract::GraphPayload) -> Self {
            GraphPayload {
                nodes: other.nodes.into_iter().map(NodePayload::from).collect(),
                relationships: other.relationships.into_iter().map(RelationshipPayload::from).collect(),
            }
        }
    }

    use crate::tag_extract::extract_graph as extract_graph_rs;

    #[pyfunction]
    fn extract_graph(repo_dir: &str) -> GraphPayload {
        let internal = extract_graph_rs(repo_dir);
        GraphPayload::from(internal)
    }
}
