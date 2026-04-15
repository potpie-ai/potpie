use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::PySet;

use crate::tag_extract::extract_tags;
use crate::traversal;
use crate::{EdgePayload, GraphPayload, NodePayload};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReferenceRecord {
    source: String,
    line: u32,
    end_line: u32,
    current_class: Option<String>,
    current_method: Option<String>,
}

pub fn build_graph_payload(repo_dir: &str) -> PyResult<GraphPayload> {
    let files = traversal::discover_files(repo_dir)
        .map_err(|error| PyOSError::new_err(error.to_string()))?;
    let mut graph = GraphPayload {
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    let mut node_types = HashMap::new();
    let mut defines: HashMap<String, HashSet<String>> = HashMap::new();
    let mut references: HashMap<String, HashSet<ReferenceRecord>> = HashMap::new();
    let mut seen_relationships: HashSet<(String, String, String)> = HashSet::new();

    for file in files {
        // Skip binary files entirely - do not create a FILE node or extract tags
        if file.is_binary {
            continue;
        }

        let file_node_name = file.relative_path.clone();
        let file_display_name = file
            .relative_path
            .rsplit('/')
            .next()
            .unwrap_or(file.relative_path.as_str())
            .to_string();

        add_node(
            &mut graph.nodes,
            &mut node_types,
            NodePayload {
                id: file_node_name.clone(),
                node_type: "FILE".to_string(),
                file: file.relative_path.clone(),
                line: 0,
                end_line: 0,
                name: file_display_name,
                class_name: None,
                text: Some(file.text.clone()),
            },
        );

        let mut current_class: Option<String> = None;
        let mut current_method: Option<String> = None;

        for tag in extract_tags(&file.relative_path, &file.text) {
            if tag.kind == "def" {
                let node_type = match tag.tag_type.as_str() {
                    "class" => {
                        current_class = Some(tag.name.clone());
                        current_method = None;
                        "CLASS"
                    }
                    "interface" => {
                        current_class = Some(tag.name.clone());
                        current_method = None;
                        "INTERFACE"
                    }
                    "method" | "function" => {
                        current_method = Some(tag.name.clone());
                        "FUNCTION"
                    }
                    _ => continue,
                }
                .to_string();

                let node_name = if let Some(class_name) = current_class.as_ref() {
                    format!("{}:{}.{}", file.relative_path, class_name, tag.name)
                } else {
                    format!("{}:{}", file.relative_path, tag.name)
                };

                let inserted = add_node(
                    &mut graph.nodes,
                    &mut node_types,
                    NodePayload {
                        id: node_name.clone(),
                        node_type: node_type.clone(),
                        file: file.relative_path.clone(),
                        line: tag.line,
                        end_line: tag.end_line,
                        name: tag.name.clone(),
                        class_name: current_class.clone(),
                        text: None,
                    },
                );

                if inserted {
                    let rel_key = (
                        file_node_name.clone(),
                        node_name.clone(),
                        "CONTAINS".to_string(),
                    );
                    if !seen_relationships.contains(&rel_key) {
                        graph.edges.push(EdgePayload {
                            source_id: file_node_name.clone(),
                            target_id: node_name.clone(),
                            edge_type: "CONTAINS".to_string(),
                            ident: Some(tag.name.clone()),
                            ref_line: None,
                            end_ref_line: None,
                        });
                        seen_relationships.insert(rel_key);
                    }
                }

                defines.entry(tag.name).or_default().insert(node_name);
            } else if tag.kind == "ref" {
                let source = match (current_class.as_ref(), current_method.as_ref()) {
                    (Some(class_name), Some(method_name)) => {
                        format!("{}:{}.{}", file.relative_path, class_name, method_name)
                    }
                    (None, Some(method_name)) => format!("{}:{}", file.relative_path, method_name),
                    _ => file.relative_path.clone(),
                };

                references
                    .entry(tag.name)
                    .or_default()
                    .insert(ReferenceRecord {
                        source,
                        line: tag.line,
                        end_line: tag.end_line,
                        current_class: current_class.clone(),
                        current_method: current_method.clone(),
                    });
            }
        }
    }

    for (ident, refs) in references {
        let Some(target_nodes) = defines.get(&ident) else {
            continue;
        };

        for reference in refs_in_python_order(refs)? {
            for target in target_nodes {
                if reference.source == *target {
                    continue;
                }

                if node_types.contains_key(&reference.source) && node_types.contains_key(target) {
                    create_relationship(
                        &mut graph.edges,
                        &node_types,
                        &reference.source,
                        target,
                        "REFERENCES",
                        &mut seen_relationships,
                        Some(ident.clone()),
                        Some(reference.line),
                        Some(reference.end_line),
                    );
                }
            }
        }
    }

    Ok(graph)
}

fn add_node(
    nodes: &mut Vec<NodePayload>,
    node_types: &mut HashMap<String, String>,
    node: NodePayload,
) -> bool {
    if node_types.contains_key(&node.id) {
        return false;
    }

    node_types.insert(node.id.clone(), node.node_type.clone());
    nodes.push(node);
    true
}

fn create_relationship(
    edges: &mut Vec<EdgePayload>,
    node_types: &HashMap<String, String>,
    source: &str,
    target: &str,
    relationship_type: &str,
    seen_relationships: &mut HashSet<(String, String, String)>,
    ident: Option<String>,
    ref_line: Option<u32>,
    end_ref_line: Option<u32>,
) -> bool {
    if source == target {
        return false;
    }

    let Some(source_type) = node_types.get(source) else {
        return false;
    };
    let Some(target_type) = node_types.get(target) else {
        return false;
    };

    let rel_key = (
        source.to_string(),
        target.to_string(),
        relationship_type.to_string(),
    );
    let reverse_key = (
        target.to_string(),
        source.to_string(),
        relationship_type.to_string(),
    );

    if seen_relationships.contains(&rel_key) || seen_relationships.contains(&reverse_key) {
        return false;
    }

    let mut valid_direction = false;

    if relationship_type == "REFERENCES" {
        if source_type == "FUNCTION" && target_type == "FUNCTION" && source.contains("Impl") {
            valid_direction = true;
        } else if source_type == "FUNCTION" {
            valid_direction = true;
        } else if target_type == "CLASS" {
            valid_direction = true;
        }
    }

    if !valid_direction {
        return false;
    }

    edges.push(EdgePayload {
        source_id: source.to_string(),
        target_id: target.to_string(),
        edge_type: relationship_type.to_string(),
        ident,
        ref_line,
        end_ref_line,
    });
    seen_relationships.insert(rel_key);
    true
}

fn refs_in_python_order(refs: HashSet<ReferenceRecord>) -> PyResult<Vec<ReferenceRecord>> {
    Python::with_gil(|py| {
        let py_set = PySet::empty_bound(py)?;

        for reference in refs {
            py_set.add((
                reference.source,
                payload_line_to_python_line(reference.line),
                payload_line_to_python_line(reference.end_line),
                reference.current_class,
                reference.current_method,
            ))?;
        }

        let mut ordered_refs = Vec::new();
        for item in py_set.iter() {
            let (source, line, end_line, current_class, current_method): (
                String,
                i64,
                i64,
                Option<String>,
                Option<String>,
            ) = item.extract()?;
            ordered_refs.push(ReferenceRecord {
                source,
                line: python_line_to_payload_line(line),
                end_line: python_line_to_payload_line(end_line),
                current_class,
                current_method,
            });
        }

        Ok(ordered_refs)
    })
}

fn payload_line_to_python_line(line: u32) -> i64 {
    if line == u32::MAX {
        -1
    } else {
        i64::from(line)
    }
}

fn python_line_to_payload_line(line: i64) -> u32 {
    if line < 0 {
        u32::MAX
    } else {
        line as u32
    }
}
