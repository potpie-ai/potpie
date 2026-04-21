use std::collections::{HashMap, HashSet};
use std::path::Path;

use rayon::prelude::*;
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use crate::code_index::{create_code_indexes, CodeFile};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TagPayload {
    pub kind: String,
    pub tag_type: String,
    pub name: String,
    pub line: u32,
    pub end_line: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodePayload {
    pub id: String,
    pub node_type: String,
    pub file: String,
    pub line: u32,
    pub end_line: u32,
    pub name: String,
    pub class_name: Option<String>,
    pub text: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationshipPayload {
    pub source_id: String,
    pub target_id: String,
    pub relationship_type: String,
    pub ident: Option<String>,
    pub ref_line: Option<u32>,
    pub end_ref_line: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphPayload {
    pub nodes: Vec<NodePayload>,
    pub relationships: Vec<RelationshipPayload>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReferenceRecord {
    source_id: String,
    line: u32,
    end_line: u32,
    current_class: Option<String>,
    current_method: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileGraphData {
    nodes: Vec<NodePayload>,
    relationships: Vec<RelationshipPayload>,
    definitions: Vec<(String, String)>,
    references: Vec<(String, ReferenceRecord)>,
}

pub fn extract_graph(repo_dir: &str) -> GraphPayload {
    let file_graphs = create_code_indexes(repo_dir)
        .into_par_iter()
        .with_min_len(16)
        .map(extract_file_graph)
        .collect::<Vec<_>>();

    let mut nodes = Vec::new();
    let mut relationships = Vec::new();
    let mut defines: HashMap<String, HashSet<String>> = HashMap::new();
    let mut references: HashMap<String, HashSet<ReferenceRecord>> = HashMap::new();
    let mut node_types: HashMap<String, String> = HashMap::new();
    let mut seen_relationships: HashSet<(String, String, String)> = HashSet::new();

    for file_graph in file_graphs {
        for node in file_graph.nodes {
            node_types.insert(node.id.clone(), node.node_type.clone());
            nodes.push(node);
        }

        for relationship in file_graph.relationships {
            seen_relationships.insert((
                relationship.source_id.clone(),
                relationship.target_id.clone(),
                relationship.relationship_type.clone(),
            ));
            relationships.push(relationship);
        }

        for (ident, node_id) in file_graph.definitions {
            defines.entry(ident).or_default().insert(node_id);
        }

        for (ident, reference) in file_graph.references {
            references.entry(ident).or_default().insert(reference);
        }
    }

    for (ident, refs) in references {
        let Some(target_nodes) = defines.get(&ident) else {
            continue;
        };

        for reference in refs {
            for target_id in target_nodes {
                if reference.source_id == *target_id {
                    continue;
                }

                let Some(source_type) = node_types.get(&reference.source_id) else {
                    continue;
                };
                let Some(target_type) = node_types.get(target_id) else {
                    continue;
                };

                if !is_valid_reference_direction(&reference.source_id, source_type, target_type) {
                    continue;
                }

                let relationship_key = (
                    reference.source_id.clone(),
                    target_id.clone(),
                    "REFERENCES".to_string(),
                );
                let reverse_key = (
                    target_id.clone(),
                    reference.source_id.clone(),
                    "REFERENCES".to_string(),
                );

                if seen_relationships.contains(&relationship_key)
                    || seen_relationships.contains(&reverse_key)
                {
                    continue;
                }

                seen_relationships.insert(relationship_key);
                relationships.push(RelationshipPayload {
                    source_id: reference.source_id.clone(),
                    target_id: target_id.clone(),
                    relationship_type: "REFERENCES".to_string(),
                    ident: Some(ident.clone()),
                    ref_line: Some(reference.line),
                    end_ref_line: Some(reference.end_line),
                });
            }
        }
    }

    GraphPayload { nodes, relationships }
}

pub fn extract_tags(relative_path: &str, text: &str) -> Vec<TagPayload> {
    if text.is_empty() {
        return Vec::new();
    }

    let Some(lang) = filename_to_lang(relative_path) else {
        return Vec::new();
    };

    let Some(language) = language_for_lang(lang) else {
        return Vec::new();
    };

    let Some(query_source) = load_query(lang) else {
        return Vec::new();
    };

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }

    let Some(tree) = parser.parse(text, None) else {
        return Vec::new();
    };

    let Ok(query) = Query::new(&language, &query_source) else {
        return Vec::new();
    };

    let mut cursor = QueryCursor::new();
    let mut tags = Vec::new();
    let bytes = text.as_bytes();
    let mut matches = cursor.matches(&query, tree.root_node(), bytes);

    while let Some(query_match) = matches.next() {
        for capture in query_match.captures {
            let Some(capture_name) = query.capture_names().get(capture.index as usize) else {
                continue;
            };

            let (kind, tag_type) = if let Some(tag_type) = capture_name.strip_prefix("name.definition.") {
                ("def", tag_type)
            } else if let Some(tag_type) = capture_name.strip_prefix("name.reference.") {
                ("ref", tag_type)
            } else {
                continue;
            };

            let mut node_text = match capture.node.utf8_text(bytes) {
                Ok(node_text) => node_text.to_string(),
                Err(_) => continue,
            };

            if lang == "java" && tag_type == "method" {
                if let Some(parent) = capture.node.parent() {
                    if parent.kind() == "method_invocation" {
                        if let Some(object_node) = parent.child_by_field_name("object") {
                            if let Ok(object_text) = object_node.utf8_text(bytes) {
                                node_text = format!("{object_text}.{node_text}");
                            }
                        }
                    }
                }
            }

            tags.push(TagPayload {
                kind: kind.to_string(),
                tag_type: tag_type.to_string(),
                name: node_text,
                line: capture.node.start_position().row as u32,
                end_line: capture.node.end_position().row as u32,
            });
        }
    }

    tags
}

fn extract_file_graph(file: CodeFile) -> FileGraphData {
    let mut nodes = Vec::new();
    let mut relationships = Vec::new();
    let mut definitions = Vec::new();
    let mut references = Vec::new();
    let mut seen_ids = HashSet::new();
    let file_name = file
        .relative_path
        .rsplit('/')
        .next()
        .unwrap_or(file.relative_path.as_str())
        .to_string();

    let file_node = NodePayload {
        id: file.relative_path.clone(),
        node_type: "FILE".to_string(),
        file: file.relative_path.clone(),
        line: 0,
        end_line: 0,
        name: file_name,
        class_name: None,
        text: Some(file.text.clone()),
    };
    seen_ids.insert(file_node.id.clone());
    nodes.push(file_node);

    let mut current_class: Option<String> = None;
    let mut current_method: Option<String> = None;
    let file_node_id = file.relative_path.clone();

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
            };

            let node_id = if let Some(class_name) = current_class.as_ref() {
                format!("{}:{}.{}", file.relative_path, class_name, tag.name)
            } else {
                format!("{}:{}", file.relative_path, tag.name)
            };

            if seen_ids.insert(node_id.clone()) {
                nodes.push(NodePayload {
                    id: node_id.clone(),
                    node_type: node_type.to_string(),
                    file: file.relative_path.clone(),
                    line: tag.line,
                    end_line: tag.end_line,
                    name: tag.name.clone(),
                    class_name: current_class.clone(),
                    text: None,
                });

                relationships.push(RelationshipPayload {
                    source_id: file_node_id.clone(),
                    target_id: node_id.clone(),
                    relationship_type: "CONTAINS".to_string(),
                    ident: Some(tag.name.clone()),
                    ref_line: None,
                    end_ref_line: None,
                });
            }

            definitions.push((tag.name, node_id));
        } else if tag.kind == "ref" {
            let source_id = if let (Some(class_name), Some(method_name)) =
                (current_class.as_ref(), current_method.as_ref())
            {
                format!("{}:{}.{}", file.relative_path, class_name, method_name)
            } else if let Some(method_name) = current_method.as_ref() {
                format!("{}:{}", file.relative_path, method_name)
            } else {
                file.relative_path.clone()
            };

            references.push((
                tag.name,
                ReferenceRecord {
                    source_id,
                    line: tag.line,
                    end_line: tag.end_line,
                    current_class: current_class.clone(),
                    current_method: current_method.clone(),
                },
            ));
        }
    }

    FileGraphData {
        nodes,
        relationships,
        definitions,
        references,
    }
}

fn is_valid_reference_direction(source_id: &str, source_type: &str, target_type: &str) -> bool {
    if source_type == "FUNCTION" && target_type == "FUNCTION" && source_id.contains("Impl") {
        return true;
    }

    if source_type == "FUNCTION" {
        return true;
    }

    if target_type == "CLASS" {
        return true;
    }

    source_type == "FILE" && target_type == "FUNCTION"
}

fn filename_to_lang(path: &str) -> Option<&'static str> {
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();

    match extension.as_str() {
        "py" => Some("python"),
        "js" | "jsx" => Some("javascript"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "c" | "h" => Some("c"),
        "cs" => Some("c_sharp"),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some("cpp"),
        "ex" | "exs" => Some("elixir"),
        "go" => Some("go"),
        "java" => Some("java"),
        "ml" | "mli" => Some("ocaml"),
        "php" => Some("php"),
        "rb" => Some("ruby"),
        "rs" => Some("rust"),
        "el" => Some("elisp"),
        "elm" => Some("elm"),
        "ql" => Some("ql"),
        _ => None,
    }
}

fn language_for_lang(lang: &str) -> Option<Language> {
    match lang {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "c_sharp" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "elixir" => Some(tree_sitter_elixir::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "ocaml" => Some(tree_sitter_ocaml::LANGUAGE_OCAML.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "elisp" => Some(tree_sitter_elisp::LANGUAGE.into()),
        "elm" => Some(tree_sitter_elm::LANGUAGE.into()),
        "ql" => Some(tree_sitter_ql::LANGUAGE.into()),
        _ => None,
    }
}

fn load_query(lang: &str) -> Option<&'static str> {
    match lang {
        "python" => Some(include_str!("../parsing/queries/tree-sitter-python-tags.scm")),
        "javascript" => Some(include_str!("../parsing/queries/tree-sitter-javascript-tags.scm")),
        "typescript" | "tsx" => {
            Some(include_str!("../parsing/queries/tree-sitter-typescript-tags.scm"))
        }
        "c" => Some(include_str!("../parsing/queries/tree-sitter-c-tags.scm")),
        "c_sharp" => Some(include_str!("../parsing/queries/tree-sitter-c_sharp-tags.scm")),
        "cpp" => Some(include_str!("../parsing/queries/tree-sitter-cpp-tags.scm")),
        "elixir" => Some(include_str!("../parsing/queries/tree-sitter-elixir-tags.scm")),
        "go" => Some(include_str!("../parsing/queries/tree-sitter-go-tags.scm")),
        "java" => Some(include_str!("../parsing/queries/tree-sitter-java-tags.scm")),
        "ocaml" => Some(include_str!("../parsing/queries/tree-sitter-ocaml-tags.scm")),
        "php" => Some(include_str!("../parsing/queries/tree-sitter-php-tags.scm")),
        "ruby" => Some(include_str!("../parsing/queries/tree-sitter-ruby-tags.scm")),
        "rust" => Some(include_str!("../parsing/queries/tree-sitter-rust-tags.scm")),
        "elisp" => Some(include_str!("../parsing/queries/tree-sitter-elisp-tags.scm")),
        "elm" => Some(include_str!("../parsing/queries/tree-sitter-elm-tags.scm")),
        "ql" => Some(include_str!("../parsing/queries/tree-sitter-ql-tags.scm")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_graph() {
        let graph = extract_graph(".");
        assert!(!graph.nodes.is_empty(), "Should extract some nodes");
        assert!(
            !graph.relationships.is_empty(),
            "Should extract some relationships"
        );
    }
}
