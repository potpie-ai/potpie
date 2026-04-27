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
    pub byte_start: usize,
    pub byte_end: usize,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DefinitionMetadata {
    node_id: String,
    file: String,
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
    let mut defines: HashMap<String, Vec<DefinitionMetadata>> = HashMap::new();
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
            let file = node_id.split(':').next().unwrap_or(&node_id).to_string();
            defines.entry(ident).or_default().push(DefinitionMetadata {
                node_id,
                file,
            });
        }

        for (ident, reference) in file_graph.references {
            references.entry(ident).or_default().insert(reference);
        }
    }

    for (ident, refs) in references {
        // For method references, use the unqualified method name for lookup
        // since definitions are keyed by plain method names (e.g., "method" not "object.method")
        let lookup_ident = if ident.contains('.') {
            ident.split('.').last().unwrap_or(&ident)
        } else {
            &ident
        };
        let Some(target_nodes) = defines.get(lookup_ident) else {
            continue;
        };

        for reference in refs {
            // Extract source file from reference.source_id for scope filtering
            let source_file = reference.source_id.split(':').next().unwrap_or(&reference.source_id);

            // Separate same-file and cross-file targets
            // Prefer same-file matches but allow cross-file when no same-file target exists
            let mut same_file_targets: Vec<&DefinitionMetadata> = Vec::new();
            let mut cross_file_targets: Vec<&DefinitionMetadata> = Vec::new();

            for target_metadata in target_nodes {
                if reference.source_id == target_metadata.node_id {
                    continue;
                }

                if source_file == target_metadata.file {
                    same_file_targets.push(target_metadata);
                } else {
                    cross_file_targets.push(target_metadata);
                }
            }

            // Use same-file targets if available, otherwise fall back to cross-file
            let targets_to_use: Vec<&DefinitionMetadata> = if !same_file_targets.is_empty() {
                same_file_targets
            } else {
                cross_file_targets
            };

            for target_metadata in targets_to_use {
                let Some(source_type) = node_types.get(&reference.source_id) else {
                    continue;
                };
                let Some(target_type) = node_types.get(&target_metadata.node_id) else {
                    continue;
                };

                if !is_valid_reference_direction(&reference.source_id, source_type, target_type) {
                    continue;
                }

                let relationship_key = (
                    reference.source_id.clone(),
                    target_metadata.node_id.clone(),
                    "REFERENCES".to_string(),
                );
                let reverse_key = (
                    target_metadata.node_id.clone(),
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
                    target_id: target_metadata.node_id.clone(),
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
                byte_start: capture.node.start_byte(),
                byte_end: capture.node.end_byte(),
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

    let file_node_id = file.relative_path.clone();

    // Parse the file once to get the tree for proper scope containment
    let lang = match filename_to_lang(&file.relative_path) {
        Some(l) => l,
        None => return FileGraphData { nodes, relationships, definitions, references },
    };
    let language = match language_for_lang(lang) {
        Some(l) => l,
        None => return FileGraphData { nodes, relationships, definitions, references },
    };
    if load_query(lang).is_none() {
        return FileGraphData { nodes, relationships, definitions, references };
    }

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return FileGraphData { nodes, relationships, definitions, references };
    }
    let tree = match parser.parse(&file.text, None) {
        Some(t) => t,
        None => return FileGraphData { nodes, relationships, definitions, references },
    };
    let root_node = tree.root_node();
    let bytes = file.text.as_bytes();

    // Collect all definition tags with their byte positions for scope building
    let all_tags = extract_tags(&file.relative_path, &file.text);

    // Helper to find enclosing scope for a reference using AST containment
    let find_enclosing_scope = |ref_byte_pos: usize| -> (Option<String>, Option<String>) {
        // Use descendant_for_byte_range to find the node at this position
        // Then walk up to find enclosing class/method
        if let Some(node_at_pos) = root_node.descendant_for_byte_range(ref_byte_pos, ref_byte_pos) {
            let mut enclosing_class: Option<String> = None;
            let mut enclosing_method: Option<String> = None;

            let mut current: Option<tree_sitter::Node> = Some(node_at_pos);
            while let Some(node) = current {
                let kind = node.kind();
                match kind {
                    // Class definitions
                    "class_definition" | "class_declaration" | "class_specifier" => {
                        if enclosing_class.is_none() {
                            if let Some(name_node) = node.child_by_field_name("name") {
                                if let Ok(name) = name_node.utf8_text(bytes) {
                                    enclosing_class = Some(name.to_string());
                                }
                            }
                        }
                    }
                    // Interface definitions
                    "interface_declaration" => {
                        if enclosing_class.is_none() {
                            if let Some(name_node) = node.child_by_field_name("name") {
                                if let Ok(name) = name_node.utf8_text(bytes) {
                                    enclosing_class = Some(name.to_string());
                                }
                            }
                        }
                    }
                    // Method/function definitions
                    "method_declaration" | "function_definition" | "function_declaration"
                    | "method_definition" | "constructor_declaration" => {
                        if enclosing_method.is_none() {
                            if let Some(name_node) = node.child_by_field_name("name") {
                                if let Ok(name) = name_node.utf8_text(bytes) {
                                    enclosing_method = Some(name.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
                current = node.parent();
            }
            return (enclosing_class, enclosing_method);
        }
        (None, None)
    };

    // First pass: collect definitions using AST-based scope resolution
    for tag in &all_tags {
        if tag.kind == "def" {
            let enclosing_class = find_enclosing_scope(tag.byte_start).0;

            let (node_type, class_for_id) = match tag.tag_type.as_str() {
                "class" => {
                    // For class definitions, find_enclosing_scope returns the class
                    // itself (not an enclosing class) - use None to avoid self-reference
                    ("CLASS", None)
                }
                "interface" => {
                    // For interface definitions, similarly avoid self-reference
                    ("INTERFACE", None)
                }
                "method" | "function" => {
                    ("FUNCTION", enclosing_class)
                }
                _ => continue,
            };

            let node_id = match tag.tag_type.as_str() {
                "method" | "function" => {
                    if let Some(ref class_name) = class_for_id {
                        format!("{}:{}.{}", file.relative_path, class_name, tag.name)
                    } else {
                        format!("{}:{}", file.relative_path, tag.name)
                    }
                }
                _ => format!("{}:{}", file.relative_path, tag.name),
            };

            if seen_ids.insert(node_id.clone()) {
                nodes.push(NodePayload {
                    id: node_id.clone(),
                    node_type: node_type.to_string(),
                    file: file.relative_path.clone(),
                    line: tag.line,
                    end_line: tag.end_line,
                    name: tag.name.clone(),
                    class_name: class_for_id,
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

            definitions.push((tag.name.clone(), node_id));
        }
    }

    // Second pass: process references with proper AST containment
    for tag in &all_tags {
        if tag.kind == "ref" {
            // Use AST containment to find the correct enclosing scope
            let (enclosing_class, enclosing_method) = find_enclosing_scope(tag.byte_start);

            let source_id = if let (Some(class_name), Some(method_name)) =
                (enclosing_class.as_ref(), enclosing_method.as_ref())
            {
                format!("{}:{}.{}", file.relative_path, class_name, method_name)
            } else if let Some(method_name) = enclosing_method.as_ref() {
                format!("{}:{}", file.relative_path, method_name)
            } else {
                file.relative_path.clone()
            };

            references.push((
                tag.name.clone(),
                ReferenceRecord {
                    source_id,
                    line: tag.line,
                    end_line: tag.end_line,
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
    use std::collections::HashSet;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_extract_graph() {
        let temp_dir = tempdir().expect("should create temp repo");
        let file_path = temp_dir.path().join("main.py");

        fs::write(
            &file_path,
            "def foo():\n    return 1\n\ndef bar():\n    return foo()\n",
        )
        .expect("should write fixture source file");

        let graph = extract_graph(temp_dir.path().to_str().expect("temp path should be utf-8"));

        assert_eq!(graph.nodes.len(), 3, "expected file node and two function nodes");
        assert_eq!(
            graph.relationships.len(),
            3,
            "expected two CONTAINS edges and one REFERENCES edge"
        );

        let node_ids = graph
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(
            node_ids,
            HashSet::from(["main.py", "main.py:foo", "main.py:bar"])
        );

        assert!(graph.relationships.iter().any(|relationship| {
            relationship.source_id == "main.py"
                && relationship.target_id == "main.py:foo"
                && relationship.relationship_type == "CONTAINS"
        }));
        assert!(graph.relationships.iter().any(|relationship| {
            relationship.source_id == "main.py"
                && relationship.target_id == "main.py:bar"
                && relationship.relationship_type == "CONTAINS"
        }));
        assert!(graph.relationships.iter().any(|relationship| {
            relationship.source_id == "main.py:bar"
                && relationship.target_id == "main.py:foo"
                && relationship.relationship_type == "REFERENCES"
                && relationship.ident.as_deref() == Some("foo")
        }));
    }

    #[test]
    fn test_extract_graph_cross_file_references() {
        let temp_dir = tempdir().expect("should create temp repo");

        // Create two files: helper.py defines foo(), main.py calls foo() from helper
        let helper_path = temp_dir.path().join("helper.py");
        fs::write(
            &helper_path,
            "def foo():\n    return 1\n",
        )
        .expect("should write helper fixture source file");

        let main_path = temp_dir.path().join("main.py");
        fs::write(
            &main_path,
            "from helper import foo\n\ndef bar():\n    return foo()\n",
        )
        .expect("should write main fixture source file");

        let graph = extract_graph(temp_dir.path().to_str().expect("temp path should be utf-8"));

        // We expect: 1 file node for main.py, 1 file node for helper.py,
        // 1 function node for bar (in main.py), 1 function node for foo (in helper.py)
        // Plus CONTAINS edges for each file
        // Plus a cross-file REFERENCES edge from bar to foo
        let node_ids: HashSet<_> = graph
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect();

        assert!(
            node_ids.contains("main.py"),
            "expected main.py file node, got: {:?}",
            node_ids
        );
        assert!(
            node_ids.contains("helper.py"),
            "expected helper.py file node, got: {:?}",
            node_ids
        );
        assert!(
            node_ids.contains("main.py:bar"),
            "expected main.py:bar function node, got: {:?}",
            node_ids
        );
        assert!(
            node_ids.contains("helper.py:foo"),
            "expected helper.py:foo function node, got: {:?}",
            node_ids
        );

        // Check for CONTAINS relationships
        assert!(graph.relationships.iter().any(|relationship| {
            relationship.source_id == "main.py"
                && relationship.target_id == "main.py:bar"
                && relationship.relationship_type == "CONTAINS"
        }));
        assert!(graph.relationships.iter().any(|relationship| {
            relationship.source_id == "helper.py"
                && relationship.target_id == "helper.py:foo"
                && relationship.relationship_type == "CONTAINS"
        }));

        // Check for cross-file REFERENCES edge from bar to foo
        assert!(
            graph.relationships.iter().any(|relationship| {
                relationship.source_id == "main.py:bar"
                    && relationship.target_id == "helper.py:foo"
                    && relationship.relationship_type == "REFERENCES"
                    && relationship.ident.as_deref() == Some("foo")
            }),
            "expected cross-file REFERENCE from bar to foo, got: {:?}",
            graph
                .relationships
                .iter()
                .filter(|r| r.relationship_type == "REFERENCES")
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extract_graph_prefers_same_file_reference_when_duplicate_exists() {
        let temp_dir = tempdir().expect("should create temp repo");

        fs::write(
            temp_dir.path().join("helper.py"),
            "def foo():\n    return 2\n",
        )
        .expect("should write helper fixture source file");

        fs::write(
            temp_dir.path().join("main.py"),
            "def foo():\n    return 1\n\n\ndef bar():\n    return foo()\n",
        )
        .expect("should write main fixture source file");

        let graph = extract_graph(temp_dir.path().to_str().expect("temp path should be utf-8"));

        let references: Vec<_> = graph
            .relationships
            .iter()
            .filter(|relationship| relationship.relationship_type == "REFERENCES")
            .collect();

        assert!(
            references.iter().any(|relationship| {
                relationship.source_id == "main.py:bar"
                    && relationship.target_id == "main.py:foo"
                    && relationship.ident.as_deref() == Some("foo")
            }),
            "expected same-file reference to win when duplicate names exist, got: {:?}",
            references
        );
        assert!(
            !references.iter().any(|relationship| {
                relationship.source_id == "main.py:bar"
                    && relationship.target_id == "helper.py:foo"
                    && relationship.ident.as_deref() == Some("foo")
            }),
            "did not expect duplicate cross-file reference when same-file target exists, got: {:?}",
            references
        );
    }

    #[test]
    fn test_extract_graph_cross_file_references_across_directories() {
        let temp_dir = tempdir().expect("should create temp repo");
        let pkg_dir = temp_dir.path().join("pkg");
        let lib_dir = pkg_dir.join("lib");
        fs::create_dir_all(&lib_dir).expect("should create nested fixture directories");

        fs::write(
            pkg_dir.join("main.py"),
            "def local_only():\n    return 0\n\nfrom lib.helpers import foo\n\ndef bar():\n    return foo()\n",
        )
        .expect("should write nested main fixture source file");

        fs::write(
            lib_dir.join("helpers.py"),
            "def foo():\n    return 1\n",
        )
        .expect("should write nested helper fixture source file");

        let graph = extract_graph(temp_dir.path().to_str().expect("temp path should be utf-8"));

        let node_ids: HashSet<_> = graph
            .nodes
            .iter()
            .map(|node| node.id.as_str())
            .collect();
        assert!(node_ids.contains("pkg/main.py:bar"), "expected nested bar node, got: {:?}", node_ids);
        assert!(
            node_ids.contains("pkg/lib/helpers.py:foo"),
            "expected nested helper foo node, got: {:?}",
            node_ids
        );

        assert!(
            graph.relationships.iter().any(|relationship| {
                relationship.source_id == "pkg/main.py:bar"
                    && relationship.target_id == "pkg/lib/helpers.py:foo"
                    && relationship.relationship_type == "REFERENCES"
                    && relationship.ident.as_deref() == Some("foo")
            }),
            "expected cross-directory REFERENCE from bar to foo, got: {:?}",
            graph
                .relationships
                .iter()
                .filter(|relationship| relationship.relationship_type == "REFERENCES")
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extract_graph_cross_file_class_reference_from_function() {
        let temp_dir = tempdir().expect("should create temp repo");
        let models_dir = temp_dir.path().join("models");
        let services_dir = temp_dir.path().join("services");
        fs::create_dir_all(&models_dir).expect("should create models fixture directory");
        fs::create_dir_all(&services_dir).expect("should create services fixture directory");

        fs::write(
            models_dir.join("user.py"),
            "class User:\n    pass\n",
        )
        .expect("should write model fixture source file");

        fs::write(
            services_dir.join("factory.py"),
            "def build_user():\n    return User()\n",
        )
        .expect("should write service fixture source file");

        let graph = extract_graph(temp_dir.path().to_str().expect("temp path should be utf-8"));

        assert!(
            graph.relationships.iter().any(|relationship| {
                relationship.source_id == "services/factory.py:build_user"
                    && relationship.target_id == "models/user.py:User"
                    && relationship.relationship_type == "REFERENCES"
                    && relationship.ident.as_deref() == Some("User")
            }),
            "expected FUNCTION -> CLASS cross-file reference, got: {:?}",
            graph
                .relationships
                .iter()
                .filter(|relationship| relationship.relationship_type == "REFERENCES")
                .collect::<Vec<_>>()
        );
    }
}
