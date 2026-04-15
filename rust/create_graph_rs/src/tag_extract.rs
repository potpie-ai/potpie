use std::fs;
use std::path::{Path, PathBuf};

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TagPayload {
    pub kind: String,
    pub tag_type: String,
    pub name: String,
    pub line: u32,
    pub end_line: u32,
}

pub fn extract_tags(relative_path: &str, text: &str) -> Vec<TagPayload> {
    if text.is_empty() {
        return Vec::new();
    }

    let Some(lang) = filename_to_lang(relative_path) else {
        return Vec::new();
    };

    let Some(language) = language_for_lang(lang) else {
        return fallback_refs(text);
    };

    let Some(query_source) = load_query(lang) else {
        return fallback_refs(text);
    };

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return fallback_refs(text);
    }

    let Some(tree) = parser.parse(text, None) else {
        return fallback_refs(text);
    };

    let Ok(query) = Query::new(&language, &query_source) else {
        return fallback_refs(text);
    };

    let mut cursor = QueryCursor::new();
    let mut captures: Vec<(tree_sitter::Node<'_>, String)> = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), text.as_bytes());
    while {
        matches.advance();
        matches.get().is_some()
    } {
        let Some(matched) = matches.get() else {
            continue;
        };
        for capture in matched.captures {
            let capture_name = query.capture_names()[capture.index as usize].to_string();
            captures.push((capture.node, capture_name));
        }
    }

    let mut tags = Vec::new();
    let mut saw_ref = false;
    let mut saw_def = false;

    for (node, capture_name) in captures {
        let (kind, tag_type) = if let Some(tag_type) = capture_name.strip_prefix("name.definition.")
        {
            saw_def = true;
            ("def", tag_type)
        } else if let Some(tag_type) = capture_name.strip_prefix("name.reference.") {
            saw_ref = true;
            ("ref", tag_type)
        } else {
            continue;
        };

        let mut node_text = match node.utf8_text(text.as_bytes()) {
            Ok(node_text) => node_text.to_string(),
            Err(_) => continue,
        };

        if lang == "java" && tag_type == "method" {
            if let Some(parent) = node.parent() {
                if parent.kind() == "method_invocation" {
                    if let Some(object_node) = parent.child_by_field_name("object") {
                        if let Ok(object_text) = object_node.utf8_text(text.as_bytes()) {
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
            line: node.start_position().row as u32,
            end_line: node.end_position().row as u32,
        });
    }

    if saw_ref || !saw_def {
        return tags;
    }

    tags.extend(fallback_refs(text));
    tags
}

fn filename_to_lang(path: &str) -> Option<&'static str> {
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())?
        .to_ascii_lowercase();

    match extension.as_str() {
        "py" => Some("python"),
        "js" | "jsx" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        "c" => Some("c"),
        "cs" => Some("c_sharp"),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some("cpp"),
        "h" => Some("c"),
        "el" => Some("elisp"),
        "ex" | "exs" => Some("elixir"),
        "elm" => Some("elm"),
        "go" => Some("go"),
        "java" => Some("java"),
        "ml" | "mli" => Some("ocaml"),
        "php" => Some("php"),
        "ql" => Some("ql"),
        "rb" => Some("ruby"),
        "rs" => Some("rust"),
        _ => None,
    }
}

fn language_for_lang(lang: &str) -> Option<Language> {
    match lang {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "c_sharp" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "elisp" => Some(tree_sitter_elisp::LANGUAGE.into()),
        "elixir" => Some(tree_sitter_elixir::LANGUAGE.into()),
        "elm" => Some(tree_sitter_elm::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "ocaml" => Some(tree_sitter_ocaml::LANGUAGE_OCAML.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "ql" => Some(tree_sitter_ql::LANGUAGE.into()),
        "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        _ => None,
    }
}

fn load_query(lang: &str) -> Option<String> {
    let query_path = query_dir().join(format!("tree-sitter-{lang}-tags.scm"));
    fs::read_to_string(query_path).ok()
}

fn query_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../app/modules/parsing/graph_construction/queries")
}

fn fallback_refs(text: &str) -> Vec<TagPayload> {
    let fallback_line = u32::MAX;
    let mut refs = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if current.is_empty() {
            if character == '_' || character.is_ascii_alphabetic() {
                current.push(character);
            }
            continue;
        }

        if character == '_' || character.is_ascii_alphanumeric() {
            current.push(character);
            continue;
        }

        refs.push(TagPayload {
            kind: "ref".to_string(),
            tag_type: "unknown".to_string(),
            name: std::mem::take(&mut current),
            line: fallback_line,
            end_line: fallback_line,
        });
    }

    if !current.is_empty() {
        refs.push(TagPayload {
            kind: "ref".to_string(),
            tag_type: "unknown".to_string(),
            name: current,
            line: fallback_line,
            end_line: fallback_line,
        });
    }

    refs
}
