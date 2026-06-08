use std::path::Path;

use parsing_rs::{
    build_bigram_index, build_file_index, build_workspace_index, search_content, search_files,
    ContentSearchResult, FffSearchError, FileSearchResult, WorkspaceIndex,
};

#[test]
fn build_file_index_searches_exact_prefix_substring_when_paths_exist() {
    let paths = vec![
        "src/auth.rs".to_string(),
        "src/authenticator.rs".to_string(),
        "README.md".to_string(),
    ];

    let index = build_file_index(paths);
    let results = search_files(&index, "auth", 10);
    let paths: Vec<_> = results.iter().map(|r| r.path.clone()).collect();

    assert_eq!(paths[0], "src/auth.rs");
    assert_eq!(results[0].score, 950);
    assert!(paths.contains(&"src/authenticator.rs".to_string()));
}

#[test]
fn file_search_is_deterministically_ranked() {
    let index = build_file_index(vec![
        "a/auth.txt".to_string(),
        "b/auth.log".to_string(),
        "auth/notes.md".to_string(),
        "src/bad/auth.rs".to_string(),
    ]);
    let results = search_files(&index, "auth", 4);

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].path, "a/auth.txt");
    assert_eq!(results[0].score, 950);
    assert_eq!(results[1].path, "b/auth.log");
    assert_eq!(results[1].score, 950);
    assert_eq!(results[2].path, "src/bad/auth.rs");
    assert_eq!(results[2].score, 950);
    assert_eq!(results[3].path, "auth/notes.md");
}

#[test]
fn workspace_index_reports_typed_error_when_root_is_missing() {
    let missing = Path::new("/tmp/does-not-exist-fff-search");
    let err = build_workspace_index(missing).expect_err("missing root should fail");
    assert!(matches!(
        err,
        FffSearchError::WorkspaceNotFound { path } if path == missing.to_string_lossy()
    ));
}

#[test]
fn file_search_tie_breaks_by_path() {
    let index = build_file_index(vec!["a/auth.txt".to_string(), "b/auth.log".to_string()]);
    let results = search_files(&index, "auth", 2);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].score, results[1].score);
    assert_eq!(results[0].path, "a/auth.txt");
    assert_eq!(results[1].path, "b/auth.log");
}

#[test]
fn search_files_matches_case_insensitively() {
    let index = build_file_index(vec![
        "Src/AuthService.RS".to_string(),
        "src/other.rs".to_string(),
    ]);

    let results = search_files(&index, "authservice", 10);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "Src/AuthService.RS");
}

#[test]
fn search_files_deduplicates_normalized_paths() {
    let index = build_file_index(vec![
        "src/auth.rs".to_string(),
        "src\\auth.rs".to_string(),
        "src/auth.rs".to_string(),
    ]);

    let results = search_files(&index, "auth", 10);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "src/auth.rs");
}

#[test]
fn search_files_returns_empty_when_query_or_limit_is_empty() {
    let index = build_file_index(vec!["src/main.rs".to_string(), "src/lib.rs".to_string()]);

    assert!(search_files(&index, "", 10).is_empty());
    assert!(search_files(&index, "src", 0).is_empty());
}

#[test]
fn build_bigram_index_searches_literal_content_when_lines_match() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/lib.rs".to_string(),
        content: "fn login() {}\nfn logout() {}".to_string(),
    }];
    let index = build_bigram_index(files);
    let results = search_content(&index, "login", 5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "src/lib.rs");
    assert_eq!(results[0].line, 1);
    assert_eq!(results[0].snippet, "fn login() {}");
    assert!(results[0].score > 1000);
}

#[test]
fn search_content_matches_case_insensitively() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/lib.rs".to_string(),
        content: "fn LoginHandler() {}\n".to_string(),
    }];
    let index = build_bigram_index(files);

    let results = search_content(&index, "loginhandler", 5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "src/lib.rs");
    assert_eq!(results[0].line, 1);
}

#[test]
fn search_content_matches_unicode_case_insensitively() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "docs/notes.md".to_string(),
        content: "CAFÉ_SEARCH lives here\n".to_string(),
    }];
    let index = build_bigram_index(files);

    let results = search_content(&index, "café_search", 5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].snippet, "CAFÉ_SEARCH lives here");
}

#[test]
fn search_content_uses_token_fallback_when_phrase_does_not_match() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/lib.rs".to_string(),
        content: "function handler routes login quickly\nother line".to_string(),
    }];
    let index = build_bigram_index(files);
    let results = search_content(&index, "login quickly handler", 5);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].line, 1);
    assert!(results[0].score > 700);
}

#[test]
fn search_content_treats_regex_chars_as_literal() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/test.txt".to_string(),
        content: "login.*test\n".to_string(),
    }];
    let index = build_bigram_index(files);

    assert_eq!(search_content(&index, "login.*", 5).len(), 1);
    assert_eq!(search_content(&index, "login.test", 5).len(), 0);
}

#[test]
fn search_content_returns_empty_when_query_or_limit_is_empty() {
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/lib.rs".to_string(),
        content: "hello world".to_string(),
    }];
    let index = build_bigram_index(files);

    assert!(search_content(&index, "", 10).is_empty());
    assert!(search_content(&index, "hello", 0).is_empty());
}

#[test]
fn search_content_snippet_is_trimmed_to_200_chars() {
    let long_line = format!("{}{}", "a".repeat(260), " end");
    let files = vec![parsing_rs::IndexedFileContent {
        path: "src/lib.rs".to_string(),
        content: format!("{long_line}\nsecond\n"),
    }];

    let index = build_bigram_index(files);
    let results = search_content(&index, "a", 1);

    assert_eq!(results[0].snippet.len(), 200);
}

#[test]
fn _contract_types_are_visible() {
    let _build_file_idx = build_file_index(Vec::<String>::new());
    let _build_bigram_idx = build_bigram_index(Vec::<parsing_rs::IndexedFileContent>::new());
    let _search_file_result = FileSearchResult {
        path: String::new(),
        score: 0,
    };
    let _search_content_result = ContentSearchResult {
        path: String::new(),
        line: 0,
        snippet: String::new(),
        score: 0,
    };

    let _workspace_index_type: Option<WorkspaceIndex> = None;
}

#[test]
fn generated_path_index_returns_stable_ranked_results() {
    let paths = (0..10_000)
        .map(|idx| {
            if idx == 9_999 {
                format!("src/{idx:05}-auth.rs")
            } else {
                format!("src/file-{idx:05}.txt")
            }
        })
        .collect::<Vec<_>>();
    let index = build_file_index(paths);

    let results = search_files(&index, "auth", 10);

    assert!(!results.is_empty());
    assert_eq!(results[0].path, "src/09999-auth.rs");
    assert_eq!(results[0].score, 600);
    assert!(results.len() <= 10);
}

#[test]
fn generated_content_search_returns_only_expected_matches() {
    let files = (0..1_000)
        .map(|idx| parsing_rs::IndexedFileContent {
            path: format!("src/{idx:04}.rs"),
            content: if idx % 10 == 0 {
                "login token handler\n".to_string()
            } else {
                format!("unrelated line {idx}\n")
            },
        })
        .collect::<Vec<_>>();

    let index = build_bigram_index(files);
    let results = search_content(&index, "login handler", 20);

    assert_eq!(results.len(), 20);
    assert_eq!(results[0].line, 1);
    assert_eq!(results[0].path, "src/0000.rs");
    assert!(results.iter().all(|r| r.path.ends_with(".rs")));
}
