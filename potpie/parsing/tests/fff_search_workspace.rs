use std::fs;
use std::path::Path;
use tempfile::tempdir;

use parsing_rs::{build_workspace_index, FffSearchError};

fn write_files(root: &Path, files: &[(&str, &[u8])]) {
    for (relative, data) in files {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directories should be creatable");
        }
        fs::write(path, data).expect("fixture file should be written");
    }
}

#[test]
fn workspace_index_indexes_checked_out_folder() {
    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();

    write_files(
        workspace,
        &[
            (".gitignore", b"*.tmp\n"),
            ("src/auth.rs", b"fn login_handler() {}\n"),
            ("src/main.rs", b"fn main() {}\n"),
            ("ignored.tmp", b"skip me\n"),
        ],
    );

    let index = build_workspace_index(workspace).expect("workspace index should be built");
    let file_results = index.search_files("auth", 10);
    assert_eq!(index.file_count(), 3);
    assert!(file_results
        .iter()
        .any(|result| result.path == "src/auth.rs"));
    assert_eq!(file_results[0].path, "src/auth.rs");
    assert_eq!(index.content_file_count(), 2);
}

#[test]
fn workspace_index_ignores_git_directory_and_ignored_entries() {
    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();

    write_files(
        workspace,
        &[
            (".git/HEAD", b"ref: refs/heads/main\n"),
            ("notes.md", b"visible note\n"),
            (".gitignore", b"ignored.md\n"),
            ("ignored.md", b"hidden note\n"),
        ],
    );

    let index = build_workspace_index(workspace).expect("workspace index should be built");
    let paths: Vec<String> = index
        .search_files("visible", 10)
        .into_iter()
        .map(|r| r.path)
        .collect();

    assert!(!paths.iter().any(|path| path.contains(".git")));
    assert!(!paths.iter().any(|path| path == "ignored.md"));
    assert_eq!(index.file_count(), 2);
}

#[test]
fn workspace_index_includes_file_but_not_content_when_binary_or_large() {
    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();

    write_files(
        workspace,
        &[
            ("src/logo.png", &[0, 1, 2, 3]),
            ("src/data.bin", &[0, 159, 0, 255]),
            ("src/notes.txt", b"tiny notes\n"),
            ("src/huge.txt", &vec![b'a'; 1_048_586][..]),
        ],
    );

    let index = build_workspace_index(workspace).expect("workspace index should be built");
    let logo_results = index.search_files("logo", 10);
    let bin_results = index.search_files("data", 10);
    assert!(logo_results
        .iter()
        .any(|result| result.path == "src/logo.png"));
    assert!(bin_results
        .iter()
        .any(|result| result.path == "src/data.bin"));
    assert_eq!(index.content_file_count(), 1);
    let content_paths: Vec<String> = index
        .search_content("notes", 10)
        .into_iter()
        .map(|result| result.path)
        .collect();
    assert!(content_paths.contains(&"src/notes.txt".to_string()));
    assert!(!content_paths.contains(&"src/huge.txt".to_string()));
}

#[test]
fn workspace_index_indexes_unicode_text_content() {
    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();

    write_files(
        workspace,
        &[("docs/unicode.md", "CAFÉ_SEARCH 東京\n".as_bytes())],
    );

    let index = build_workspace_index(workspace).expect("workspace index should be built");
    let results = index.search_content("café_search", 10);

    assert_eq!(index.content_file_count(), 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, "docs/unicode.md");
    assert_eq!(results[0].snippet, "CAFÉ_SEARCH 東京");
}

#[cfg(unix)]
#[test]
fn workspace_index_does_not_follow_symlinks() {
    use std::os::unix::fs::symlink;

    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();
    write_files(
        workspace,
        &[
            ("real/target.txt", b"needle in real file\n"),
            ("visible.txt", b"visible\n"),
        ],
    );
    symlink(
        workspace.join("real/target.txt"),
        workspace.join("linked-target.txt"),
    )
    .expect("symlink should be created");

    let index = build_workspace_index(workspace).expect("workspace index should be built");
    let file_paths: Vec<String> = index
        .search_files("target", 10)
        .into_iter()
        .map(|result| result.path)
        .collect();

    assert!(file_paths.contains(&"real/target.txt".to_string()));
    assert!(!file_paths.contains(&"linked-target.txt".to_string()));
    assert_eq!(index.search_content("needle", 10).len(), 1);
}

#[cfg(unix)]
#[test]
fn workspace_index_keeps_unreadable_files_searchable_by_path_only() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();
    write_files(
        workspace,
        &[
            ("src/readable.txt", b"visible content\n"),
            ("src/locked.txt", b"hidden content\n"),
        ],
    );
    let locked_path = workspace.join("src/locked.txt");
    let mut permissions = fs::metadata(&locked_path)
        .expect("locked file metadata should be readable")
        .permissions();
    permissions.set_mode(0o000);
    fs::set_permissions(&locked_path, permissions).expect("locked file should become unreadable");

    let index = build_workspace_index(workspace).expect("workspace index should be built");

    let mut restored = fs::metadata(&locked_path)
        .expect("locked file metadata should remain readable")
        .permissions();
    restored.set_mode(0o644);
    fs::set_permissions(&locked_path, restored)
        .expect("locked file permissions should be restored");

    assert!(index
        .search_files("locked", 10)
        .iter()
        .any(|result| result.path == "src/locked.txt"));
    assert!(index.search_content("hidden", 10).is_empty());
    assert_eq!(index.search_content("visible", 10).len(), 1);
}

#[test]
fn workspace_index_rebuilds_and_refreshes_file_list() {
    let temp_dir = tempdir().expect("temporary folder should be created");
    let workspace = temp_dir.path();

    write_files(workspace, &[("a.txt", b"first\n")]);
    let initial =
        build_workspace_index(workspace).expect("initial workspace index should be built");
    assert_eq!(initial.file_count(), 1);

    write_files(workspace, &[("b.txt", b"second\n")]);
    let rebuilt =
        build_workspace_index(workspace).expect("rebuilt workspace index should be built");
    assert_eq!(rebuilt.file_count(), 2);
    assert!(rebuilt.file_count() != initial.file_count());
}

#[test]
fn workspace_index_reports_not_found_or_directory_errors() {
    let temp_root = temp_dir();
    let missing = temp_root.path().join("no-such-workspace");
    let missing_root = missing;
    let missing_err =
        build_workspace_index(&missing_root).expect_err("missing workspace should fail");
    assert!(matches!(
        missing_err,
        FffSearchError::WorkspaceNotFound { .. }
    ));

    let file_fixture = tempdir().expect("temporary folder should be created");
    let file_path = file_fixture.path().join("file.txt");
    fs::write(&file_path, b"not a dir\n").expect("file should be written");
    let not_dir_err = build_workspace_index(&file_path).expect_err("file root should fail");
    assert!(matches!(
        not_dir_err,
        FffSearchError::WorkspaceNotDirectory { .. }
    ));
}

fn temp_dir() -> tempfile::TempDir {
    tempdir().expect("temporary folder should be created")
}
