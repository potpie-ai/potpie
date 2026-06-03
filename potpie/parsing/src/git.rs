use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use regex::Regex;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;
use url::Url;
use wait_timeout::ChildExt;

const TOKEN_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'+')
    .add(b'/')
    .add(b':')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

#[derive(Debug)]
pub enum GitError {
    RepositoryNotFound,
    RefNotFound,
    MalformedTreeEntry { line: String, reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[pyo3::pyclass]
pub struct FileEntry {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dir: String,
    #[pyo3(get)]
    pub ext: String,
    #[pyo3(get)]
    pub depth: u16,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub sha: String,
}

pub fn token_username_prefix(token: &str) -> &'static str {
    if token.starts_with("ghs_") {
        "x-access-token"
    } else {
        "oauth2"
    }
}

pub fn build_authenticated_url(repo_url: &str, auth_token: Option<&str>) -> String {
    let Some(raw_token) = auth_token else {
        return repo_url.to_string();
    };

    let token = raw_token.trim();
    if token.is_empty() {
        return repo_url.to_string();
    }

    let Ok(parsed) = Url::parse(repo_url) else {
        return repo_url.to_string();
    };

    if !matches!(parsed.scheme(), "http" | "https") {
        return repo_url.to_string();
    }

    let Some(host) = parsed.host_str() else {
        return repo_url.to_string();
    };

    let encoded_token = utf8_percent_encode(token, TOKEN_ENCODE_SET).to_string();
    let username_prefix = token_username_prefix(token);
    let authority = match parsed.port() {
        Some(port) => format!("{username_prefix}:{encoded_token}@{host}:{port}"),
        None => format!("{username_prefix}:{encoded_token}@{host}"),
    };

    let mut rebuilt = format!("{}://{}{}", parsed.scheme(), authority, parsed.path());
    if let Some(query) = parsed.query() {
        rebuilt.push('?');
        rebuilt.push_str(query);
    }
    if let Some(fragment) = parsed.fragment() {
        rebuilt.push('#');
        rebuilt.push_str(fragment);
    }
    rebuilt
}

pub fn sanitize_git_message(message: &str) -> String {
    Regex::new(r"://[^@/]+@")
        .expect("credential sanitization regex should compile")
        .replace_all(message, "://***@")
        .into_owned()
}

pub fn validate_ref(git_ref: &str) -> bool {
    !git_ref.is_empty()
        && !git_ref.contains('\n')
        && !git_ref.contains('\r')
        && !git_ref.contains("..")
}

pub fn is_bare_repo(path: &Path) -> bool {
    path.exists()
        && path.join("HEAD").is_file()
        && path.join("objects").is_dir()
        && path.join("refs").exists()
}

fn strip_url_credentials(url: &str) -> String {
    let Ok(mut parsed) = Url::parse(url) else {
        return url.to_string();
    };

    if parsed.has_authority() {
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
    }

    parsed.to_string()
}

fn remote_origin_matches_repo_url(dest_path: &str, repo_url: &str) -> Result<bool, GitError> {
    let remote_args = vec![
        "-C".to_string(),
        dest_path.to_string(),
        "remote".to_string(),
        "get-url".to_string(),
        "origin".to_string(),
    ];
    let remote_output = run_git(remote_args, 30)?;
    if !remote_output.status.success() {
        let _ = sanitize_git_message(&String::from_utf8_lossy(&remote_output.stderr));
        return Ok(false);
    }

    let remote_url = String::from_utf8_lossy(&remote_output.stdout)
        .trim()
        .to_string();

    Ok(strip_url_credentials(&remote_url) == strip_url_credentials(repo_url))
}

pub fn run_git<C, I>(args: I, timeout_secs: u64) -> Result<Output, GitError>
where
    C: AsRef<OsStr>,
    I: IntoIterator<Item = C>,
{
    let mut child = Command::new("git")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| GitError::RepositoryNotFound)?;

    match child
        .wait_timeout(Duration::from_secs(timeout_secs))
        .map_err(|_| GitError::RepositoryNotFound)?
    {
        Some(_) => child
            .wait_with_output()
            .map_err(|_| GitError::RepositoryNotFound),
        None => {
            let _ = child.kill();
            let _ = child.wait();
            Err(GitError::RepositoryNotFound)
        }
    }
}

pub fn parse_ls_tree_record(record: &[u8]) -> Result<FileEntry, GitError> {
    let record_for_error = String::from_utf8_lossy(record).to_string();
    let tab_index = record
        .iter()
        .position(|byte| *byte == b'\t')
        .ok_or_else(|| GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: "missing tab separator".to_string(),
        })?;

    let metadata =
        std::str::from_utf8(&record[..tab_index]).map_err(|_| GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: "metadata is not valid utf-8".to_string(),
        })?;
    let raw_path = &record[tab_index + 1..];

    let mut parts = metadata.split_whitespace();
    let mode = parts.next().ok_or_else(|| GitError::MalformedTreeEntry {
        line: record_for_error.clone(),
        reason: "missing mode".to_string(),
    })?;
    let entry_type = parts.next().ok_or_else(|| GitError::MalformedTreeEntry {
        line: record_for_error.clone(),
        reason: "missing type".to_string(),
    })?;
    let sha = parts.next().ok_or_else(|| GitError::MalformedTreeEntry {
        line: record_for_error.clone(),
        reason: "missing sha".to_string(),
    })?;

    if parts.next().is_some() {
        return Err(GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: "unexpected metadata fields".to_string(),
        });
    }

    let path = String::from_utf8_lossy(raw_path).to_string();
    if path.is_empty() {
        return Err(GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: "missing path".to_string(),
        });
    }

    let kind = if mode == "160000" || entry_type == "commit" {
        "submodule"
    } else if entry_type == "blob" {
        "file"
    } else if entry_type == "tree" {
        "directory"
    } else {
        return Err(GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: format!("unsupported tree type '{entry_type}'"),
        });
    };

    let path_buf = PathBuf::from(&path);
    let name = path_buf
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| GitError::MalformedTreeEntry {
            line: record_for_error.clone(),
            reason: "path basename is not valid utf-8".to_string(),
        })?
        .to_string();

    let dir = path_buf
        .parent()
        .and_then(|parent| parent.to_str())
        .filter(|parent| !parent.is_empty() && *parent != ".")
        .unwrap_or("")
        .to_string();

    let ext = if kind == "file" {
        path_buf
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };

    let depth = path.matches('/').count() as u16;

    Ok(FileEntry {
        path,
        name,
        dir,
        ext,
        depth,
        kind: kind.to_string(),
        sha: sha.to_string(),
    })
}

pub fn parse_ls_tree_line(line: &str) -> Result<FileEntry, GitError> {
    parse_ls_tree_record(line.as_bytes())
}

pub fn bare_clone(
    repo_url: &str,
    dest_path: &str,
    git_ref: &str,
    auth_token: Option<&str>,
) -> Result<(), GitError> {
    if !validate_ref(git_ref) {
        return Err(GitError::RefNotFound);
    }

    let dest = Path::new(dest_path);
    if is_bare_repo(dest) {
        if !remote_origin_matches_repo_url(dest_path, repo_url)? {
            fs::remove_dir_all(dest).map_err(|_| GitError::RepositoryNotFound)?;
        } else {
            let fetch_args = vec![
                "-C".to_string(),
                dest_path.to_string(),
                "fetch".to_string(),
                "origin".to_string(),
                "--".to_string(),
                git_ref.to_string(),
            ];
            let fetch_output = run_git(fetch_args, 300)?;
            if !fetch_output.status.success() {
                let _ = sanitize_git_message(&String::from_utf8_lossy(&fetch_output.stderr));
                return Err(GitError::RefNotFound);
            }

            let verify_ref = format!("{git_ref}^{{commit}}");
            let verify_args = vec![
                "-C".to_string(),
                dest_path.to_string(),
                "rev-parse".to_string(),
                "--verify".to_string(),
                verify_ref,
            ];
            let verify_output = run_git(verify_args, 30)?;
            if !verify_output.status.success() {
                let _ = sanitize_git_message(&String::from_utf8_lossy(&verify_output.stderr));
                return Err(GitError::RefNotFound);
            }

            return Ok(());
        }
    }

    if dest.exists() {
        return Err(GitError::RepositoryNotFound);
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|_| GitError::RepositoryNotFound)?;
    }

    let clone_url = build_authenticated_url(repo_url, auth_token);
    let mut clone_started = false;

    let result = (|| {
        let clone_args = vec![
            "clone".to_string(),
            "--bare".to_string(),
            "--filter=blob:none".to_string(),
            "--".to_string(),
            clone_url,
            dest_path.to_string(),
        ];
        clone_started = true;
        let clone_output = run_git(clone_args, 600)?;
        if !clone_output.status.success() {
            let _ = sanitize_git_message(&String::from_utf8_lossy(&clone_output.stderr));
            return Err(GitError::RepositoryNotFound);
        }

        let fetch_args = vec![
            "-C".to_string(),
            dest_path.to_string(),
            "fetch".to_string(),
            "origin".to_string(),
            "--".to_string(),
            git_ref.to_string(),
        ];
        let fetch_output = run_git(fetch_args, 300)?;
        if !fetch_output.status.success() {
            let _ = sanitize_git_message(&String::from_utf8_lossy(&fetch_output.stderr));
            return Err(GitError::RefNotFound);
        }

        let verify_ref = format!("{git_ref}^{{commit}}");
        let verify_args = vec![
            "-C".to_string(),
            dest_path.to_string(),
            "rev-parse".to_string(),
            "--verify".to_string(),
            verify_ref,
        ];
        let verify_output = run_git(verify_args, 30)?;
        if !verify_output.status.success() {
            let _ = sanitize_git_message(&String::from_utf8_lossy(&verify_output.stderr));
            return Err(GitError::RefNotFound);
        }

        Ok(())
    })();

    if result.is_err() && clone_started {
        let _ = fs::remove_dir_all(dest);
    }

    result
}

pub fn list_files(bare_repo_path: &str, git_ref: &str) -> Result<Vec<FileEntry>, GitError> {
    let repo_path = Path::new(bare_repo_path);
    if !is_bare_repo(repo_path) {
        return Err(GitError::RepositoryNotFound);
    }

    if !validate_ref(git_ref) {
        return Err(GitError::RefNotFound);
    }

    let verify_ref = format!("{git_ref}^{{commit}}");
    let verify_args = vec![
        "-C".to_string(),
        bare_repo_path.to_string(),
        "rev-parse".to_string(),
        "--verify".to_string(),
        verify_ref,
    ];
    let verify_output = run_git(verify_args, 30)?;
    if !verify_output.status.success() {
        let _ = sanitize_git_message(&String::from_utf8_lossy(&verify_output.stderr));
        return Err(GitError::RefNotFound);
    }

    let list_args = vec![
        "-C".to_string(),
        bare_repo_path.to_string(),
        "ls-tree".to_string(),
        "-r".to_string(),
        "-t".to_string(),
        "-z".to_string(),
        git_ref.to_string(),
    ];
    let output = run_git(list_args, 60)?;
    if !output.status.success() {
        let _ = sanitize_git_message(&String::from_utf8_lossy(&output.stderr));
        return Err(GitError::RefNotFound);
    }

    output
        .stdout
        .split(|byte| *byte == b'\0')
        .filter(|record| !record.is_empty())
        .map(parse_ls_tree_record)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use tempfile::{tempdir, TempDir};
    use url::Url;

    fn run_git_in(repo_path: &Path, args: &[&str]) -> std::process::Output {
        Command::new("git")
            .current_dir(repo_path)
            .args(args)
            .output()
            .expect("git command should execute")
    }

    fn assert_git_ok(repo_path: &Path, args: &[&str]) {
        let output = run_git_in(repo_path, args);
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn git_stdout(repo_path: &Path, args: &[&str]) -> String {
        let output = run_git_in(repo_path, args);
        assert!(
            output.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("git stdout should be utf-8")
            .trim()
            .to_string()
    }

    fn create_repo_with_files(files: &[(&str, &str)]) -> (TempDir, String) {
        let temp_dir = tempdir().expect("should create temp dir");
        let repo_path = temp_dir.path();

        assert_git_ok(repo_path, &["init"]);
        assert_git_ok(repo_path, &["config", "user.name", "Potpie Tests"]);
        assert_git_ok(
            repo_path,
            &["config", "user.email", "potpie-tests@example.com"],
        );

        for (relative_path, contents) in files {
            let file_path = repo_path.join(relative_path);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).expect("should create fixture directories");
            }
            fs::write(&file_path, contents).expect("should write fixture file");
        }

        assert_git_ok(repo_path, &["add", "."]);
        assert_git_ok(repo_path, &["commit", "-m", "fixture"]);

        let head_sha = git_stdout(repo_path, &["rev-parse", "HEAD"]);
        (temp_dir, head_sha)
    }

    fn clone_fixture_repo(files: &[(&str, &str)]) -> (TempDir, TempDir, String) {
        let (source_repo, head_sha) = create_repo_with_files(files);
        let clone_root = tempdir().expect("should create clone temp dir");
        let bare_path = clone_root.path().join("repo.git");
        let repo_url = Url::from_directory_path(source_repo.path())
            .expect("fixture repo path should convert to file url")
            .to_string();

        bare_clone(
            &repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
            None,
        )
        .expect("bare clone should succeed");

        (source_repo, clone_root, head_sha)
    }

    #[test]
    fn test_token_username_prefix_ghs() {
        assert_eq!(token_username_prefix("ghs_abc123"), "x-access-token");
    }

    #[test]
    fn test_token_username_prefix_ghp() {
        assert_eq!(token_username_prefix("ghp_abc123"), "oauth2");
    }

    #[test]
    fn test_token_username_prefix_gho() {
        assert_eq!(token_username_prefix("gho_abc123"), "oauth2");
    }

    #[test]
    fn test_token_username_prefix_none() {
        assert_eq!(token_username_prefix(""), "oauth2");
    }

    #[test]
    fn test_build_auth_url_no_token() {
        let url = "https://github.com/owner/repo.git";
        assert_eq!(build_authenticated_url(url, None), url);
    }

    #[test]
    fn test_build_auth_url_ghs_token() {
        assert_eq!(
            build_authenticated_url("https://github.com/owner/repo.git", Some("ghs_abc123")),
            "https://x-access-token:ghs_abc123@github.com/owner/repo.git"
        );
    }

    #[test]
    fn test_build_auth_url_strips_existing_creds() {
        assert_eq!(
            build_authenticated_url(
                "https://old:pass@github.com/owner/repo.git",
                Some("ghp_abc123")
            ),
            "https://oauth2:ghp_abc123@github.com/owner/repo.git"
        );
    }

    #[test]
    fn test_build_auth_url_preserves_port() {
        assert_eq!(
            build_authenticated_url("https://github.com:8443/owner/repo.git", Some("ghp_abc123")),
            "https://oauth2:ghp_abc123@github.com:8443/owner/repo.git"
        );
    }

    #[test]
    fn test_build_auth_url_special_chars_encoded() {
        assert_eq!(
            build_authenticated_url("https://github.com/owner/repo.git", Some("tok+/=")),
            "https://oauth2:tok%2B%2F%3D@github.com/owner/repo.git"
        );
    }

    #[test]
    fn test_validate_ref_valid() {
        assert!(validate_ref("main"));
        assert!(validate_ref("v1.0.0"));
        assert!(validate_ref("abc123"));
        assert!(validate_ref("refs/heads/main"));
    }

    #[test]
    fn test_validate_ref_invalid_empty() {
        assert!(!validate_ref(""));
    }

    #[test]
    fn test_validate_ref_invalid_newline() {
        assert!(!validate_ref("main\nother"));
    }

    #[test]
    fn test_validate_ref_invalid_dotdot() {
        assert!(!validate_ref("refs/heads/../main"));
    }

    #[test]
    fn test_sanitize_strips_credential_patterns() {
        assert_eq!(
            sanitize_git_message("fatal: https://user:token@github.com/owner/repo.git"),
            "fatal: https://***@github.com/owner/repo.git"
        );
    }

    #[test]
    fn test_bare_clone_success() {
        let (source_repo, _) = create_repo_with_files(&[("README.md", "hello\n")]);
        let clone_root = tempdir().expect("should create clone temp dir");
        let bare_path = clone_root.path().join("repo.git");
        let repo_url = Url::from_directory_path(source_repo.path())
            .expect("fixture repo path should convert to file url")
            .to_string();

        bare_clone(
            &repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
            None,
        )
        .expect("bare clone should succeed");

        assert!(
            bare_path.join("HEAD").exists(),
            "bare repo should contain HEAD"
        );
    }

    #[test]
    fn test_bare_clone_recreates_cache_when_origin_differs() {
        let (first_repo, _) = create_repo_with_files(&[("README.md", "first\n")]);
        let (second_repo, _) = create_repo_with_files(&[("README.md", "second\n")]);
        let clone_root = tempdir().expect("should create clone temp dir");
        let bare_path = clone_root.path().join("repo.git");
        let first_repo_url = Url::from_directory_path(first_repo.path())
            .expect("fixture repo path should convert to file url")
            .to_string();
        let second_repo_url = Url::from_directory_path(second_repo.path())
            .expect("fixture repo path should convert to file url")
            .to_string();

        bare_clone(
            &first_repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
            None,
        )
        .expect("initial bare clone should succeed");

        bare_clone(
            &second_repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
            None,
        )
        .expect("bare clone should recreate cache for changed origin");

        assert_eq!(
            git_stdout(&bare_path, &["remote", "get-url", "origin"]),
            second_repo_url
        );
        assert_eq!(
            git_stdout(&bare_path, &["show", "HEAD:README.md"]),
            "second"
        );
    }

    #[test]
    fn test_bare_clone_ref_not_found() {
        let (source_repo, _) = create_repo_with_files(&[("README.md", "hello\n")]);
        let clone_root = tempdir().expect("should create clone temp dir");
        let bare_path = clone_root.path().join("repo.git");
        let repo_url = Url::from_directory_path(source_repo.path())
            .expect("fixture repo path should convert to file url")
            .to_string();

        let error = bare_clone(
            &repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "missing-ref",
            None,
        )
        .expect_err("clone should fail for missing ref");

        assert!(matches!(error, GitError::RefNotFound));
        assert!(!bare_path.exists(), "failed clone should be cleaned up");
    }

    #[test]
    fn test_bare_clone_nonexistent_repo() {
        let clone_root = tempdir().expect("should create clone temp dir");
        let bare_path = clone_root.path().join("repo.git");
        let missing_repo_url = format!(
            "file://{}",
            clone_root.path().join("missing").to_string_lossy()
        );

        let error = bare_clone(
            &missing_repo_url,
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
            None,
        )
        .expect_err("clone should fail for nonexistent repo");

        assert!(matches!(error, GitError::RepositoryNotFound));
    }

    #[test]
    fn test_list_files_basic() {
        let (source_repo, clone_root, _head_sha) = clone_fixture_repo(&[
            ("README.md", "hello\n"),
            ("src/lib.rs", "pub fn demo() {}\n"),
        ]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        let readme = entries
            .iter()
            .find(|entry| entry.path == "README.md")
            .expect("README.md entry should exist");
        assert_eq!(readme.kind, "file");

        let src_file = entries
            .iter()
            .find(|entry| entry.path == "src/lib.rs")
            .expect("src/lib.rs entry should exist");
        let expected_sha = git_stdout(source_repo.path(), &["rev-parse", "HEAD:src/lib.rs"]);
        assert_eq!(src_file.kind, "file");
        assert_eq!(src_file.sha, expected_sha);
    }

    #[test]
    fn test_list_files_includes_directories() {
        let (_source_repo, clone_root, _) =
            clone_fixture_repo(&[("src/nested/lib.rs", "pub fn demo() {}\n")]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        assert!(entries
            .iter()
            .any(|entry| entry.path == "src" && entry.kind == "directory"));
        assert!(entries
            .iter()
            .any(|entry| entry.path == "src/nested" && entry.kind == "directory"));
    }

    #[test]
    fn test_list_files_root_file() {
        let (_source_repo, clone_root, _) = clone_fixture_repo(&[("README.md", "hello\n")]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        let entry = entries
            .iter()
            .find(|entry| entry.path == "README.md")
            .expect("README.md entry should exist");
        assert_eq!(entry.dir, "");
        assert_eq!(entry.depth, 0);
        assert_eq!(entry.ext, "md");
    }

    #[test]
    fn test_list_files_extensionless_file() {
        let (_source_repo, clone_root, _) = clone_fixture_repo(&[("Makefile", "all:\n\ttrue\n")]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        let entry = entries
            .iter()
            .find(|entry| entry.path == "Makefile")
            .expect("Makefile entry should exist");
        assert_eq!(entry.ext, "");
    }

    #[test]
    fn test_list_files_depth_calculation() {
        let (_source_repo, clone_root, _) = clone_fixture_repo(&[("a/b/c.txt", "hi\n")]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        let file_entry = entries
            .iter()
            .find(|entry| entry.path == "a/b/c.txt")
            .expect("nested file entry should exist");
        let dir_entry = entries
            .iter()
            .find(|entry| entry.path == "a/b")
            .expect("nested directory entry should exist");
        assert_eq!(file_entry.depth, 2);
        assert_eq!(dir_entry.depth, 1);
    }

    #[test]
    fn test_list_files_preserves_newline_in_filename() {
        let filename = "dir/file\nwith-newline.rs";
        let (_source_repo, clone_root, _) = clone_fixture_repo(&[(filename, "hi\n")]);
        let bare_path = clone_root.path().join("repo.git");
        let entries = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "HEAD",
        )
        .expect("list_files should succeed");

        let entry = entries
            .iter()
            .find(|entry| entry.path == filename)
            .expect("newline-containing file entry should exist");
        assert_eq!(entry.name, "file\nwith-newline.rs");
        assert_eq!(entry.dir, "dir");
        assert_eq!(entry.ext, "rs");
    }

    #[test]
    fn test_parse_ls_tree_record_accepts_non_utf8_path_bytes() {
        let entry = parse_ls_tree_record(
            b"100644 blob 0123456789abcdef0123456789abcdef01234567\tdir/file-\xff.rs",
        )
        .expect("non-utf8 path bytes should not break tree parsing");

        assert_eq!(entry.path, "dir/file-�.rs");
        assert_eq!(entry.name, "file-�.rs");
        assert_eq!(entry.dir, "dir");
        assert_eq!(entry.ext, "rs");
    }

    #[test]
    fn test_list_files_submodule() {
        let entry = parse_ls_tree_line(
            "160000 commit 0123456789abcdef0123456789abcdef01234567\tvendor/module",
        )
        .expect("submodule entry should parse");

        assert_eq!(entry.kind, "submodule");
        assert_eq!(entry.path, "vendor/module");
        assert_eq!(entry.name, "module");
        assert_eq!(entry.dir, "vendor");
        assert_eq!(entry.ext, "");
    }

    #[test]
    fn test_list_files_invalid_bare_path() {
        let temp_dir = tempdir().expect("should create temp dir");
        let missing = temp_dir.path().join("missing.git");

        let error = list_files(
            missing.to_str().expect("missing path should be utf-8"),
            "HEAD",
        )
        .expect_err("list_files should fail for missing bare repo");

        assert!(matches!(error, GitError::RepositoryNotFound));
    }

    #[test]
    fn test_list_files_invalid_ref() {
        let (_source_repo, clone_root, _) = clone_fixture_repo(&[("README.md", "hello\n")]);
        let bare_path = clone_root.path().join("repo.git");

        let error = list_files(
            bare_path.to_str().expect("bare path should be utf-8"),
            "missing-ref",
        )
        .expect_err("list_files should fail for missing ref");

        assert!(matches!(error, GitError::RefNotFound));
    }

    #[test]
    fn test_parse_malformed_line() {
        let error =
            parse_ls_tree_line("garbage input").expect_err("parser should reject malformed lines");

        match error {
            GitError::MalformedTreeEntry { line, reason } => {
                assert_eq!(line, "garbage input");
                assert!(reason.contains("tab"));
            }
            other => panic!("expected malformed tree entry error, got {other:?}"),
        }
    }
}
