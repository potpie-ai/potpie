use crate::text_filter::{is_text_file, read_text};
use crate::FilePayload;
use std::io;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

pub fn discover_files(repo_dir: &str) -> io::Result<Vec<FilePayload>> {
    let repo_path = Path::new(repo_dir);
    let mut payloads = Vec::new();

    let walker = WalkDir::new(repo_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| should_descend(repo_path, entry));

    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        if !is_text_file(path) {
            continue;
        }

        let relative_path = relative_path(repo_path, path)?;
        let text = read_text(path);
        let is_binary = !text.is_empty() && text.chars().any(|character| character == '\0');

        payloads.push(FilePayload {
            relative_path,
            text,
            is_binary,
        });
    }

    Ok(payloads)
}

fn should_descend(repo_path: &Path, entry: &DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }

    if !entry.file_type().is_dir() {
        return true;
    }

    let Ok(relative) = entry.path().strip_prefix(repo_path) else {
        return false;
    };

    let parts: Vec<String> = path_parts(relative);

    if parts.iter().any(|part| part == ".git") {
        return false;
    }

    !parts
        .iter()
        .any(|part| part.starts_with('.') && part != ".github" && part != ".vscode")
}

fn relative_path(repo_path: &Path, path: &Path) -> io::Result<String> {
    let relative = path.strip_prefix(repo_path).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "failed to compute relative path for {}: {error}",
                path.display()
            ),
        )
    })?;

    Ok(path_to_string(relative))
}

fn path_parts(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}

fn path_to_string(path: &Path) -> String {
    let path_buf = PathBuf::from(path);
    path_buf.to_string_lossy().replace('\\', "/")
}
