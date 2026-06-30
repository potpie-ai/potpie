use crate::fff_search::{
    build_content_index, build_file_index, ContentIndex, FffSearchError, FileIndex,
    IndexedFileContent,
};
use ignore::{gitignore::Gitignore, WalkBuilder};
use std::fs;
use std::path::Path;

const KNOWN_BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp", "ico", "svg", "eps", "mp4", "avi",
    "mov", "wmv", "flv", "mkv", "webm", "mpeg", "mpg", "m4v", "mp3", "wav", "ogg", "flac", "aac",
    "wma", "m4a", "zip", "tar", "gz", "bz2", "xz", "rar", "7z", "deb", "rpm", "dmg", "pdf", "doc",
    "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp", "exe", "dll", "so", "dylib", "a",
    "o", "obj", "bin", "ttf", "otf", "woff", "woff2", "eot", "db", "sqlite", "sqlite3", "mdb",
    "class", "pyc", "pyo", "pyd", "swp", "swo", "lock", "psd", "ai",
];

const NON_TEXT_THRESHOLD: f64 = 0.30;
const MAX_CONTENT_BYTES: usize = 1_048_576;

#[derive(Debug, Clone)]
pub struct WorkspaceIndex {
    file_index: FileIndex,
    content_index: ContentIndex,
}

impl WorkspaceIndex {
    pub fn file_count(&self) -> usize {
        self.file_index.entries().len()
    }

    pub fn content_file_count(&self) -> usize {
        self.content_index.files().len()
    }

    pub(crate) fn file_index(&self) -> &FileIndex {
        &self.file_index
    }

    pub(crate) fn content_index(&self) -> &ContentIndex {
        &self.content_index
    }

    pub fn search_files(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<crate::fff_search::FileSearchResult> {
        crate::fff_search::search_files(self.file_index(), query, limit)
    }

    pub fn search_content(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<crate::fff_search::ContentSearchResult> {
        crate::fff_search::search_content(self.content_index(), query, limit)
    }
}

pub fn build_workspace_index(root: &Path) -> Result<WorkspaceIndex, FffSearchError> {
    if !root.exists() {
        return Err(FffSearchError::WorkspaceNotFound {
            path: root.display().to_string(),
        });
    }
    if !root.is_dir() {
        return Err(FffSearchError::WorkspaceNotDirectory {
            path: root.display().to_string(),
        });
    }

    let file_paths = collect_workspace_files(root)?;

    let contents = file_paths
        .iter()
        .filter_map(|path| build_indexed_file_content(root, path))
        .collect::<Vec<_>>();

    let content_index = build_content_index(contents);
    let file_index = build_file_index(file_paths);

    Ok(WorkspaceIndex {
        file_index,
        content_index,
    })
}

fn collect_workspace_files(root: &Path) -> Result<Vec<String>, FffSearchError> {
    let gitignore = root.join(".gitignore");
    let gitignore = if gitignore.exists() {
        Some(ignore::gitignore::Gitignore::new(&gitignore).0)
    } else {
        None
    };

    let mut entries = WalkBuilder::new(root)
        .hidden(false)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .follow_links(false)
        .build()
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_type()
                .map(|file_type| file_type.is_file())
                .unwrap_or(false)
                && !entry
                    .path()
                    .components()
                    .any(|component| component.as_os_str() == ".git")
        })
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(root).ok()?.to_path_buf();
            if should_ignore_by_gitignore(&relative, &gitignore) {
                return None;
            }
            Some(relative)
        })
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .collect::<Vec<_>>();

    entries.sort_unstable();

    Ok(entries)
}

fn should_ignore_by_gitignore(path: &Path, ignore: &Option<Gitignore>) -> bool {
    ignore
        .as_ref()
        .is_some_and(|gitignore| gitignore.matched(path, false).is_ignore())
}

fn build_indexed_file_content(root: &Path, relative_path: &str) -> Option<IndexedFileContent> {
    let path = Path::new(relative_path);
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.'))
    {
        return None;
    }

    let absolute = root.join(relative_path);
    let metadata = fs::metadata(&absolute).ok()?;
    if metadata.len() as usize > MAX_CONTENT_BYTES {
        return None;
    }

    if is_known_binary_extension(
        absolute
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default(),
    ) {
        return None;
    }

    let bytes = fs::read(&absolute).ok()?;
    let sample = &bytes[..bytes.len().min(8192)];
    if sample_is_binary(sample) {
        return None;
    }

    let content = String::from_utf8_lossy(&bytes).into_owned();
    Some(IndexedFileContent {
        path: relative_path.to_string(),
        content,
    })
}

fn is_known_binary_extension(ext: &str) -> bool {
    KNOWN_BINARY_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

fn sample_is_binary(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    if std::str::from_utf8(sample).is_ok() {
        return false;
    }

    let non_newline = sample
        .iter()
        .filter(|&&byte| byte != b'\n' && byte != b'\r')
        .count();
    if non_newline == 0 {
        return false;
    }

    let non_printable = sample
        .iter()
        .filter(|&&byte| {
            !byte.is_ascii_graphic()
                && !byte.is_ascii_whitespace()
                && byte != 0x1B
                && byte != b'\t'
                && byte != b'\n'
                && byte != b'\r'
        })
        .count();

    (non_printable as f64 / non_newline as f64) > NON_TEXT_THRESHOLD
}
