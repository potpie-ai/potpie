use ignore::WalkBuilder;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

const KNOWN_BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp", "ico", "svg", "eps",
    "mp4", "avi", "mov", "wmv", "flv", "mkv", "webm", "mpeg", "mpg", "m4v",
    "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a",
    "zip", "tar", "gz", "bz2", "xz", "rar", "7z", "deb", "rpm", "dmg",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp",
    "exe", "dll", "so", "dylib", "a", "o", "obj", "bin",
    "ttf", "otf", "woff", "woff2", "eot", "db", "sqlite", "sqlite3", "mdb",
    "class", "pyc", "pyo", "pyd", "swp", "swo", "lock", "psd", "ai",
];

const BINARY_SAMPLE_SIZE: usize = 8192;
const NON_TEXT_THRESHOLD: f64 = 0.30;

#[derive(Debug, Clone)]
pub struct CodeFile {
    pub path: PathBuf,
    pub relative_path: String,
    pub extension: String,
    pub text: String,
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

    let non_newline = sample.iter().filter(|&&b| b != b'\n' && b != b'\r').count();
    if non_newline == 0 {
        return false;
    }

    let non_printable = sample
        .iter()
        .filter(|&&b| {
            !b.is_ascii_graphic()
                && !b.is_ascii_whitespace()
                && b != 0x1B
                && b != 0x09
                && b != 0x0A
                && b != 0x0D
        })
        .count();

    non_printable as f64 / non_newline as f64 > NON_TEXT_THRESHOLD
}

fn read_code_file(path: PathBuf, repo_path: &Path) -> Option<CodeFile> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_string();

    if !extension.is_empty() && is_known_binary_extension(&extension) {
        return None;
    }

    let bytes = fs::read(&path).ok()?;
    let sample = &bytes[..bytes.len().min(BINARY_SAMPLE_SIZE)];
    if sample_is_binary(sample) {
        return None;
    }

    let relative_path = path
        .strip_prefix(repo_path)
        .ok()?
        .to_string_lossy()
        .into_owned();

    Some(CodeFile {
        path,
        relative_path,
        extension,
        text: String::from_utf8_lossy(&bytes).into_owned(),
    })
}

pub fn create_code_indexes(repo_dir: &str) -> Vec<CodeFile> {
    let repo_path = Path::new(repo_dir);

    if !repo_path.exists() {
        log::error!("Repository path does not exist: {}", repo_dir);
        return Vec::new();
    }

    let paths: Vec<PathBuf> = WalkBuilder::new(repo_path)
        .hidden(false)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true)
        .follow_links(true)
        .build()
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().map(|file_type| file_type.is_file()).unwrap_or(false)
                && !entry.path().components().any(|component| component.as_os_str() == ".git")
        })
        .map(|entry| entry.into_path())
        .collect();

    let files: Vec<CodeFile> = paths
        .into_par_iter()
        .filter_map(|path| read_code_file(path, repo_path))
        .collect();

    log::info!("Found {} text files to index in {}", files.len(), repo_dir);
    files
}

pub fn get_text_files(repo_dir: &str) -> Vec<PathBuf> {
    create_code_indexes(repo_dir)
        .into_iter()
        .map(|file| file.path)
        .collect()
}

pub fn process_files_parallel<F, R>(repo_dir: &str, processor: F) -> Vec<R>
where
    F: Fn(CodeFile) -> R + Send + Sync,
    R: Send,
{
    create_code_indexes(repo_dir)
        .into_par_iter()
        .with_min_len(16)
        .map(processor)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_extensions() {
        assert!(is_known_binary_extension("png"));
        assert!(is_known_binary_extension("PNG"));
        assert!(is_known_binary_extension("jpg"));
        assert!(is_known_binary_extension("exe"));
        assert!(!is_known_binary_extension("py"));
        assert!(!is_known_binary_extension("rs"));
    }

    #[test]
    fn test_create_code_indexes() {
        let files = create_code_indexes(".");
        assert!(!files.is_empty(), "Should find some files");

        for file in &files {
            assert!(file.path.exists(), "File should exist: {:?}", file.path);
        }
    }
}
