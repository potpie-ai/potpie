use std::fmt;

#[derive(Debug, Clone)]
pub struct IndexedFileContent {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSearchResult {
    pub path: String,
    pub score: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentSearchResult {
    pub path: String,
    pub line: u32,
    pub snippet: String,
    pub score: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FffSearchError {
    WorkspaceNotFound { path: String },
    WorkspaceNotDirectory { path: String },
    PathOutsideWorkspace { path: String },
}

impl fmt::Display for FffSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceNotFound { path } => write!(formatter, "workspace not found: {path}"),
            Self::WorkspaceNotDirectory { path } => {
                write!(formatter, "workspace is not a directory: {path}")
            }
            Self::PathOutsideWorkspace { path } => {
                write!(formatter, "path outside workspace: {path}")
            }
        }
    }
}

impl std::error::Error for FffSearchError {}

#[derive(Debug, Clone)]
pub struct FileIndex {
    pub(crate) entries: Vec<String>,
}

impl FileIndex {
    pub fn entries(&self) -> &[String] {
        &self.entries
    }
}

#[derive(Debug, Clone)]
pub struct ContentIndex {
    pub(crate) files: Vec<IndexedFileContent>,
}

impl ContentIndex {
    pub fn files(&self) -> &[IndexedFileContent] {
        &self.files
    }
}
