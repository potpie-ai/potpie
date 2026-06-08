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
