use crate::fff_search::{
    self, build_workspace_index as build_workspace_index_rs, FffSearchError, WorkspaceIndex,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[derive(Clone, Debug)]
#[pyclass]
pub struct FileSearchResultPy {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub score: u32,
}

#[derive(Clone, Debug)]
#[pyclass]
pub struct ContentSearchResultPy {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub line: u32,
    #[pyo3(get)]
    pub snippet: String,
    #[pyo3(get)]
    pub score: u32,
}

#[derive(Clone)]
#[pyclass]
pub struct WorkspaceSearchIndex {
    inner: WorkspaceIndex,
}

#[pymethods]
impl WorkspaceSearchIndex {
    pub fn file_count(&self) -> usize {
        self.inner.file_count()
    }

    pub fn content_file_count(&self) -> usize {
        self.inner.content_file_count()
    }

    pub fn search_files(&self, query: String, limit: usize) -> Vec<FileSearchResultPy> {
        fff_search::search_files(self.inner.file_index(), &query, limit)
            .into_iter()
            .map(|result| FileSearchResultPy {
                path: result.path,
                score: result.score,
            })
            .collect()
    }

    pub fn search_content(&self, query: String, limit: usize) -> Vec<ContentSearchResultPy> {
        fff_search::search_content(self.inner.content_index(), &query, limit)
            .into_iter()
            .map(|result| ContentSearchResultPy {
                path: result.path,
                line: result.line,
                snippet: result.snippet,
                score: result.score,
            })
            .collect()
    }
}

#[pyfunction]
pub fn build_workspace_index(repo_dir: String) -> PyResult<WorkspaceSearchIndex> {
    let root = std::path::Path::new(&repo_dir);
    let inner = build_workspace_index_rs(root).map_err(map_fff_error_to_py)?;
    Ok(WorkspaceSearchIndex { inner })
}

fn map_fff_error_to_py(error: FffSearchError) -> PyErr {
    PyRuntimeError::new_err(match error {
        FffSearchError::WorkspaceNotFound { path } => {
            format!("workspace not found: {path}")
        }
        FffSearchError::WorkspaceNotDirectory { path } => {
            format!("workspace is not a directory: {path}")
        }
        FffSearchError::PathOutsideWorkspace { path } => {
            format!("path outside workspace: {path}")
        }
    })
}
