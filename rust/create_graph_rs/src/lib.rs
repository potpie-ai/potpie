// Local build command:
// uv run --with maturin maturin develop --manifest-path rust/create_graph_rs/Cargo.toml --quiet

mod graph_payload;
mod tag_extract;
mod text_filter;
mod traversal;

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilePayload {
    #[pyo3(get)]
    pub relative_path: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub is_binary: bool,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodePayload {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub node_type: String,
    #[pyo3(get)]
    pub file: String,
    #[pyo3(get)]
    pub line: u32,
    #[pyo3(get)]
    pub end_line: u32,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub class_name: Option<String>,
    #[pyo3(get)]
    pub text: Option<String>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgePayload {
    #[pyo3(get)]
    pub source_id: String,
    #[pyo3(get)]
    pub target_id: String,
    #[pyo3(get)]
    pub edge_type: String,
    #[pyo3(get)]
    pub ident: Option<String>,
    #[pyo3(get)]
    pub ref_line: Option<u32>,
    #[pyo3(get)]
    pub end_ref_line: Option<u32>,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphPayload {
    #[pyo3(get)]
    pub nodes: Vec<NodePayload>,
    #[pyo3(get)]
    pub edges: Vec<EdgePayload>,
}

#[pyfunction]
pub fn discover_files(repo_dir: &str) -> PyResult<Vec<FilePayload>> {
    traversal::discover_files(repo_dir).map_err(|error| PyOSError::new_err(error.to_string()))
}

#[pyfunction]
pub fn build_graph_payload(repo_dir: &str) -> PyResult<GraphPayload> {
    graph_payload::build_graph_payload(repo_dir)
}

#[pymodule]
#[pyo3(name = "create_graph_rs")]
fn create_graph_rs_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FilePayload>()?;
    m.add_class::<NodePayload>()?;
    m.add_class::<EdgePayload>()?;
    m.add_class::<GraphPayload>()?;
    m.add_function(wrap_pyfunction!(discover_files, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_payload, m)?)?;
    Ok(())
}
