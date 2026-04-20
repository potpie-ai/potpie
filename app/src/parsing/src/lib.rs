mod parse;

pub use parse::{index, process_file};

#[pyo3::prelude::pymodule]
mod parsing_rs {
    use super::index as walkdir_index;
    use pyo3::prelude::*;

    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pyfunction]
    fn index(py: Python<'_>, folder: &str) -> PyResult<()> {
        py.detach(|| walkdir_index(folder))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
}
