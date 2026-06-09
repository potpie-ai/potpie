use crate::fff_search::{FileIndex, FileSearchResult};
use std::collections::HashMap;

pub fn build_file_index(paths: Vec<String>) -> FileIndex {
    let entries = paths
        .into_iter()
        .map(|path| path.replace('\\', "/"))
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>();
    FileIndex { entries }
}

pub fn search_files(index: &FileIndex, query: &str, limit: usize) -> Vec<FileSearchResult> {
    let query = query.trim().to_lowercase();
    if query.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut scores = HashMap::<String, u32>::new();
    for raw_path in index.entries() {
        let normalized_path = raw_path.to_lowercase();
        let mut basename = "";
        if let Some(last_segment) = normalized_path.rsplit('/').next() {
            basename = last_segment;
        }
        let basename_stem = file_stem(basename);
        let segments = normalized_path.split('/').collect::<Vec<_>>();
        let segment_match = segments.iter().any(|segment| segment.starts_with(&query));

        let score = if normalized_path == query {
            1000
        } else if basename == query || basename_stem == query {
            950
        } else if normalized_path.starts_with(&query) {
            850
        } else if basename.starts_with(&query) {
            800
        } else if segment_match {
            700
        } else if basename.contains(&query) {
            600
        } else if normalized_path.contains(&query) {
            500
        } else {
            0
        };

        if score > 0 {
            let path = raw_path.clone();
            if let Some(existing) = scores.get_mut(&path) {
                if score > *existing {
                    *existing = score;
                }
            } else {
                scores.insert(path, score);
            }
        }
    }

    let mut results = scores
        .into_iter()
        .map(|(path, score)| FileSearchResult { path, score })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
    results.truncate(limit);
    results
}

fn file_stem(basename: &str) -> &str {
    basename
        .rsplit_once('.')
        .and_then(|(stem, _extension)| (!stem.is_empty()).then_some(stem))
        .unwrap_or(basename)
}
