use crate::fff_search::{ContentIndex, ContentSearchResult, IndexedFileContent};

pub fn build_bigram_index(files: Vec<IndexedFileContent>) -> ContentIndex {
    ContentIndex { files }
}

pub fn search_content(index: &ContentIndex, query: &str, limit: usize) -> Vec<ContentSearchResult> {
    let query = query.trim().to_lowercase();
    if query.is_empty() || limit == 0 {
        return Vec::new();
    }

    let exact_terms = query
        .split_whitespace()
        .filter(|term| term.chars().any(char::is_alphanumeric))
        .collect::<Vec<_>>();

    let mut results = Vec::<ContentSearchResult>::new();

    for file in &index.files {
        for (i, line) in file.content.lines().enumerate() {
            let normalized_line = line.to_lowercase();
            let line_number = u32::try_from(i + 1).unwrap_or(u32::MAX);

            if normalized_line.contains(&query) {
                let occurrence_count = count_substring_occurrences(&normalized_line, &query);
                results.push(ContentSearchResult {
                    path: file.path.clone(),
                    line: line_number,
                    snippet: trim_snippet(line),
                    score: 1000 + occurrence_count,
                });
                continue;
            }

            if exact_terms.len() >= 2 {
                let matched = exact_terms
                    .iter()
                    .filter(|term| normalized_line.contains(*term))
                    .count();
                if matched == exact_terms.len() {
                    results.push(ContentSearchResult {
                        path: file.path.clone(),
                        line: line_number,
                        snippet: trim_snippet(line),
                        score: 700 + u32::try_from(matched).unwrap_or(u32::MAX),
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
    results.truncate(limit);
    results
}

fn count_substring_occurrences(text: &str, pattern: &str) -> u32 {
    if pattern.is_empty() {
        return 0;
    }

    let mut remaining = text;
    let mut count = 0u32;
    while let Some(pos) = remaining.find(pattern) {
        count = count.saturating_add(1);
        remaining = &remaining[pos + pattern.len()..];
    }
    count
}

fn trim_snippet(line: &str) -> String {
    line.chars().take(200).collect()
}
