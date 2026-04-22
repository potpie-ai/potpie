use jwalk::WalkDir;
use log::{error, info};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const MMAP_THRESHOLD: u64 = 256 * 1024;

pub fn process_file(path: &Path) -> std::io::Result<usize> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();

    if len == 0 {
        return Ok(0);
    }

    if len >= MMAP_THRESHOLD {
        // SAFETY: external modification during the mapping's lifetime is UB.
        // We accept that risk for a read-only indexing pass.
        let mmap = unsafe { Mmap::map(&file)? };
        return Ok(bytecount::count(&mmap, b'\n'));
    }

    let capacity: usize = len.try_into().unwrap_or_else(|_| {
        debug_assert!(false, "File size {} exceeds usize::MAX, truncating capacity", len);
        usize::MAX
    });
    let mut buf = Vec::with_capacity(capacity);
    let mut f = file;
    f.read_to_end(&mut buf)?;
    Ok(bytecount::count(&buf, b'\n'))
}

pub fn index(folder: &str) -> std::io::Result<()> {
    env_logger::try_init().ok();
    info!("Starting index for folder: {}", folder);

    let path = Path::new(folder);
    if !path.exists() {
        error!("Folder does not exist: {}", folder);
        return Err(ErrorKind::NotFound.into());
    }

    let mut walk_error_count: usize = 0;
    let files: Vec<PathBuf> = WalkDir::new(path)
        .follow_links(true)
        .skip_hidden(false)
        .into_iter()
        .filter_map(|e| match e {
            Ok(entry) => Some(entry),
            Err(e) => {
                walk_error_count += 1;
                error!("WalkDir error: {}", e);
                None
            }
        })
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path())
        .collect();

    let file_count = AtomicUsize::new(0);
    let total_lines: usize = files
        .par_iter()
        .with_min_len(32)
        .map(|p| match process_file(p) {
            Ok(lines) => {
                file_count.fetch_add(1, Ordering::Relaxed);
                lines
            }
            Err(e) => {
                error!("Failed to process {:?}: {}", p, e);
                0
            }
        })
        .sum();

    if walk_error_count > 0 {
        info!(
            "Completed index for {}: {} files, {} total lines, {} walk errors",
            folder,
            file_count.load(Ordering::Relaxed),
            total_lines,
            walk_error_count
        );
    } else {
        info!(
            "Completed index for {}: {} files, {} total lines",
            folder,
            file_count.load(Ordering::Relaxed),
            total_lines
        );
    }
    Ok(())
}
