use std::fs;
use std::io;
use std::path::Path;

const EXCLUDE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "ico", "svg", "mp4", "avi", "mov", "wmv",
    "flv", "ipynb",
];

const INCLUDE_EXTENSIONS: &[&str] = &[
    "py", "js", "ts", "c", "cs", "cpp", "h", "hpp", "el", "ex", "exs", "elm", "go", "java", "ml",
    "mli", "php", "ql", "rb", "rs", "md", "txt", "json", "yaml", "yml", "toml", "ini", "cfg",
    "conf", "xml", "html", "css", "sh", "ps1", "psm1", "mdx", "xsq", "proto",
];

pub fn is_text_file(path: &Path) -> bool {
    let path_string = path.to_string_lossy();
    let ext = path_string.rsplit('.').next().unwrap_or_default();

    if EXCLUDE_EXTENSIONS.contains(&ext) {
        return false;
    }

    if INCLUDE_EXTENSIONS.contains(&ext) {
        return true;
    }

    open_text_file(path).unwrap_or(false)
}

pub fn read_text(path: &Path) -> String {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return String::new(),
    };

    decode_text(&bytes).unwrap_or_default()
}

fn open_text_file(path: &Path) -> io::Result<bool> {
    let bytes = fs::read(path)?;
    let probe = &bytes[..bytes.len().min(8192)];
    Ok(decode_text(probe).is_some())
}

fn decode_text(bytes: &[u8]) -> Option<String> {
    decode_utf8(bytes)
        .or_else(|| decode_utf8_sig(bytes))
        .or_else(|| decode_utf16(bytes))
        .or_else(|| Some(decode_latin1(bytes)))
}

fn decode_utf8(bytes: &[u8]) -> Option<String> {
    String::from_utf8(bytes.to_vec()).ok()
}

fn decode_utf8_sig(bytes: &[u8]) -> Option<String> {
    let stripped = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF])?;
    String::from_utf8(stripped.to_vec()).ok()
}

fn decode_utf16(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 2 || bytes.len() % 2 != 0 {
        return None;
    }

    let (big_endian, body) = if let Some(stripped) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        (true, stripped)
    } else if let Some(stripped) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        (false, stripped)
    } else {
        return None;
    };

    if body.len() % 2 != 0 {
        return None;
    }

    let code_units: Vec<u16> = body
        .chunks_exact(2)
        .map(|chunk| {
            if big_endian {
                u16::from_be_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_le_bytes([chunk[0], chunk[1]])
            }
        })
        .collect();

    String::from_utf16(&code_units).ok()
}

fn decode_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}
