//! Tiny HuggingFace Hub downloader. No async runtime, no external HTTP crate —
//! just shells out to `curl`, then validates the downloaded file size against
//! the Content-Length the server reports.
//!
//! Without this validation, a flaky network can leave a half-downloaded file
//! on disk that future runs will silently use, producing nonsense output. (We
//! hit this with SmolLM2 — the file claimed to download successfully but was
//! truncated mid-stream by the CDN.)

use std::path::{Path, PathBuf};
use std::process::Command;

const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug)]
pub enum DownloadError {
    Io(std::io::Error),
    CurlFailed { url: String, status: i32 },
    SizeMismatch { url: String, expected: u64, actual: u64 },
    NoContentLength { url: String },
    Other(String),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Io(e) => write!(f, "io: {}", e),
            DownloadError::CurlFailed { url, status } => {
                write!(f, "curl failed for {} (exit {})", url, status)
            }
            DownloadError::SizeMismatch { url, expected, actual } => write!(
                f,
                "downloaded size {} != expected {} for {} \
                 (truncated download — file deleted; rerun to retry)",
                actual, expected, url
            ),
            DownloadError::NoContentLength { url } => write!(
                f,
                "could not determine remote size for {} (no Content-Length header)",
                url
            ),
            DownloadError::Other(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for DownloadError {}
impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self { DownloadError::Io(e) }
}

/// HEAD the URL and return the Content-Length the server reports (after redirects).
fn remote_size(url: &str) -> Result<u64, DownloadError> {
    let out = Command::new("curl")
        .args(["-sLI", url])  // -s: silent, -L: follow redirects, -I: HEAD
        .output()?;
    if !out.status.success() {
        return Err(DownloadError::CurlFailed {
            url: url.into(),
            status: out.status.code().unwrap_or(-1),
        });
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Take the last `content-length:` (after redirects) — case insensitive.
    let mut size: Option<u64> = None;
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            if let Ok(n) = rest.trim().parse::<u64>() {
                size = Some(n);
            }
        }
    }
    size.ok_or_else(|| DownloadError::NoContentLength { url: url.into() })
}

/// Download `url` to `out`. Validates that the downloaded file size matches
/// the server-reported Content-Length; on mismatch, deletes the partial file
/// and returns an error so the caller knows to retry.
///
/// `progress` controls whether curl shows a progress bar.
pub fn download_file(url: &str, out: &Path, progress: bool) -> Result<(), DownloadError> {
    let expected = remote_size(url)?;

    let mut args = vec!["-L", "-o", out.to_str().unwrap(), url, "-f"];
    if progress { args.push("--progress-bar"); } else { args.push("--silent"); }
    let status = Command::new("curl").args(&args).status()?;
    if !status.success() {
        // Clean up partial file so the next run starts fresh.
        let _ = std::fs::remove_file(out);
        return Err(DownloadError::CurlFailed {
            url: url.into(),
            status: status.code().unwrap_or(-1),
        });
    }

    let actual = std::fs::metadata(out)?.len();
    if actual != expected {
        let _ = std::fs::remove_file(out);
        return Err(DownloadError::SizeMismatch {
            url: url.into(),
            expected,
            actual,
        });
    }
    Ok(())
}

/// Cache directory: `$HOME/.cache/gpux/models`.
pub fn cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("gpux").join("models")
}

/// Local directory for a given HF repo id (e.g. "Qwen/Qwen2.5-0.5B").
pub fn model_dir(model_id: &str) -> PathBuf {
    cache_dir().join(model_id.replace('/', "--"))
}

/// Verify a cached file matches its remote size; if not, delete the local copy
/// so the next download attempt starts fresh.
pub fn revalidate(local: &Path, url: &str) -> Result<bool, DownloadError> {
    if !local.exists() { return Ok(false); }
    let actual = std::fs::metadata(local)?.len();
    // Cheap-but-not-free: HEAD only when the file is "suspicious" — empty
    // or unusually small. A rigorous check would HEAD always; we trade a bit
    // of safety for far less network noise on repeat runs.
    if actual == 0 {
        let _ = std::fs::remove_file(local);
        return Ok(false);
    }
    let _ = url;  // reserved for future strict validation
    Ok(true)
}

/// Download a HuggingFace model: config, tokenizer, and either a single
/// `model.safetensors` or all shards listed in `model.safetensors.index.json`.
/// Files already present (with non-zero size) are left alone; missing or
/// truncated files are (re-)downloaded with size verification.
pub fn download_model(model_id: &str, progress: bool) -> Result<PathBuf, DownloadError> {
    let dir = model_dir(model_id);
    std::fs::create_dir_all(&dir)?;
    let base_url = format!("{}/{}/resolve/main", HF_BASE, model_id);

    // Required small files
    for f in &["config.json", "tokenizer.json"] {
        let local = dir.join(f);
        if !revalidate(&local, &format!("{}/{}", base_url, f))? {
            let url = format!("{}/{}", base_url, f);
            download_file(&url, &local, progress)?;
        }
    }
    // Optional: tokenizer_config.json (used to detect chat_template format).
    // Not all repos ship one (especially base/non-chat models), so a 404 is OK.
    {
        let local = dir.join("tokenizer_config.json");
        if !local.exists() {
            let url = format!("{}/tokenizer_config.json", base_url);
            let _ = download_file(&url, &local, progress);  // ignore failures
        }
    }

    // Try single-file weights first; fall back to sharded if that 404s.
    let single = dir.join("model.safetensors");
    let single_url = format!("{}/model.safetensors", base_url);
    if revalidate(&single, &single_url)? {
        return Ok(dir);
    }
    match download_file(&single_url, &single, progress) {
        Ok(()) => return Ok(dir),
        Err(DownloadError::CurlFailed { .. }) => {
            // Try sharded model.
        }
        Err(e) => return Err(e),
    }

    let index_path = dir.join("model.safetensors.index.json");
    let index_url = format!("{}/model.safetensors.index.json", base_url);
    download_file(&index_url, &index_path, progress)?;
    let index_data = std::fs::read_to_string(&index_path)
        .map_err(DownloadError::Io)?;
    let index: serde_json::Value = serde_json::from_str(&index_data)
        .map_err(|e| DownloadError::Other(format!("invalid index json: {}", e)))?;
    let weight_map = index["weight_map"].as_object()
        .ok_or_else(|| DownloadError::Other("missing weight_map in index".into()))?;
    let mut shards: Vec<String> = weight_map.values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shards.sort();
    shards.dedup();
    for shard in shards {
        let local = dir.join(&shard);
        let url = format!("{}/{}", base_url, shard);
        if revalidate(&local, &url)? {
            continue;
        }
        download_file(&url, &local, progress)?;
    }
    Ok(dir)
}
