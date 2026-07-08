//! DashScope-backed Qwen text embedder.
//!
//! This adapter uses Alibaba Cloud Model Studio's OpenAI-compatible embedding
//! endpoint for `text-embedding-v4`. It is intended for remote semantic search:
//! cass still owns local indexing/search, while DashScope only produces query
//! and document vectors.

use std::borrow::Cow;
use std::time::Duration;

use frankensearch::{ModelCategory, ModelTier};
use serde::{Deserialize, Serialize};

use super::embedder::{Embedder, EmbedderError, EmbedderResult};

pub const QWEN_V4_EMBEDDER_NAME: &str = "qwen-v4";
pub const DEFAULT_QWEN_EMBEDDING_MODEL: &str = "text-embedding-v4";
pub const DEFAULT_QWEN_EMBEDDING_DIMENSION: usize = 2048;
pub const QWEN_V4_EMBEDDER_ID: &str = "dashscope-text-embedding-v4-2048";

const DEFAULT_EMBEDDING_ENDPOINT: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings";
const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TEXT_EMBEDDING_V4_BATCH: usize = 10;
const VALID_TEXT_EMBEDDING_V4_DIMENSIONS: &[usize] = &[2048, 1536, 1024, 768, 512, 256, 128, 64];
/// In-flight request cap for one `embed_batch_sync` call. Default 1 preserves
/// the serial behavior; `CASS_DASHSCOPE_EMBEDDING_CONCURRENCY` /
/// `DASHSCOPE_EMBEDDING_CONCURRENCY` opt in to parallel waves.
const DEFAULT_EMBEDDING_CONCURRENCY: usize = 1;
const MAX_EMBEDDING_CONCURRENCY: usize = 16;
const EMBED_RETRY_MAX_ATTEMPTS: u32 = 3;

/// Embedder implementation backed by Alibaba Cloud DashScope `text-embedding-v4`.
pub struct DashScopeEmbedder {
    client: reqwest::blocking::Client,
    endpoint: String,
    api_key: String,
    model: String,
    dimension: usize,
    id: String,
}

impl DashScopeEmbedder {
    /// Load configuration from environment variables.
    ///
    /// Supported variables:
    /// - `DASHSCOPE_API_KEY` (preferred), with compatibility fallbacks
    ///   `QWEN_API_KEY`, `BAILIAN_API_KEY`, `MODELSTUDIO_API_KEY`,
    ///   `ALIBABA_CLOUD_API_KEY`.
    /// - `CASS_DASHSCOPE_EMBEDDING_URL` / `DASHSCOPE_EMBEDDING_URL` for the
    ///   exact embeddings endpoint.
    /// - `CASS_DASHSCOPE_EMBEDDING_BASE_URL` / `DASHSCOPE_EMBEDDING_BASE_URL`
    ///   or the shared DashScope base URL variables for a base endpoint.
    /// - `DASHSCOPE_WORKSPACE_ID` / `BAILIAN_WORKSPACE_ID` to build a
    ///   workspace endpoint when an exact/base URL is not set.
    /// - `CASS_DASHSCOPE_EMBEDDING_DIMENSIONS` to choose one of the official
    ///   text-embedding-v4 dimensions. Default: 2048.
    pub fn from_env() -> EmbedderResult<Self> {
        let api_key = first_nonempty_env(API_KEY_ENV_KEYS).ok_or_else(|| {
            embedder_unavailable(
                QWEN_V4_EMBEDDER_NAME,
                "missing DashScope API key; set DASHSCOPE_API_KEY",
            )
        })?;

        let endpoint = resolve_dashscope_embedding_endpoint();
        let model = first_nonempty_env(&[
            "CASS_DASHSCOPE_EMBEDDING_MODEL",
            "DASHSCOPE_EMBEDDING_MODEL",
        ])
        .unwrap_or_else(|| DEFAULT_QWEN_EMBEDDING_MODEL.to_string());
        let dimension = resolve_embedding_dimension()?;
        let timeout_ms = first_nonempty_env(&[
            "CASS_DASHSCOPE_EMBEDDING_TIMEOUT_MS",
            "DASHSCOPE_EMBEDDING_TIMEOUT_MS",
        ])
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_MS);

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| {
                embedding_failed(
                    QWEN_V4_EMBEDDER_NAME,
                    format!("HTTP client init failed: {e}"),
                )
            })?;

        Ok(Self {
            client,
            endpoint,
            api_key,
            id: embedding_id_for(&model, dimension),
            model,
            dimension,
        })
    }

    /// Whether a usable DashScope API key is present in the environment.
    pub fn is_configured_from_env() -> bool {
        first_nonempty_env(API_KEY_ENV_KEYS).is_some()
    }

    /// Stable embedder id for the default Qwen v4 configuration.
    pub fn default_embedder_id() -> &'static str {
        QWEN_V4_EMBEDDER_ID
    }

    #[cfg(test)]
    fn for_test() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            endpoint: DEFAULT_EMBEDDING_ENDPOINT.to_string(),
            api_key: "test-key".to_string(),
            model: DEFAULT_QWEN_EMBEDDING_MODEL.to_string(),
            dimension: DEFAULT_QWEN_EMBEDDING_DIMENSION,
            id: Self::default_embedder_id().to_string(),
        }
    }
}

impl Embedder for DashScopeEmbedder {
    fn embed_sync(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        let mut batch = self.embed_batch_sync(&[text])?;
        batch.pop().ok_or_else(|| {
            embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                "DashScope response did not contain an embedding",
            )
        })
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                "texts cannot be empty",
            ));
        }
        if let Some((idx, _)) = texts
            .iter()
            .enumerate()
            .find(|(_, text)| text.trim().is_empty())
        {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!("text {idx} is empty"),
            ));
        }

        let chunks: Vec<&[&str]> = texts.chunks(MAX_TEXT_EMBEDDING_V4_BATCH).collect();
        let concurrency = resolve_embedding_concurrency().min(chunks.len());

        if concurrency <= 1 {
            let mut embeddings = Vec::with_capacity(texts.len());
            for chunk in chunks {
                embeddings.extend(self.embed_request_chunk_with_retry(chunk)?);
            }
            return Ok(embeddings);
        }

        // Order-preserving wave parallelism: at most `concurrency` requests in
        // flight, and each wave joins fully before the next starts so an error
        // never leaves orphan requests running past the batch.
        let mut embeddings = Vec::with_capacity(texts.len());
        for wave in chunks.chunks(concurrency) {
            let wave_results: Vec<EmbedderResult<Vec<Vec<f32>>>> =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = wave
                        .iter()
                        .map(|&chunk| {
                            scope.spawn(move || self.embed_request_chunk_with_retry(chunk))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|handle| {
                            handle.join().unwrap_or_else(|_| {
                                Err(embedding_failed(
                                    QWEN_V4_EMBEDDER_NAME,
                                    "embedding worker thread panicked",
                                ))
                            })
                        })
                        .collect()
                });
            for result in wave_results {
                embeddings.extend(result?);
            }
        }
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }

    fn tier(&self) -> ModelTier {
        ModelTier::Quality
    }
}

impl DashScopeEmbedder {
    /// Retry wrapper: embedding requests are idempotent, so transient
    /// DashScope failures (timeouts, 429s, 5xx) get up to two linear-backoff
    /// retries before the whole batch is failed.
    fn embed_request_chunk_with_retry(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let mut last_err = None;
        for attempt in 0..EMBED_RETRY_MAX_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(1_000 * u64::from(attempt)));
            }
            match self.embed_request_chunk(texts) {
                Ok(embeddings) => return Ok(embeddings),
                Err(err) => last_err = Some(err),
            }
        }
        Err(last_err.expect("at least one embedding attempt"))
    }

    fn embed_request_chunk(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let request = DashScopeEmbeddingRequest {
            model: self.model.as_str(),
            input: texts,
            dimensions: self.dimension,
            encoding_format: "float",
        };

        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|e| {
                embedding_failed(
                    QWEN_V4_EMBEDDER_NAME,
                    format!("DashScope request failed: {e}"),
                )
            })?;

        let status = response.status();
        let body = response.text().map_err(|e| {
            embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!("DashScope response read failed: {e}"),
            )
        })?;

        let parsed: DashScopeEmbeddingResponse = serde_json::from_str(&body).map_err(|e| {
            embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!("DashScope response JSON parse failed: {e}; status={status}"),
            )
        })?;

        if !status.is_success() {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                parsed.error_summary(status.as_u16()),
            ));
        }

        response_to_embeddings(texts.len(), self.dimension, &parsed)
    }
}

#[derive(Serialize)]
struct DashScopeEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    dimensions: usize,
    encoding_format: &'a str,
}

#[derive(Debug, Deserialize)]
struct DashScopeEmbeddingResponse {
    data: Option<Vec<DashScopeEmbeddingItem>>,
    code: Option<String>,
    message: Option<String>,
    request_id: Option<String>,
    id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeEmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

impl DashScopeEmbeddingResponse {
    fn error_summary(&self, status: u16) -> String {
        let code = self.code.as_deref().unwrap_or("unknown");
        let message = self.message.as_deref().unwrap_or("no error message");
        let request_id = self
            .request_id
            .as_deref()
            .or(self.id.as_deref())
            .unwrap_or("unknown");
        format!("DashScope HTTP {status}: {code}: {message} (request_id={request_id})")
    }
}

fn response_to_embeddings(
    expected_count: usize,
    expected_dimension: usize,
    response: &DashScopeEmbeddingResponse,
) -> EmbedderResult<Vec<Vec<f32>>> {
    let data = response.data.as_ref().ok_or_else(|| {
        embedding_failed(
            QWEN_V4_EMBEDDER_NAME,
            "DashScope response did not contain embedding data",
        )
    })?;
    if data.len() != expected_count {
        return Err(embedding_failed(
            QWEN_V4_EMBEDDER_NAME,
            format!(
                "DashScope returned {} embeddings for {} inputs",
                data.len(),
                expected_count
            ),
        ));
    }

    let mut ordered: Vec<Option<Vec<f32>>> = (0..expected_count).map(|_| None).collect();
    for item in data {
        if item.index >= expected_count {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!(
                    "DashScope returned out-of-range embedding index {}",
                    item.index
                ),
            ));
        }
        if item.embedding.len() != expected_dimension {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!(
                    "DashScope embedding dimension mismatch at index {}: expected {}, got {}",
                    item.index,
                    expected_dimension,
                    item.embedding.len()
                ),
            ));
        }
        if ordered[item.index].is_some() {
            return Err(embedding_failed(
                QWEN_V4_EMBEDDER_NAME,
                format!(
                    "DashScope returned duplicate embedding index {}",
                    item.index
                ),
            ));
        }
        ordered[item.index] = Some(l2_normalize(item.embedding.clone()));
    }

    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, embedding)| {
            embedding.ok_or_else(|| {
                embedding_failed(
                    QWEN_V4_EMBEDDER_NAME,
                    format!("DashScope response omitted embedding index {idx}"),
                )
            })
        })
        .collect()
}

fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
    if norm_sq <= f32::EPSILON || !norm_sq.is_finite() {
        return vector;
    }
    let inv_norm = norm_sq.sqrt().recip();
    for value in &mut vector {
        *value *= inv_norm;
    }
    vector
}

fn resolve_embedding_concurrency() -> usize {
    first_nonempty_env(&[
        "CASS_DASHSCOPE_EMBEDDING_CONCURRENCY",
        "DASHSCOPE_EMBEDDING_CONCURRENCY",
    ])
    .and_then(|value| value.parse::<usize>().ok())
    .map(|value| value.clamp(1, MAX_EMBEDDING_CONCURRENCY))
    .unwrap_or(DEFAULT_EMBEDDING_CONCURRENCY)
}

fn resolve_embedding_dimension() -> EmbedderResult<usize> {
    let Some(raw) = first_nonempty_env(&[
        "CASS_DASHSCOPE_EMBEDDING_DIMENSIONS",
        "DASHSCOPE_EMBEDDING_DIMENSIONS",
    ]) else {
        return Ok(DEFAULT_QWEN_EMBEDDING_DIMENSION);
    };
    let dimension = raw.parse::<usize>().map_err(|_| {
        embedder_unavailable(
            QWEN_V4_EMBEDDER_NAME,
            format!("invalid DashScope embedding dimension {raw:?}"),
        )
    })?;
    if VALID_TEXT_EMBEDDING_V4_DIMENSIONS.contains(&dimension) {
        Ok(dimension)
    } else {
        Err(embedder_unavailable(
            QWEN_V4_EMBEDDER_NAME,
            format!(
                "unsupported text-embedding-v4 dimension {dimension}; use one of {}",
                VALID_TEXT_EMBEDDING_V4_DIMENSIONS
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ))
    }
}

fn resolve_dashscope_embedding_endpoint() -> String {
    if let Some(url) =
        first_nonempty_env(&["CASS_DASHSCOPE_EMBEDDING_URL", "DASHSCOPE_EMBEDDING_URL"])
    {
        return normalize_embedding_endpoint(&url).into_owned();
    }

    if let Some(base) = first_nonempty_env(&[
        "CASS_DASHSCOPE_EMBEDDING_BASE_URL",
        "DASHSCOPE_EMBEDDING_BASE_URL",
        "CASS_DASHSCOPE_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_API_BASE",
        "DASHSCOPE_BASE_HTTP_API_URL",
    ]) {
        return normalize_embedding_endpoint(&base).into_owned();
    }

    if let Some(workspace_id) = first_nonempty_env(&[
        "DASHSCOPE_WORKSPACE_ID",
        "BAILIAN_WORKSPACE_ID",
        "MODELSTUDIO_WORKSPACE_ID",
    ]) {
        let region = first_nonempty_env(&[
            "DASHSCOPE_REGION",
            "ALIBABA_CLOUD_REGION_ID",
            "ALICLOUD_REGION",
        ])
        .unwrap_or_else(|| "cn-beijing".to_string());
        return format!(
            "https://{}.{}.maas.aliyuncs.com/compatible-mode/v1/embeddings",
            workspace_id.trim(),
            region.trim()
        );
    }

    DEFAULT_EMBEDDING_ENDPOINT.to_string()
}

fn normalize_embedding_endpoint(raw: &str) -> Cow<'_, str> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        Cow::Owned(trimmed.to_string())
    } else if trimmed.ends_with("/compatible-mode/v1") {
        Cow::Owned(format!("{trimmed}/embeddings"))
    } else if let Some(host) = trimmed.strip_suffix("/api/v1") {
        Cow::Owned(format!("{host}/compatible-mode/v1/embeddings"))
    } else {
        Cow::Owned(format!("{trimmed}/compatible-mode/v1/embeddings"))
    }
}

fn first_nonempty_env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        dotenvy::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

const API_KEY_ENV_KEYS: &[&str] = &[
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "BAILIAN_API_KEY",
    "MODELSTUDIO_API_KEY",
    "ALIBABA_CLOUD_API_KEY",
];

fn embedding_id_for(model: &str, dimension: usize) -> String {
    let raw = model
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    let sanitized = raw
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    format!("dashscope-{sanitized}-{dimension}")
}

fn embedder_unavailable(model: &str, reason: impl Into<String>) -> EmbedderError {
    EmbedderError::EmbedderUnavailable {
        model: model.to_string(),
        reason: reason.into(),
    }
}

fn embedding_failed(model: &str, source: impl Into<String>) -> EmbedderError {
    EmbedderError::EmbeddingFailed {
        model: model.to_string(),
        source: Box::new(std::io::Error::other(source.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_exact_embedding_endpoint() {
        let endpoint = normalize_embedding_endpoint(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        );
        assert_eq!(
            endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        );
    }

    #[test]
    fn normalize_compatible_base_endpoint() {
        let endpoint =
            normalize_embedding_endpoint("https://dashscope.aliyuncs.com/compatible-mode/v1/");
        assert_eq!(
            endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        );
    }

    #[test]
    fn normalize_sdk_api_base_to_compatible_mode_endpoint() {
        let endpoint =
            normalize_embedding_endpoint("https://ws.cn-beijing.maas.aliyuncs.com/api/v1");
        assert_eq!(
            endpoint,
            "https://ws.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"
        );
    }

    #[test]
    fn default_id_matches_registry_id() {
        assert_eq!(
            embedding_id_for(
                DEFAULT_QWEN_EMBEDDING_MODEL,
                DEFAULT_QWEN_EMBEDDING_DIMENSION
            ),
            DashScopeEmbedder::default_embedder_id()
        );
    }

    #[test]
    fn parses_openai_compatible_response_in_input_order() {
        let response: DashScopeEmbeddingResponse = serde_json::from_str(
            r#"{
                "object": "list",
                "model": "text-embedding-v4",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }"#,
        )
        .unwrap();

        let embeddings = response_to_embeddings(2, 3, &response).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![1.0, 0.0, 0.0]);
        assert!((embeddings[1][1] - 0.6).abs() < 1e-6);
        assert!((embeddings[1][2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let response: DashScopeEmbeddingResponse = serde_json::from_str(
            r#"{
                "data": [
                    {"index": 0, "embedding": [1.0, 2.0]}
                ]
            }"#,
        )
        .unwrap();

        let err = response_to_embeddings(1, 3, &response).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn qwen_embedder_metadata_is_quality_semantic() {
        let embedder = DashScopeEmbedder::for_test();
        assert_eq!(embedder.id(), DashScopeEmbedder::default_embedder_id());
        assert_eq!(embedder.dimension(), DEFAULT_QWEN_EMBEDDING_DIMENSION);
        assert_eq!(embedder.model_name(), DEFAULT_QWEN_EMBEDDING_MODEL);
        assert!(embedder.is_semantic());
        assert_eq!(embedder.category(), ModelCategory::TransformerEmbedder);
        assert_eq!(embedder.tier(), ModelTier::Quality);
    }
}
