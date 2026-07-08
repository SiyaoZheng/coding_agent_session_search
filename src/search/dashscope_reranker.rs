//! DashScope-backed Qwen reranker.
//!
//! This adapter keeps cass's local retrieval path intact: Tantivy/BM25 finds
//! candidate session messages locally, then DashScope `qwen3-rerank` scores only
//! those candidate texts.

use std::borrow::Cow;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::search::reranker::{
    RerankDocument, RerankScore, Reranker, RerankerError, RerankerResult,
};

pub const QWEN3_RERANKER_NAME: &str = "qwen3-rerank";
pub const QWEN3_RERANKER_ID: &str = "dashscope-qwen3-rerank";

const DEFAULT_RERANK_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks";
const DEFAULT_TIMEOUT_MS: u64 = 10_000;
const MAX_QWEN3_DOCUMENTS: usize = 500;
const DEFAULT_INSTRUCT: &str = "Retrieve semantically similar developer session records. Prefer exact named entities, file paths, commands, error messages, project names, Chinese titles, and task intent.";

/// Reranker implementation backed by Alibaba Cloud DashScope `qwen3-rerank`.
pub struct DashScopeReranker {
    client: reqwest::blocking::Client,
    endpoint: String,
    api_key: String,
    model: String,
    instruct: String,
}

impl DashScopeReranker {
    /// Load configuration from environment variables.
    ///
    /// Supported variables:
    /// - `DASHSCOPE_API_KEY` (preferred), with compatibility fallbacks
    ///   `QWEN_API_KEY`, `BAILIAN_API_KEY`, `MODELSTUDIO_API_KEY`,
    ///   `ALIBABA_CLOUD_API_KEY`.
    /// - `CASS_DASHSCOPE_RERANK_URL` / `DASHSCOPE_RERANK_URL` for the exact
    ///   rerank endpoint.
    /// - `DASHSCOPE_WORKSPACE_ID` / `BAILIAN_WORKSPACE_ID` to build the
    ///   workspace endpoint when an exact URL is not set.
    /// - `CASS_DASHSCOPE_RERANK_INSTRUCT` to tune the ranking policy.
    pub fn from_env() -> RerankerResult<Self> {
        let api_key = first_nonempty_env(&[
            "DASHSCOPE_API_KEY",
            "QWEN_API_KEY",
            "BAILIAN_API_KEY",
            "MODELSTUDIO_API_KEY",
            "ALIBABA_CLOUD_API_KEY",
        ])
        .ok_or_else(|| {
            rerank_failed(
                QWEN3_RERANKER_NAME,
                "missing DashScope API key; set DASHSCOPE_API_KEY",
            )
        })?;

        let endpoint = resolve_dashscope_rerank_endpoint();
        let model = first_nonempty_env(&["CASS_DASHSCOPE_RERANK_MODEL", "DASHSCOPE_RERANK_MODEL"])
            .unwrap_or_else(|| QWEN3_RERANKER_NAME.to_string());
        let instruct = first_nonempty_env(&[
            "CASS_DASHSCOPE_RERANK_INSTRUCT",
            "DASHSCOPE_RERANK_INSTRUCT",
        ])
        .unwrap_or_else(|| DEFAULT_INSTRUCT.to_string());
        let timeout_ms = first_nonempty_env(&[
            "CASS_DASHSCOPE_RERANK_TIMEOUT_MS",
            "DASHSCOPE_RERANK_TIMEOUT_MS",
        ])
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_MS);

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| {
                rerank_failed(QWEN3_RERANKER_NAME, format!("HTTP client init failed: {e}"))
            })?;

        Ok(Self {
            client,
            endpoint,
            api_key,
            model,
            instruct,
        })
    }

    #[cfg(test)]
    fn for_test(endpoint: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            endpoint: endpoint.to_string(),
            api_key: "test-key".to_string(),
            model: QWEN3_RERANKER_NAME.to_string(),
            instruct: DEFAULT_INSTRUCT.to_string(),
        }
    }
}

impl Reranker for DashScopeReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> RerankerResult<Vec<RerankScore>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(rerank_failed(QWEN3_RERANKER_NAME, "query cannot be empty"));
        }
        if documents.is_empty() {
            return Err(rerank_failed(
                QWEN3_RERANKER_NAME,
                "documents cannot be empty",
            ));
        }
        if documents.len() > MAX_QWEN3_DOCUMENTS {
            return Err(rerank_failed(
                QWEN3_RERANKER_NAME,
                format!(
                    "qwen3-rerank accepts at most {MAX_QWEN3_DOCUMENTS} documents per request; rerank fewer search hits with --limit"
                ),
            ));
        }

        let mut request_documents = Vec::with_capacity(documents.len());
        for doc in documents {
            let text = doc.text.trim();
            if text.is_empty() {
                return Err(rerank_failed(
                    QWEN3_RERANKER_NAME,
                    format!("document {} is empty", doc.doc_id),
                ));
            }
            request_documents.push(text);
        }

        let request = DashScopeRerankRequest {
            model: self.model.as_str(),
            documents: request_documents,
            query,
            top_n: documents.len(),
            instruct: self.instruct.as_str(),
        };

        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|e| {
                rerank_failed(
                    QWEN3_RERANKER_NAME,
                    format!("DashScope request failed: {e}"),
                )
            })?;

        let status = response.status();
        let body = response.text().map_err(|e| {
            rerank_failed(
                QWEN3_RERANKER_NAME,
                format!("DashScope response read failed: {e}"),
            )
        })?;

        let parsed: DashScopeRerankResponse = serde_json::from_str(&body).map_err(|e| {
            rerank_failed(
                QWEN3_RERANKER_NAME,
                format!("DashScope response JSON parse failed: {e}; status={status}"),
            )
        })?;

        if !status.is_success() {
            return Err(rerank_failed(
                QWEN3_RERANKER_NAME,
                parsed.error_summary(status.as_u16()),
            ));
        }

        response_to_scores(documents, &parsed)
    }

    fn id(&self) -> &str {
        QWEN3_RERANKER_ID
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[derive(Serialize)]
struct DashScopeRerankRequest<'a> {
    model: &'a str,
    documents: Vec<&'a str>,
    query: &'a str,
    top_n: usize,
    instruct: &'a str,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankResponse {
    results: Option<Vec<DashScopeRerankItem>>,
    output: Option<DashScopeRerankOutput>,
    code: Option<String>,
    message: Option<String>,
    request_id: Option<String>,
    id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankOutput {
    results: Vec<DashScopeRerankItem>,
}

#[derive(Debug, Deserialize)]
struct DashScopeRerankItem {
    index: usize,
    relevance_score: f32,
}

impl DashScopeRerankResponse {
    fn results(&self) -> Option<&[DashScopeRerankItem]> {
        self.results
            .as_deref()
            .or_else(|| self.output.as_ref().map(|output| output.results.as_slice()))
    }

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

fn response_to_scores(
    documents: &[RerankDocument],
    response: &DashScopeRerankResponse,
) -> RerankerResult<Vec<RerankScore>> {
    let results = response.results().ok_or_else(|| {
        rerank_failed(
            QWEN3_RERANKER_NAME,
            "DashScope response did not contain rerank results",
        )
    })?;

    let mut scores = vec![0.0f32; documents.len()];
    for item in results {
        if item.index < scores.len() {
            scores[item.index] = item.relevance_score;
        }
    }

    Ok(documents
        .iter()
        .enumerate()
        .map(|(idx, doc)| RerankScore {
            doc_id: doc.doc_id.clone(),
            score: scores[idx],
            original_rank: idx,
            raw_logit: None,
        })
        .collect())
}

fn resolve_dashscope_rerank_endpoint() -> String {
    if let Some(url) = first_nonempty_env(&["CASS_DASHSCOPE_RERANK_URL", "DASHSCOPE_RERANK_URL"]) {
        return normalize_rerank_endpoint(&url).into_owned();
    }

    if let Some(base) = first_nonempty_env(&[
        "CASS_DASHSCOPE_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_API_BASE",
        "DASHSCOPE_BASE_HTTP_API_URL",
    ]) {
        return normalize_rerank_endpoint(&base).into_owned();
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
            "https://{}.{}.maas.aliyuncs.com/compatible-api/v1/reranks",
            workspace_id.trim(),
            region.trim()
        );
    }

    DEFAULT_RERANK_ENDPOINT.to_string()
}

fn normalize_rerank_endpoint(raw: &str) -> Cow<'_, str> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/reranks") {
        Cow::Owned(trimmed.to_string())
    } else if trimmed.ends_with("/compatible-api/v1") {
        Cow::Owned(format!("{trimmed}/reranks"))
    } else if let Some(host) = trimmed.strip_suffix("/api/v1") {
        Cow::Owned(format!("{host}/compatible-api/v1/reranks"))
    } else {
        Cow::Owned(format!("{trimmed}/compatible-api/v1/reranks"))
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

fn rerank_failed(model: &str, source: impl Into<String>) -> RerankerError {
    RerankerError::RerankFailed {
        model: model.to_string(),
        source: source.into().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_exact_rerank_endpoint() {
        let endpoint =
            normalize_rerank_endpoint("https://dashscope.aliyuncs.com/compatible-api/v1/reranks");
        assert_eq!(
            endpoint,
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        );
    }

    #[test]
    fn normalize_compatible_base_endpoint() {
        let endpoint =
            normalize_rerank_endpoint("https://dashscope.aliyuncs.com/compatible-api/v1/");
        assert_eq!(
            endpoint,
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        );
    }

    #[test]
    fn normalize_sdk_api_base_to_compatible_endpoint() {
        let endpoint = normalize_rerank_endpoint("https://ws.cn-beijing.maas.aliyuncs.com/api/v1");
        assert_eq!(
            endpoint,
            "https://ws.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
        );
    }

    #[test]
    fn parses_qwen3_top_level_response() {
        let docs = vec![
            RerankDocument {
                doc_id: "a".to_string(),
                text: "alpha".to_string(),
            },
            RerankDocument {
                doc_id: "b".to_string(),
                text: "beta".to_string(),
            },
        ];
        let response: DashScopeRerankResponse = serde_json::from_str(
            r#"{
                "object": "list",
                "model": "qwen3-rerank",
                "results": [
                    {"index": 1, "relevance_score": 0.25},
                    {"index": 0, "relevance_score": 0.75}
                ]
            }"#,
        )
        .unwrap();

        let scores = response_to_scores(&docs, &response).unwrap();
        assert_eq!(scores[0].doc_id, "a");
        assert_eq!(scores[0].score, 0.75);
        assert_eq!(scores[1].doc_id, "b");
        assert_eq!(scores[1].score, 0.25);
    }

    #[test]
    fn rejects_more_than_qwen3_document_limit() {
        let reranker = DashScopeReranker::for_test("https://example.invalid/reranks");
        let docs: Vec<RerankDocument> = (0..=MAX_QWEN3_DOCUMENTS)
            .map(|idx| RerankDocument {
                doc_id: idx.to_string(),
                text: "doc".to_string(),
            })
            .collect();
        let err = reranker.rerank_sync("query", &docs).unwrap_err();
        assert!(err.to_string().contains("at most 500 documents"));
    }
}
