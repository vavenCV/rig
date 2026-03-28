//! GitHub Copilot embedding model support.
//!
//! Uses the OpenAI-compatible `/embeddings` endpoint at
//! `https://api.individual.githubcopilot.com/embeddings`.
//!
//! # Example
//! ```no_run
//! use rig::providers::github_copilot;
//! use rig::client::EmbeddingsClient;
//!
//! let client = github_copilot::Client::new("COPILOT_API_TOKEN");
//! let model = client.embedding_model(github_copilot::TEXT_EMBEDDING_3_SMALL);
//! ```

use super::Client;
use crate::embeddings::{self, EmbeddingError};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai::embedding::EmbeddingResponse;
use serde::Deserialize;
use serde_json::json;

/// `text-embedding-3-small` — 1 536-dim embedding model via GitHub Copilot.
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-3-large` — 3 072-dim embedding model via GitHub Copilot.
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL => Some(1_536),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

/// A Copilot embedding model.
#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let dims = ndims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();

        let mut body = json!({
            "model": self.model,
            "input": documents,
        });

        if self.ndims > 0 {
            body["dimensions"] = json!(self.ndims);
        }

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "GitHub Copilot embedding token usage: {:?}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding
                                .embedding
                                .into_iter()
                                .filter_map(|n| n.as_f64())
                                .collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_request_serialization() {
        let body = json!({
            "model": "text-embedding-3-small",
            "input": ["Hello, world!"],
            "dimensions": 1536,
        });

        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["input"][0], "Hello, world!");
        assert_eq!(json["dimensions"], 1536);
    }

    #[test]
    fn embedding_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }"#;

        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "text-embedding-3-small");
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[0].embedding.len(), 3);
    }

    #[test]
    fn model_dimensions() {
        assert_eq!(
            model_dimensions_from_identifier(TEXT_EMBEDDING_3_SMALL),
            Some(1_536)
        );
        assert_eq!(
            model_dimensions_from_identifier(TEXT_EMBEDDING_3_LARGE),
            Some(3_072)
        );
        assert_eq!(model_dimensions_from_identifier("unknown-model"), None);
    }

    #[test]
    fn embedding_model_creation() {
        let _client = crate::providers::github_copilot::Client::new("dummy-token")
            .expect("Client::new() failed");
    }
}
