//! GitHub Copilot model listing support.
//!
//! Queries the `/models` endpoint to discover available models.
//!
//! # Example
//! ```no_run
//! use rig::providers::github_copilot;
//! use rig::client::ModelListingClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = github_copilot::Client::new("COPILOT_API_TOKEN");
//! let models = client.list_models().await?;
//! for model in models.iter() {
//!     println!("{}: {}", model.id, model.display_name());
//! }
//! # Ok(())
//! # }
//! ```

use super::Client;
use crate::client::ModelLister;
use crate::http_client::HttpClientExt;
use crate::model::{Model, ModelList, ModelListingError};
use serde::Deserialize;

/// Response from the Copilot `/models` endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(alias = "models")]
    data: Vec<ModelEntry>,
}

/// A single model entry from the API response.
#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(alias = "display_name", alias = "displayName")]
    name: Option<String>,
    #[serde(alias = "created")]
    created_at: Option<u64>,
    owned_by: Option<String>,
}

impl From<ModelEntry> for Model {
    fn from(entry: ModelEntry) -> Self {
        Model {
            id: entry.id.clone(),
            name: entry.name.or(Some(entry.id)),
            description: None,
            r#type: None,
            created_at: entry.created_at,
            owned_by: entry.owned_by,
            context_length: None,
        }
    }
}

/// Lister that fetches models from the Copilot API.
#[derive(Clone)]
pub struct CopilotModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for CopilotModelLister<H>
where
    H: HttpClientExt + Clone + Send + Sync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let req = self
            .client
            .get("/models")
            .map_err(|e| ModelListingError::RequestError {
                message: e.to_string(),
            })?
            .body(Vec::<u8>::new())
            .map_err(|e| ModelListingError::RequestError {
                message: e.to_string(),
            })?;

        let response = self
            .client
            .send::<_, bytes::Bytes>(req)
            .await
            .map_err(|e| ModelListingError::RequestError {
                message: e.to_string(),
            })?;

        let status = response.status();

        if !status.is_success() {
            let body = response
                .into_body()
                .into_future()
                .await
                .map(|b| String::from_utf8_lossy(&b).to_string())
                .unwrap_or_default();

            return Err(ModelListingError::ApiError {
                status_code: status.as_u16(),
                message: body,
            });
        }

        let body = response
            .into_body()
            .into_future()
            .await
            .map_err(|e| ModelListingError::RequestError {
                message: e.to_string(),
            })?
            .to_vec();

        let models_resp: ModelsResponse =
            serde_json::from_slice(&body).map_err(|e| ModelListingError::ParseError {
                message: e.to_string(),
            })?;

        let models: Vec<Model> = models_resp.data.into_iter().map(Model::from).collect();
        Ok(ModelList::new(models))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_models_response_data_key() {
        let json = r#"{
            "data": [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "created_at": 1700000000,
                    "owned_by": "openai"
                },
                {
                    "id": "claude-sonnet-4",
                    "display_name": "Claude Sonnet 4"
                }
            ]
        }"#;

        let resp: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
        assert_eq!(resp.data[0].name.as_deref(), Some("GPT-4o"));
        assert_eq!(resp.data[0].created_at, Some(1700000000));
        assert_eq!(resp.data[1].id, "claude-sonnet-4");
        assert_eq!(resp.data[1].name.as_deref(), Some("Claude Sonnet 4"));
    }

    #[test]
    fn deserialize_models_response_models_key() {
        let json = r#"{
            "models": [
                {"id": "gpt-4.1-mini"},
                {"id": "o3-mini", "name": "o3-mini"}
            ]
        }"#;

        let resp: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4.1-mini");
        assert_eq!(resp.data[1].id, "o3-mini");
    }

    #[test]
    fn model_entry_to_model() {
        let entry = ModelEntry {
            id: "gpt-4o".to_string(),
            name: Some("GPT-4o".to_string()),
            created_at: Some(1700000000),
            owned_by: Some("openai".to_string()),
        };
        let model: Model = entry.into();
        assert_eq!(model.id, "gpt-4o");
        assert_eq!(model.name.as_deref(), Some("GPT-4o"));
        assert_eq!(model.created_at, Some(1700000000));
        assert_eq!(model.owned_by.as_deref(), Some("openai"));
    }

    #[test]
    fn model_entry_without_name_uses_id() {
        let entry = ModelEntry {
            id: "gpt-4o".to_string(),
            name: None,
            created_at: None,
            owned_by: None,
        };
        let model: Model = entry.into();
        assert_eq!(model.name.as_deref(), Some("gpt-4o"));
    }
}
