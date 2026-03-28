//! GitHub Copilot client API implementation
use http::HeaderValue;

use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client::{self, HttpClientExt},
};

use super::embedding;
use super::model_listing;

// ================================================================
// Main GitHub Copilot Client
// ================================================================
const COPILOT_API_BASE_URL: &str = "https://api.individual.githubcopilot.com";

/// Required by the Copilot API — requests without this header are rejected.
const EDITOR_VERSION: &str = "vscode/1.96.2";
/// Required User-Agent for Copilot API requests.
const COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.26.7";

#[derive(Debug, Default, Clone, Copy)]
pub struct CopilotExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct CopilotExtBuilder;

type CopilotApiKey = BearerAuth;

impl Provider for CopilotExt {
    type Builder = CopilotExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for CopilotExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;
    type Embeddings = Capable<embedding::EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Capable<model_listing::CopilotModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for CopilotExt {}

impl ProviderBuilder for CopilotExtBuilder {
    type Extension<H>
        = CopilotExt
    where
        H: HttpClientExt;
    type ApiKey = CopilotApiKey;

    const BASE_URL: &'static str = COPILOT_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(CopilotExt)
    }

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, CopilotApiKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, CopilotApiKey, H>> {
        builder
            .headers_mut()
            .insert("Editor-Version", HeaderValue::from_static(EDITOR_VERSION));
        builder.headers_mut().insert(
            "User-Agent",
            HeaderValue::from_static(COPILOT_USER_AGENT),
        );
        Ok(builder)
    }
}

pub type Client<H = reqwest::Client> = client::Client<CopilotExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<CopilotExtBuilder, String, H>;

impl ProviderClient for Client {
    type Input = String;

    /// Create a new GitHub Copilot client from the `GITHUB_COPILOT_TOKEN` environment variable.
    ///
    /// This should be the short-lived Copilot API token obtained from
    /// `https://api.github.com/copilot_internal/v2/token`, **not** a GitHub OAuth token.
    ///
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key =
            std::env::var("GITHUB_COPILOT_TOKEN").expect("GITHUB_COPILOT_TOKEN not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

// ================================================================
// Token Exchange
// ================================================================

const COPILOT_TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// A short-lived Copilot API token obtained by exchanging a GitHub OAuth token.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CopilotToken {
    /// The Bearer token for Copilot API requests.
    pub token: String,
    /// Unix timestamp when this token expires.
    pub expires_at: u64,
}

/// Exchange a GitHub OAuth/personal-access token for a short-lived Copilot API token.
///
/// The returned [`CopilotToken::token`] can be passed to [`Client::new`].
///
/// # Errors
/// Returns an error if the HTTP request fails or the token exchange is rejected
/// (e.g. the user does not have an active Copilot subscription).
pub async fn exchange_github_token(github_token: &str) -> Result<CopilotToken, CopilotTokenError> {
    let client = reqwest::Client::new();
    let resp = client
        .get(COPILOT_TOKEN_EXCHANGE_URL)
        .header("Authorization", format!("token {github_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", COPILOT_USER_AGENT)
        .send()
        .await
        .map_err(CopilotTokenError::Http)?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(CopilotTokenError::Exchange(body));
    }

    resp.json::<CopilotToken>()
        .await
        .map_err(CopilotTokenError::Http)
}

/// Errors that can occur during Copilot token exchange.
#[derive(Debug, thiserror::Error)]
pub enum CopilotTokenError {
    #[error("HTTP request failed: {0}")]
    Http(reqwest::Error),
    #[error("Copilot token exchange failed: {0}")]
    Exchange(String),
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::github_copilot::Client::new("dummy-token")
            .expect("Client::new() failed");
        let _client_from_builder = crate::providers::github_copilot::Client::builder()
            .api_key("dummy-token")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn deserialize_copilot_token() {
        let json = r#"{"token":"ghu_abc123","expires_at":1711539600}"#;
        let token: super::CopilotToken = serde_json::from_str(json).unwrap();
        assert_eq!(token.token, "ghu_abc123");
        assert_eq!(token.expires_at, 1711539600);
    }
}
