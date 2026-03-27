//! GitHub Copilot API client and Rig integration
//!
//! The Copilot API is OpenAI-compatible (`/chat/completions`) and requires a
//! short-lived Copilot API token obtained through a two-step flow:
//!
//! 1. Authenticate with GitHub via OAuth device flow to get a GitHub token.
//! 2. Exchange the GitHub token for a Copilot API token via
//!    `https://api.github.com/copilot_internal/v2/token`.
//!
//! Pass the resulting Copilot API token to [`Client::new`] (or set the
//! `GITHUB_COPILOT_TOKEN` environment variable). A helper
//! [`exchange_github_token`] is provided for step 2.
//!
//! # Example
//! ```no_run
//! use rig::providers::github_copilot;
//!
//! // Using a pre-obtained Copilot API token
//! let client = github_copilot::Client::new("COPILOT_API_TOKEN");
//!
//! let model = client.completion_model(github_copilot::GPT_4O);
//! ```
//!
//! # Token exchange example
//! ```no_run
//! use rig::providers::github_copilot;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Exchange a GitHub OAuth token for a Copilot API token
//! let copilot_token = github_copilot::exchange_github_token("gho_your_github_token").await?;
//! let client = github_copilot::Client::new(&copilot_token.token);
//! let model = client.completion_model(github_copilot::GPT_4O);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use http::{HeaderValue, Request};
use serde::{Deserialize, Serialize};
use tracing::info_span;
use tracing_futures::Instrument;

use super::openai::{
    AssistantContent, CompletionResponse, Function, Message as OpenAIMessage, StreamingToolCall,
    ToolType, Usage,
};
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::GetTokenUsage;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::empty_or_none;
use crate::providers::openai::ToolDefinition;
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
};

// ================================================================
// Main GitHub Copilot Client
// ================================================================
const COPILOT_API_BASE_URL: &str = "https://api.individual.githubcopilot.com";
const COPILOT_TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";

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
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
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

/// A short-lived Copilot API token obtained by exchanging a GitHub OAuth token.
#[derive(Debug, Clone, Deserialize)]
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
        .header(
            "Authorization",
            format!("token {github_token}"),
        )
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

// ================================================================
// API Error Types
// ================================================================

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

// ================================================================
// Completion Model Constants
// ================================================================

/// `gpt-4o` — GPT-4o via GitHub Copilot.
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4.1` — GPT-4.1 via GitHub Copilot.
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.1-mini` — GPT-4.1 Mini via GitHub Copilot.
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano` — GPT-4.1 Nano via GitHub Copilot.
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `o1` — o1 reasoning model via GitHub Copilot.
pub const O1: &str = "o1";
/// `o1-mini` — o1-mini reasoning model via GitHub Copilot.
pub const O1_MINI: &str = "o1-mini";
/// `o3-mini` — o3-mini reasoning model via GitHub Copilot.
pub const O3_MINI: &str = "o3-mini";
/// `claude-sonnet-4` — Claude Sonnet 4 via GitHub Copilot.
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `gemini-2.0-flash` — Gemini 2.0 Flash via GitHub Copilot.
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

// ================================================================
// Completion Request
// ================================================================

#[derive(Debug, Serialize, Deserialize)]
struct CopilotCompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct StreamOptions {
    include_usage: bool,
}

impl TryFrom<(&str, CompletionRequest)> for CopilotCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let model = req.model.clone().unwrap_or_else(|| model.to_string());

        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        let mut full_history: Vec<OpenAIMessage> = match &req.preamble {
            Some(preamble) => vec![OpenAIMessage::system(preamble)],
            None => vec![],
        };

        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<OpenAIMessage>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openai::ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model,
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
            stream: false,
            stream_options: None,
        })
    }
}

// ================================================================
// Completion Model
// ================================================================

#[derive(Clone, Debug)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "github-copilot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request =
            CopilotCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "GitHub Copilot completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        let async_block = async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.clone());
                        span.record("gen_ai.response.model_name", response.model.clone());
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                            span.record(
                                "gen_ai.usage.cached_tokens",
                                usage
                                    .prompt_tokens_details
                                    .as_ref()
                                    .map(|d| d.cached_tokens)
                                    .unwrap_or(0),
                            );
                        }

                        if tracing::enabled!(tracing::Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "GitHub Copilot completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "github-copilot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &request.preamble);

        let mut request =
            CopilotCompletionRequest::try_from((self.model.as_ref(), request))?;

        request.stream = true;
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "GitHub Copilot streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

// ================================================================
// Streaming
// ================================================================

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
        usage.cached_input_tokens = self
            .usage
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens as u64)
            .unwrap_or(0);

        Some(usage)
    }
}

async fn send_compatible_streaming_request<T>(
    client: T,
    req: Request<Vec<u8>>,
) -> Result<
    crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
    CompletionError,
>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();

    let mut event_source = GenericEventSource::new(client, req);

    let stream = stream! {
        let span = tracing::Span::current();
        let mut final_usage = Usage {
            prompt_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
        };

        let mut text_response = String::new();
        let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }

                Ok(Event::Message(message)) => {
                    let data_str = message.data.trim();

                    let parsed = serde_json::from_str::<StreamingCompletionChunk>(data_str);
                    let Ok(data) = parsed else {
                        let err = parsed.unwrap_err();
                        tracing::debug!(
                            "Couldn't parse SSE payload as StreamingCompletionChunk: {:?}",
                            err
                        );
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        let delta = &choice.delta;

                        // Handle tool calls
                        for tool_call in &delta.tool_calls {
                            let function = &tool_call.function;

                            // Start of tool call
                            if function.name.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
                                && empty_or_none(&function.arguments)
                            {
                                let id = tool_call.id.clone().unwrap_or_default();
                                let name = function.name.clone().unwrap();
                                calls.insert(tool_call.index, (id, name, String::new()));
                            }
                            // Continuation
                            else if function
                                .name
                                .as_ref()
                                .map(|s| s.is_empty())
                                .unwrap_or(true)
                                && let Some(arguments) = &function.arguments
                                && !arguments.is_empty()
                            {
                                if let Some((id, name, existing_args)) =
                                    calls.get(&tool_call.index)
                                {
                                    let combined =
                                        format!("{}{}", existing_args, arguments);
                                    calls.insert(
                                        tool_call.index,
                                        (id.clone(), name.clone(), combined),
                                    );
                                } else {
                                    tracing::debug!(
                                        "Partial tool call received but tool call was never started."
                                    );
                                }
                            }
                            // Complete tool call
                            else {
                                let id = tool_call.id.clone().unwrap_or_default();
                                let name = function.name.clone().unwrap_or_default();
                                let arguments_str =
                                    function.arguments.clone().unwrap_or_default();

                                let Ok(arguments_json) =
                                    serde_json::from_str::<serde_json::Value>(&arguments_str)
                                else {
                                    tracing::debug!(
                                        "Couldn't parse tool call args '{}'",
                                        arguments_str
                                    );
                                    continue;
                                };

                                yield Ok(crate::streaming::RawStreamingChoice::ToolCall(
                                    crate::streaming::RawStreamingToolCall::new(
                                        id,
                                        name,
                                        arguments_json,
                                    ),
                                ));
                            }
                        }

                        // Streamed content
                        if let Some(content) = &delta.content {
                            text_response += content;
                            yield Ok(crate::streaming::RawStreamingChoice::Message(
                                content.clone(),
                            ));
                        }
                    }

                    if let Some(usage) = data.usage {
                        final_usage = usage.clone();
                    }
                }

                Err(crate::http_client::Error::StreamEnded) => break,
                Err(err) => {
                    tracing::error!(?err, "SSE error");
                    yield Err(CompletionError::ResponseError(err.to_string()));
                    break;
                }
            }
        }

        event_source.close();

        let mut tool_calls = Vec::new();
        // Flush accumulated tool calls
        for (_, (id, name, arguments)) in calls {
            let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments) else {
                continue;
            };

            tool_calls.push(crate::providers::openai::completion::ToolCall {
                id: id.clone(),
                r#type: ToolType::Function,
                function: Function {
                    name: name.clone(),
                    arguments: arguments_json.clone(),
                },
            });
            yield Ok(crate::streaming::RawStreamingChoice::ToolCall(
                crate::streaming::RawStreamingToolCall::new(id, name, arguments_json),
            ));
        }

        let response_message = crate::providers::openai::completion::Message::Assistant {
            content: vec![AssistantContent::Text {
                text: text_response,
            }],
            refusal: None,
            audio: None,
            name: None,
            tool_calls,
        };

        span.record(
            "gen_ai.output.messages",
            serde_json::to_string(&vec![response_message]).unwrap(),
        );
        span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
        span.record(
            "gen_ai.usage.output_tokens",
            final_usage.total_tokens - final_usage.prompt_tokens,
        );
        span.record(
            "gen_ai.usage.cached_tokens",
            final_usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
        );

        // Final response
        yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse {
                usage: final_usage.clone(),
            },
        ));
    }
    .instrument(span);

    Ok(crate::streaming::StreamingCompletionResponse::stream(
        Box::pin(stream),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        OneOrMany,
        providers::openai::{Message, UserContent},
    };

    #[test]
    fn serialize_copilot_request() {
        let request = CopilotCompletionRequest {
            model: "gpt-4o".to_string(),
            temperature: Some(0.7),
            tool_choice: None,
            stream_options: None,
            tools: Vec::new(),
            messages: vec![Message::User {
                content: OneOrMany::one(UserContent::Text {
                    text: "Hello!".to_string(),
                }),
                name: None,
            }],
            stream: false,
            additional_params: None,
        };

        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(
            json,
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!"
                    }
                ],
                "temperature": 0.7,
                "stream": false
            })
        );
    }

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
        let token: CopilotToken = serde_json::from_str(json).unwrap();
        assert_eq!(token.token, "ghu_abc123");
        assert_eq!(token.expires_at, 1711539600);
    }
}
