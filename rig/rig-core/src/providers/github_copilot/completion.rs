//! GitHub Copilot completion API implementation

use std::collections::HashMap;

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use tracing::info_span;
use tracing_futures::Instrument;

use crate::completion::GetTokenUsage;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::empty_or_none;
use crate::providers::openai::{
    AssistantContent, CompletionResponse, Function, Message as OpenAIMessage, StreamingToolCall,
    ToolDefinition, ToolType, Usage,
};
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
};

use super::client::Client;

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
// Completion Request
// ================================================================

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CopilotCompletionRequest {
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
                tracing::debug!(
                    target: "rig::completions",
                    "GitHub Copilot raw response body: {}",
                    String::from_utf8_lossy(&response_body)
                );

                // Copilot responses omit "object" and "created" fields that the
                // OpenAI CompletionResponse struct requires — inject defaults.
                let patched = {
                    let mut json: serde_json::Value =
                        serde_json::from_slice(&response_body)?;
                    if let Some(obj) = json.as_object_mut() {
                        obj.entry("object").or_insert_with(|| {
                            serde_json::Value::String("chat.completion".to_string())
                        });
                        obj.entry("created")
                            .or_insert_with(|| serde_json::Value::Number(0.into()));
                    }
                    json
                };

                match serde_json::from_value::<CompletionResponse>(patched) {
                    Ok(response) => {
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
                    Err(e) => {
                        // Fall back: check if it's an API error response
                        if let Ok(err_resp) =
                            serde_json::from_slice::<ApiErrorResponse>(&response_body)
                        {
                            Err(CompletionError::ProviderError(err_resp.message))
                        } else {
                            Err(CompletionError::ProviderError(format!(
                                "Failed to parse Copilot response: {e}. Body: {}",
                                String::from_utf8_lossy(&response_body)
                            )))
                        }
                    }
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
}
