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

pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_listing;
pub mod oauth;

pub use client::{Client, ClientBuilder, CopilotToken, CopilotTokenError, exchange_github_token};
pub use completion::{
    CompletionModel, StreamingCompletionResponse, CLAUDE_SONNET_4, GEMINI_2_0_FLASH, GPT_4O,
    GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO, O1, O1_MINI, O3_MINI,
};
pub use embedding::{TEXT_EMBEDDING_3_LARGE, TEXT_EMBEDDING_3_SMALL};

