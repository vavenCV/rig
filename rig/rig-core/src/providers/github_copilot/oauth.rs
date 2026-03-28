//! GitHub OAuth device flow for Copilot API authentication.
//!
//! Implements the [RFC 8628](https://tools.ietf.org/html/rfc8628) OAuth 2.0
//! Device Authorization Grant flow for GitHub, then exchanges the resulting
//! GitHub token for a short-lived Copilot API token.
//!
//! # Full authentication example
//! ```no_run
//! use rig::providers::github_copilot::oauth;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Step 1: Request a device code
//! let device = oauth::request_device_code(oauth::GITHUB_COPILOT_CLIENT_ID).await?;
//! println!("Go to {} and enter code: {}", device.verification_uri, device.user_code);
//!
//! // Step 2: Poll until the user authorises
//! let tokens = oauth::poll_for_token(
//!     oauth::GITHUB_COPILOT_CLIENT_ID,
//!     &device.device_code,
//!     device.interval,
//! ).await?;
//!
//! // Step 3: Exchange GitHub token for Copilot API token
//! let copilot = super::exchange_github_token(&tokens.access_token).await?;
//! let client = super::Client::new(&copilot.token);
//! # Ok(())
//! # }
//! ```

use serde::Deserialize;

/// VS Code's public OAuth client ID for GitHub Copilot.
pub const GITHUB_COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

/// Response from GitHub's device code endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    /// The device verification code.
    pub device_code: String,
    /// The code the user must enter at `verification_uri`.
    pub user_code: String,
    /// The URL the user should visit to enter the code.
    pub verification_uri: String,
    /// Polling interval in seconds (default 5).
    #[serde(default = "default_interval")]
    pub interval: u64,
    /// Number of seconds until the device code expires.
    #[serde(default)]
    pub expires_in: u64,
}

fn default_interval() -> u64 {
    5
}

/// Successful token response from the polling endpoint.
#[derive(Debug, Clone)]
pub struct TokenResponse {
    /// The GitHub OAuth access token.
    pub access_token: String,
}

/// Internal polling response — may contain an error or a token.
#[derive(Debug, Deserialize)]
struct TokenPollResponse {
    access_token: Option<String>,
    error: Option<String>,
}

/// Errors that can occur during the OAuth device flow.
#[derive(Debug, thiserror::Error)]
pub enum DeviceFlowError {
    /// HTTP transport error.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// The user denied the authorisation request.
    #[error("Access denied by user")]
    AccessDenied,
    /// The device code has expired before the user authorised.
    #[error("Device code expired — please restart the flow")]
    Expired,
    /// GitHub returned an unrecognised error.
    #[error("GitHub OAuth error: {0}")]
    Other(String),
}

/// Request a device code from GitHub for the given `client_id`.
///
/// The returned [`DeviceCodeResponse`] contains the `user_code` and
/// `verification_uri` that must be presented to the user.
pub async fn request_device_code(client_id: &str) -> Result<DeviceCodeResponse, DeviceFlowError> {
    let client = reqwest::Client::new();
    let body: String = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("client_id", client_id)
        .append_pair("scope", "")
        .finish();

    let resp = client
        .post(GITHUB_DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await?;

    let device: DeviceCodeResponse = resp.json().await?;
    Ok(device)
}

/// Poll GitHub's token endpoint until the user completes authorisation.
///
/// Respects the `interval` from the device code response and handles
/// `authorization_pending` and `slow_down` responses per RFC 8628.
pub async fn poll_for_token(
    client_id: &str,
    device_code: &str,
    interval: u64,
) -> Result<TokenResponse, DeviceFlowError> {
    let client = reqwest::Client::new();
    let mut poll_interval = std::time::Duration::from_secs(interval);

    loop {
        tokio::time::sleep(poll_interval).await;

        let body: String = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", client_id)
            .append_pair("device_code", device_code)
            .append_pair(
                "grant_type",
                "urn:ietf:params:oauth:grant-type:device_code",
            )
            .finish();

        let resp = client
            .post(GITHUB_TOKEN_URL)
            .header("Accept", "application/json")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await?;

        let poll: TokenPollResponse = resp.json().await?;

        if let Some(token) = poll.access_token {
            return Ok(TokenResponse {
                access_token: token,
            });
        }

        match poll.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                poll_interval += std::time::Duration::from_secs(5);
                continue;
            }
            Some("expired_token") => return Err(DeviceFlowError::Expired),
            Some("access_denied") => return Err(DeviceFlowError::AccessDenied),
            Some(other) => return Err(DeviceFlowError::Other(other.to_string())),
            None => continue,
        }
    }
}

/// Run the full device-flow authentication and Copilot token exchange.
///
/// 1. Requests a device code from GitHub.
/// 2. Prints instructions for the user to stdout.
/// 3. Polls until the user authorises.
/// 4. Exchanges the GitHub token for a Copilot API token.
///
/// Returns a [`super::CopilotToken`] ready for use with [`super::Client::new`].
pub async fn authenticate(
    client_id: &str,
) -> Result<super::CopilotToken, AuthenticateError> {
    let device = request_device_code(client_id).await?;
    println!(
        "Open {} and enter code: {}",
        device.verification_uri, device.user_code
    );

    let tokens = poll_for_token(client_id, &device.device_code, device.interval).await?;
    let copilot_token = super::exchange_github_token(&tokens.access_token).await?;
    Ok(copilot_token)
}

/// Errors from the combined authenticate flow.
#[derive(Debug, thiserror::Error)]
pub enum AuthenticateError {
    #[error("Device flow error: {0}")]
    DeviceFlow(#[from] DeviceFlowError),
    #[error("Copilot token exchange error: {0}")]
    TokenExchange(#[from] super::CopilotTokenError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_device_code_response() {
        let json = r#"{
            "device_code": "abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "interval": 5,
            "expires_in": 900
        }"#;
        let resp: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.device_code, "abc123");
        assert_eq!(resp.user_code, "ABCD-1234");
        assert_eq!(resp.verification_uri, "https://github.com/login/device");
        assert_eq!(resp.interval, 5);
        assert_eq!(resp.expires_in, 900);
    }

    #[test]
    fn deserialize_device_code_response_default_interval() {
        let json = r#"{
            "device_code": "abc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device"
        }"#;
        let resp: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.interval, 5);
        assert_eq!(resp.expires_in, 0);
    }

    #[test]
    fn deserialize_token_poll_pending() {
        let json = r#"{"error": "authorization_pending"}"#;
        let poll: TokenPollResponse = serde_json::from_str(json).unwrap();
        assert!(poll.access_token.is_none());
        assert_eq!(poll.error.as_deref(), Some("authorization_pending"));
    }

    #[test]
    fn deserialize_token_poll_success() {
        let json = r#"{"access_token": "gho_abc123"}"#;
        let poll: TokenPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(poll.access_token.as_deref(), Some("gho_abc123"));
        assert!(poll.error.is_none());
    }

    #[test]
    fn deserialize_token_poll_slow_down() {
        let json = r#"{"error": "slow_down"}"#;
        let poll: TokenPollResponse = serde_json::from_str(json).unwrap();
        assert!(poll.access_token.is_none());
        assert_eq!(poll.error.as_deref(), Some("slow_down"));
    }
}
