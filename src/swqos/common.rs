use crate::common::types::SolanaRpcClient;
use anyhow::Result;
use base64::engine::general_purpose::{self, STANDARD};
use base64::Engine;
use bincode::serialize;
use reqwest::Client;
use serde_json;
use serde_json::json;
use solana_client::rpc_client::SerializableTransaction;
use solana_client::rpc_config::RpcTransactionConfig;
use solana_sdk::signature::Signature;
use solana_sdk::transaction::VersionedTransaction;
use solana_sdk::transaction::{Transaction, TransactionError};
use solana_transaction_status::{TransactionConfirmationStatus, UiTransactionEncoding};
use std::str::FromStr;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct TradeError {
    pub code: u32,
    pub message: String,
    pub instruction: Option<u8>,
}

impl std::fmt::Display for TradeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TradeError {}

impl From<anyhow::Error> for TradeError {
    fn from(e: anyhow::Error) -> Self {
        if let Some(te) = e.downcast_ref::<TradeError>() {
            return te.clone();
        }
        TradeError { code: 500, message: format!("{}", e), instruction: None }
    }
}

// 使用高性能序列化

pub trait FormatBase64VersionedTransaction {
    fn to_base64_string(&self) -> String;
}

impl FormatBase64VersionedTransaction for VersionedTransaction {
    fn to_base64_string(&self) -> String {
        let tx_bytes = bincode::serialize(self).unwrap();
        general_purpose::STANDARD.encode(tx_bytes)
    }
}

pub async fn poll_transaction_confirmation(
    rpc: &SolanaRpcClient,
    txt_sig: Signature,
    wait_confirmation: bool,
) -> Result<Signature> {
    // 如果不需要等待确认，立即返回签名
    if !wait_confirmation {
        return Ok(txt_sig);
    }

    let timeout: Duration = Duration::from_secs(15); // 🔧 增加到15秒，避免网络拥堵时超时
    let interval: Duration = Duration::from_millis(1000);
    let start: Instant = Instant::now();
    let mut poll_count = 0u32;

    loop {
        if start.elapsed() >= timeout {
            return Err(anyhow::anyhow!("Transaction {}'s confirmation timed out", txt_sig));
        }

        poll_count += 1;

        let status = rpc.get_signature_statuses(&[txt_sig]).await?;
        match status.value[0].clone() {
            Some(status) => {
                if status.err.is_none()
                    && (status.confirmation_status
                        == Some(TransactionConfirmationStatus::Confirmed)
                        || status.confirmation_status
                            == Some(TransactionConfirmationStatus::Finalized))
                {
                    return Ok(txt_sig);
                }
                // 如果 getSignatureStatuses 返回了错误，立即获取详细信息
                if status.err.is_some() {
                    // 直接跳转到获取交易详情
                }
            }
            None => {
                // 交易还未上链，继续等待，不调用 getTransaction
                sleep(interval).await;
                continue;
            }
        }

        // 优化：只在以下情况调用 getTransaction
        // 1. getSignatureStatuses 返回了错误
        // 2. 或者已经轮询了较长时间（超过10次，即10秒）
        let should_get_transaction =
            status.value[0].as_ref().map(|s| s.err.is_some()).unwrap_or(false) || poll_count >= 10;

        if !should_get_transaction {
            sleep(interval).await;
            continue;
        }

        let tx_details = match rpc
            .get_transaction_with_config(
                &txt_sig,
                RpcTransactionConfig {
                    encoding: Some(UiTransactionEncoding::JsonParsed),
                    max_supported_transaction_version: Some(0),
                    commitment: Some(solana_commitment_config::CommitmentConfig::confirmed()),
                },
            )
            .await
        {
            Ok(details) => details,
            Err(_) => {
                // 交易可能还未上链，继续等待
                sleep(interval).await;
                continue;
            }
        };

        if let Some(meta) = tx_details.transaction.meta {
            if meta.err.is_none() {
                return Ok(txt_sig);
            } else {
                // 从 log_messages 中提取错误信息
                let mut error_msg = String::new();
                if let solana_transaction_status::option_serializer::OptionSerializer::Some(logs) =
                    &meta.log_messages
                {
                    for log in logs {
                        if let Some(idx) = log.find("Error Message: ") {
                            let msg = log[idx + 15..].trim_end_matches('.').to_string();
                            if !error_msg.is_empty() {
                                error_msg.push_str("; ");
                            }
                            error_msg.push_str(&msg);
                        } else if let Some(idx) = log.find("Program log: Error: ") {
                            let msg = log[idx + 20..].trim_end_matches('.').to_string();
                            if !error_msg.is_empty() {
                                error_msg.push_str("; ");
                            }
                            error_msg.push_str(&msg);
                        }
                    }
                }

                let ui_err = meta.err.unwrap();
                let tx_err: TransactionError =
                    serde_json::from_value(serde_json::to_value(&ui_err)?)?;

                // 直接使用Solana原生的InstructionError中的错误码
                let mut code = 0u32;
                let mut index = None;
                if let TransactionError::InstructionError(i, i_error) = &tx_err {
                    // 直接匹配所有InstructionError类型，Custom也是其中之一
                    code = match i_error {
                        solana_sdk::instruction::InstructionError::Custom(c) => *c,
                        solana_sdk::instruction::InstructionError::GenericError => 1,
                        solana_sdk::instruction::InstructionError::InvalidArgument => 2,
                        solana_sdk::instruction::InstructionError::InvalidInstructionData => 3,
                        solana_sdk::instruction::InstructionError::InvalidAccountData => 4,
                        solana_sdk::instruction::InstructionError::AccountDataTooSmall => 5,
                        solana_sdk::instruction::InstructionError::InsufficientFunds => 6,
                        solana_sdk::instruction::InstructionError::IncorrectProgramId => 7,
                        solana_sdk::instruction::InstructionError::MissingRequiredSignature => 8,
                        solana_sdk::instruction::InstructionError::AccountAlreadyInitialized => 9,
                        solana_sdk::instruction::InstructionError::UninitializedAccount => 10,
                        _ => 999, // 其他未知错误
                    };
                    index = Some(*i);
                }

                return Err(anyhow::Error::new(TradeError {
                    code,
                    message: format!("{} {:?}", tx_err, error_msg),
                    instruction: index,
                }));
            }
        } else {
            sleep(interval).await;
        }
    }
}

pub async fn send_nb_transaction(
    client: Client,
    endpoint: &str,
    auth_token: &str,
    transaction: &Transaction,
) -> Result<Signature, anyhow::Error> {
    // 序列化交易
    let serialized = bincode::serialize(transaction)
        .map_err(|e| anyhow::anyhow!("Transaction serialization failed: {}", e))?;

    // Base64编码
    let encoded = STANDARD.encode(serialized);

    let request_data = json!({
        "transaction": {
            "content": encoded
        },
        "frontRunningProtection": true
    });

    let url = format!("{}/api/v2/submit", endpoint);
    let response = client
        .post(url)
        .header("Authorization", auth_token)
        .header("Content-Type", "application/json")
        .json(&request_data)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Request failed: {}", e))?;

    let resp = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| anyhow::anyhow!("Response parsing failed: {}", e))?;

    if let Some(reason) = resp["reason"].as_str() {
        return Err(anyhow::anyhow!(reason.to_string()));
    }

    let signature = resp["signature"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing signature field in response"))?;

    let signature =
        Signature::from_str(signature).map_err(|e| anyhow::anyhow!("Invalid signature: {}", e))?;

    Ok(signature)
}

pub async fn serialize_and_encode(
    transaction: &[u8],
    encoding: UiTransactionEncoding,
) -> Result<String> {
    let serialized = match encoding {
        UiTransactionEncoding::Base58 => bs58::encode(transaction).into_string(),
        UiTransactionEncoding::Base64 => STANDARD.encode(transaction),
        _ => return Err(anyhow::anyhow!("Unsupported encoding")),
    };
    Ok(serialized)
}

pub async fn serialize_transaction_and_encode(
    transaction: &impl SerializableTransaction,
    encoding: UiTransactionEncoding,
) -> Result<(String, Signature)> {
    let signature = transaction.get_signature();
    let serialized_tx = serialize(transaction)?;
    let serialized = match encoding {
        UiTransactionEncoding::Base58 => bs58::encode(serialized_tx).into_string(),
        UiTransactionEncoding::Base64 => STANDARD.encode(serialized_tx),
        _ => return Err(anyhow::anyhow!("Unsupported encoding")),
    };
    Ok((serialized, *signature))
}

pub async fn serialize_smart_transaction_and_encode(
    transaction: &impl SerializableTransaction,
    encoding: UiTransactionEncoding,
) -> Result<(String, Signature)> {
    let signature = transaction.get_signature();
    let serialized_tx = serialize(transaction)?;
    let serialized = match encoding {
        UiTransactionEncoding::Base58 => bs58::encode(serialized_tx).into_string(),
        UiTransactionEncoding::Base64 => STANDARD.encode(serialized_tx),
        _ => return Err(anyhow::anyhow!("Unsupported encoding")),
    };
    Ok((serialized, *signature))
}
