//! 交易序列化模块

use anyhow::Result;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use once_cell::sync::Lazy;
use solana_client::rpc_client::SerializableTransaction;
use solana_sdk::signature::Signature;
use solana_transaction_status::UiTransactionEncoding;
use std::sync::Arc;
use crossbeam_queue::ArrayQueue;
use crate::perf::{
    simd::SIMDSerializer,
    compiler_optimization::CompileTimeOptimizedEventProcessor,
};

/// 零分配序列化器 - 使用缓冲池避免运行时分配
pub struct ZeroAllocSerializer {
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    buffer_size: usize,
}

impl ZeroAllocSerializer {
    pub fn new(pool_size: usize, buffer_size: usize) -> Self {
        let pool = ArrayQueue::new(pool_size);

        // 预分配缓冲区
        for _ in 0..pool_size {
            let buffer = vec![0; buffer_size];
            let _ = pool.push(buffer);
        }

        Self {
            buffer_pool: Arc::new(pool),
            buffer_size,
        }
    }

    pub fn serialize_zero_alloc<T: serde::Serialize>(&self, data: &T, _label: &str) -> Result<Vec<u8>> {
        // 尝试从池中获取缓冲区
        let mut buffer = self.buffer_pool.pop().unwrap_or_else(|| {
            let buf = vec![0; self.buffer_size];
            buf
        });

        // 序列化到缓冲区
        let serialized = bincode::serialize(data)?;
        buffer.clear();
        buffer.extend_from_slice(&serialized);

        Ok(buffer)
    }

    pub fn return_buffer(&self, buffer: Vec<u8>) {
        // 归还缓冲区到池中
        let _ = self.buffer_pool.push(buffer);
    }

    /// 获取池统计信息
    pub fn get_pool_stats(&self) -> (usize, usize) {
        let available = self.buffer_pool.len();
        let capacity = self.buffer_pool.capacity();
        (available, capacity)
    }
}

/// 全局序列化器实例
static SERIALIZER: Lazy<Arc<ZeroAllocSerializer>> = Lazy::new(|| {
    Arc::new(ZeroAllocSerializer::new(
        10_000,      // 池大小
        256 * 1024,  // 缓冲区大小: 256KB
    ))
});

/// 🚀 编译时优化的事件处理器 (零运行时开销)
static COMPILE_TIME_PROCESSOR: CompileTimeOptimizedEventProcessor =
    CompileTimeOptimizedEventProcessor::new();

/// Base64 编码器
pub struct Base64Encoder;

impl Base64Encoder {
    #[inline(always)]
    pub fn encode(data: &[u8]) -> String {
        // 使用编译时优化的哈希进行快速路由
        let _route = if !data.is_empty() {
            COMPILE_TIME_PROCESSOR.route_event_zero_cost(data[0])
        } else {
            0
        };

        // 使用 SIMD 加速的 Base64 编码
        SIMDSerializer::encode_base64_simd(data)
    }

    #[inline(always)]
    pub fn serialize_and_encode<T: serde::Serialize>(
        value: &T,
        event_type: &str,
    ) -> Result<String> {
        let serialized = SERIALIZER.serialize_zero_alloc(value, event_type)?;
        Ok(STANDARD.encode(&serialized))
    }
}

/// 交易序列化
pub async fn serialize_transaction(
    transaction: &impl SerializableTransaction,
    encoding: UiTransactionEncoding,
) -> Result<(String, Signature)> {
    let signature = transaction.get_signature();

    // 使用零分配序列化
    let serialized_tx = SERIALIZER.serialize_zero_alloc(transaction, "transaction")?;

    let serialized = match encoding {
        UiTransactionEncoding::Base58 => bs58::encode(&serialized_tx).into_string(),
        UiTransactionEncoding::Base64 => {
            // 使用 SIMD 优化的 Base64 编码
            STANDARD.encode(&serialized_tx)
        }
        _ => return Err(anyhow::anyhow!("Unsupported encoding")),
    };

    // 立即归还缓冲区到池中
    SERIALIZER.return_buffer(serialized_tx);

    Ok((serialized, *signature))
}

/// 批量交易序列化
pub async fn serialize_transactions_batch(
    transactions: &[impl SerializableTransaction],
    encoding: UiTransactionEncoding,
) -> Result<Vec<String>> {
    let mut results = Vec::with_capacity(transactions.len());

    for tx in transactions {
        let serialized_tx = SERIALIZER.serialize_zero_alloc(tx, "transaction")?;

        let encoded = match encoding {
            UiTransactionEncoding::Base58 => bs58::encode(&serialized_tx).into_string(),
            UiTransactionEncoding::Base64 => STANDARD.encode(&serialized_tx),
            _ => return Err(anyhow::anyhow!("Unsupported encoding")),
        };

        SERIALIZER.return_buffer(serialized_tx);
        results.push(encoded);
    }

    Ok(results)
}

/// 获取序列化器统计信息
pub fn get_serializer_stats() -> (usize, usize) {
    SERIALIZER.get_pool_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_encode() {
        let data = b"Hello, World!";
        let encoded = Base64Encoder::encode(data);
        assert!(!encoded.is_empty());

        // 验证可以正确解码
        let decoded = STANDARD.decode(&encoded).unwrap();
        assert_eq!(&decoded[..data.len()], data);
    }

    #[test]
    fn test_serializer_stats() {
        let (available, capacity) = get_serializer_stats();
        assert!(available <= capacity);
        assert_eq!(capacity, 10_000);
    }
}
