//! 🚀 硬件级性能优化 - CPU缓存行对齐 & SIMD加速
//!
//! 实现CPU硬件特性的深度利用，包括：
//! - 缓存行对齐和缓存预取
//! - SIMD指令集优化
//! - 分支预测优化
//! - 内存屏障控制
//! - CPU指令流水线优化

use anyhow::Result;
use crossbeam_utils::CachePadded;
use std::mem::size_of;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

// CPU缓存行大小常量 (通常为64字节)
pub const CACHE_LINE_SIZE: usize = 64;

/// 🚀 硬件优化的数据结构基础特征
pub trait CacheLineAligned {
    /// 确保数据结构按缓存行对齐
    fn ensure_cache_aligned(&self) -> bool;
    /// 预取数据到CPU缓存
    fn prefetch_data(&self);
}

/// 🚀 SIMD优化的内存操作
pub struct SIMDMemoryOps;

impl SIMDMemoryOps {
    /// 🚀 SIMD加速的内存拷贝 - 针对小数据包优化
    ///
    /// # Safety
    ///
    /// This function is unsafe because it performs raw pointer manipulation and
    /// uses SIMD intrinsics. The caller must ensure that `dst` and `src` are
    /// valid for at least `len` bytes and are non-overlapping.
    #[inline(always)]
    pub unsafe fn memcpy_simd_optimized(dst: *mut u8, src: *const u8, len: usize) {
        match len {
            // 针对不同数据大小使用不同优化策略
            0 => (),
            1..=8 => Self::memcpy_small(dst, src, len),
            9..=16 => Self::memcpy_sse(dst, src, len),
            17..=32 => Self::memcpy_avx(dst, src, len),
            33..=64 => Self::memcpy_avx2(dst, src, len),
            _ => Self::memcpy_avx512_or_fallback(dst, src, len),
        }
    }

    /// 小数据拷贝优化 (1-8字节)
    #[inline(always)]
    unsafe fn memcpy_small(dst: *mut u8, src: *const u8, len: usize) {
        match len {
            1 => *dst = *src,
            2 => *(dst as *mut u16) = *(src as *const u16),
            3 => {
                *(dst as *mut u16) = *(src as *const u16);
                *dst.add(2) = *src.add(2);
            }
            4 => *(dst as *mut u32) = *(src as *const u32),
            5..=8 => {
                *(dst as *mut u64) = *(src as *const u64);
                if len > 8 {
                    ptr::copy_nonoverlapping(src.add(8), dst.add(8), len - 8);
                }
            }
            _ => unreachable!(),
        }
    }

    /// SSE优化拷贝 (9-16字节)
    #[inline(always)]
    unsafe fn memcpy_sse(dst: *mut u8, src: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_storeu_si128};

            if len <= 16 {
                let chunk = _mm_loadu_si128(src as *const __m128i);
                _mm_storeu_si128(dst as *mut __m128i, chunk);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            ptr::copy_nonoverlapping(src, dst, len);
        }
    }

    /// AVX优化拷贝 (17-32字节)
    #[inline(always)]
    unsafe fn memcpy_avx(dst: *mut u8, src: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

            if len <= 32 {
                let chunk = _mm256_loadu_si256(src as *const __m256i);
                _mm256_storeu_si256(dst as *mut __m256i, chunk);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            ptr::copy_nonoverlapping(src, dst, len);
        }
    }

    /// AVX2优化拷贝 (33-64字节)
    #[inline(always)]
    unsafe fn memcpy_avx2(dst: *mut u8, src: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256};

            // 拷贝前32字节
            let chunk1 = _mm256_loadu_si256(src as *const __m256i);
            _mm256_storeu_si256(dst as *mut __m256i, chunk1);

            if len > 32 {
                // 拷贝剩余字节
                let remaining = len - 32;
                if remaining <= 32 {
                    let chunk2 = _mm256_loadu_si256(src.add(32) as *const __m256i);
                    _mm256_storeu_si256(dst.add(32) as *mut __m256i, chunk2);
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            ptr::copy_nonoverlapping(src, dst, len);
        }
    }

    /// AVX512或回退拷贝 (>64字节)
    #[inline(always)]
    unsafe fn memcpy_avx512_or_fallback(dst: *mut u8, src: *const u8, len: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use std::arch::x86_64::{__m512i, _mm512_loadu_si512, _mm512_storeu_si512};

            let chunks = len / 64;
            let mut offset = 0;

            // 使用AVX512处理64字节块
            for _ in 0..chunks {
                let chunk = _mm512_loadu_si512(src.add(offset) as *const __m512i);
                _mm512_storeu_si512(dst.add(offset) as *mut __m512i, chunk);
                offset += 64;
            }

            // 处理剩余字节
            let remaining = len % 64;
            if remaining > 0 {
                Self::memcpy_avx2(dst.add(offset), src.add(offset), remaining);
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            // 回退到AVX2分块处理
            let chunks = len / 32;
            let mut offset = 0;

            for _ in 0..chunks {
                Self::memcpy_avx2(dst.add(offset), src.add(offset), 32);
                offset += 32;
            }

            let remaining = len % 32;
            if remaining > 0 {
                Self::memcpy_avx(dst.add(offset), src.add(offset), remaining);
            }
        }
    }

    /// 🚀 SIMD加速的内存比较
    ///
    /// # Safety
    ///
    /// This function is unsafe because it performs raw pointer manipulation and
    /// uses SIMD intrinsics. The caller must ensure that `a` and `b` are valid
    /// for at least `len` bytes.
    #[inline(always)]
    pub unsafe fn memcmp_simd_optimized(a: *const u8, b: *const u8, len: usize) -> bool {
        match len {
            0 => true,
            1..=8 => Self::memcmp_small(a, b, len),
            9..=16 => Self::memcmp_sse(a, b, len),
            17..=32 => Self::memcmp_avx2(a, b, len),
            _ => Self::memcmp_large(a, b, len),
        }
    }

    /// 小数据比较
    #[inline(always)]
    unsafe fn memcmp_small(a: *const u8, b: *const u8, len: usize) -> bool {
        match len {
            1 => *a == *b,
            2 => *(a as *const u16) == *(b as *const u16),
            3 => *(a as *const u16) == *(b as *const u16) && *a.add(2) == *b.add(2),
            4 => *(a as *const u32) == *(b as *const u32),
            5..=8 => *(a as *const u64) == *(b as *const u64),
            _ => unreachable!(),
        }
    }

    /// SSE比较
    #[inline(always)]
    unsafe fn memcmp_sse(a: *const u8, b: *const u8, len: usize) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};

            let chunk_a = _mm_loadu_si128(a as *const __m128i);
            let chunk_b = _mm_loadu_si128(b as *const __m128i);
            let cmp_result = _mm_cmpeq_epi8(chunk_a, chunk_b);
            let mask = _mm_movemask_epi8(cmp_result) as u32;

            // 检查前len字节是否相等
            let valid_mask = if len >= 16 { 0xFFFF } else { (1u32 << len) - 1 };
            (mask & valid_mask) == valid_mask
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            (0..len).all(|i| *a.add(i) == *b.add(i))
        }
    }

    /// AVX2比较
    #[inline(always)]
    unsafe fn memcmp_avx2(a: *const u8, b: *const u8, len: usize) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{
                __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
            };

            let chunk_a = _mm256_loadu_si256(a as *const __m256i);
            let chunk_b = _mm256_loadu_si256(b as *const __m256i);
            let cmp_result = _mm256_cmpeq_epi8(chunk_a, chunk_b);
            let mask = _mm256_movemask_epi8(cmp_result) as u32;

            let valid_mask = if len >= 32 { 0xFFFFFFFF } else { (1u32 << len) - 1 };
            (mask & valid_mask) == valid_mask
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            (0..len).all(|i| *a.add(i) == *b.add(i))
        }
    }

    /// 大数据比较
    #[inline(always)]
    unsafe fn memcmp_large(a: *const u8, b: *const u8, len: usize) -> bool {
        let chunks = len / 32;

        for i in 0..chunks {
            let offset = i * 32;
            if !Self::memcmp_avx2(a.add(offset), b.add(offset), 32) {
                return false;
            }
        }

        let remaining = len % 32;
        if remaining > 0 {
            return Self::memcmp_avx2(a.add(chunks * 32), b.add(chunks * 32), remaining);
        }

        true
    }

    /// 🚀 SIMD加速的内存清零
    ///
    /// # Safety
    ///
    /// This function is unsafe because it performs raw pointer manipulation and
    /// uses SIMD intrinsics. The caller must ensure that `ptr` is valid for
    /// at least `len` bytes.
    #[inline(always)]
    pub unsafe fn memzero_simd_optimized(ptr: *mut u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__m256i, _mm256_setzero_si256, _mm256_storeu_si256};

            let zero = _mm256_setzero_si256();
            let chunks = len / 32;
            let mut offset = 0;

            for _ in 0..chunks {
                _mm256_storeu_si256(ptr.add(offset) as *mut __m256i, zero);
                offset += 32;
            }

            // 处理剩余字节
            let remaining = len % 32;
            for i in 0..remaining {
                *ptr.add(offset + i) = 0;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            ptr::write_bytes(ptr, 0, len);
        }
    }
}

/// 🚀 缓存行对齐的原子计数器
#[repr(align(64))] // 强制64字节对齐
pub struct CacheAlignedCounter {
    value: AtomicU64,
    _padding: [u8; CACHE_LINE_SIZE - size_of::<AtomicU64>()],
}

impl CacheAlignedCounter {
    pub fn new(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
            _padding: [0; CACHE_LINE_SIZE - size_of::<AtomicU64>()],
        }
    }

    #[inline(always)]
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn load(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn store(&self, val: u64) {
        self.value.store(val, Ordering::Relaxed)
    }
}

impl CacheLineAligned for CacheAlignedCounter {
    fn ensure_cache_aligned(&self) -> bool {
        (self as *const Self as usize).is_multiple_of(CACHE_LINE_SIZE)
    }

    fn prefetch_data(&self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            _mm_prefetch(self as *const Self as *const i8, _MM_HINT_T0);
        }
    }
}

/// 🚀 缓存友好的环形缓冲区
#[repr(align(64))]
pub struct CacheOptimizedRingBuffer<T> {
    /// 数据缓冲区
    buffer: Vec<T>,
    /// 生产者头指针 (独占缓存行)
    producer_head: CachePadded<AtomicU64>,
    /// 消费者尾指针 (独占缓存行)
    consumer_tail: CachePadded<AtomicU64>,
    /// 容量 (2的幂次方)
    capacity: usize,
    /// 掩码 (capacity - 1)
    mask: usize,
}

impl<T: Copy + Default> CacheOptimizedRingBuffer<T> {
    /// 创建缓存优化的环形缓冲区
    pub fn new(capacity: usize) -> Result<Self> {
        if !capacity.is_power_of_two() {
            return Err(anyhow::anyhow!("Capacity must be a power of 2"));
        }

        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, Default::default);

        Ok(Self {
            buffer,
            producer_head: CachePadded::new(AtomicU64::new(0)),
            consumer_tail: CachePadded::new(AtomicU64::new(0)),
            capacity,
            mask: capacity - 1,
        })
    }

    /// 🚀 无锁写入元素
    #[inline(always)]
    pub fn try_push(&self, item: T) -> bool {
        let current_head = self.producer_head.load(Ordering::Relaxed);
        let current_tail = self.consumer_tail.load(Ordering::Acquire);

        // 检查是否还有空间
        if (current_head + 1) & self.mask as u64 == current_tail & self.mask as u64 {
            return false; // 缓冲区满
        }

        // 写入数据
        unsafe {
            let index = current_head & self.mask as u64;
            let ptr = self.buffer.as_ptr().add(index as usize) as *mut T;
            ptr.write(item);
        }

        // 发布新的头指针
        self.producer_head.store(current_head + 1, Ordering::Release);
        true
    }

    /// 🚀 无锁读取元素
    #[inline(always)]
    pub fn try_pop(&self) -> Option<T> {
        let current_tail = self.consumer_tail.load(Ordering::Relaxed);
        let current_head = self.producer_head.load(Ordering::Acquire);

        // 检查是否有数据
        if current_tail == current_head {
            return None; // 缓冲区空
        }

        // 读取数据
        let item = unsafe {
            let index = current_tail & self.mask as u64;
            let ptr = self.buffer.as_ptr().add(index as usize);
            ptr.read()
        };

        // 发布新的尾指针
        self.consumer_tail.store(current_tail + 1, Ordering::Release);
        Some(item)
    }

    /// 获取当前元素数量
    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.producer_head.load(Ordering::Relaxed);
        let tail = self.consumer_tail.load(Ordering::Relaxed);
        ((head + self.capacity as u64 - tail) & self.mask as u64) as usize
    }

    /// 检查是否为空
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.producer_head.load(Ordering::Relaxed) == self.consumer_tail.load(Ordering::Relaxed)
    }
}

impl<T> CacheLineAligned for CacheOptimizedRingBuffer<T> {
    fn ensure_cache_aligned(&self) -> bool {
        (self as *const Self as usize).is_multiple_of(CACHE_LINE_SIZE)
    }

    fn prefetch_data(&self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;

            // 预取头指针
            _mm_prefetch(self.producer_head.as_ptr() as *const i8, _MM_HINT_T0);

            // 预取尾指针
            _mm_prefetch(self.consumer_tail.as_ptr() as *const i8, _MM_HINT_T0);

            // 预取缓冲区开始位置
            _mm_prefetch(self.buffer.as_ptr() as *const i8, _MM_HINT_T0);
        }
    }
}

/// 🚀 CPU分支预测优化工具
pub struct BranchOptimizer;

impl BranchOptimizer {
    /// likely宏 - 告诉编译器条件大概率为真
    #[inline(always)]
    pub fn likely(condition: bool) -> bool {
        #[cold]
        fn cold() {}

        if !condition {
            cold();
        }
        condition
    }

    /// unlikely宏 - 告诉编译器条件大概率为假
    #[inline(always)]
    pub fn unlikely(condition: bool) -> bool {
        #[cold]
        fn cold() {}

        if condition {
            cold();
        }
        condition
    }

    /// 预取指令 - 提前加载数据到缓存
    ///
    /// # Safety
    ///
    /// This function is unsafe because it uses processor-specific intrinsics
    /// for memory prefetching. The caller must ensure `ptr` points to valid memory.
    #[inline(always)]
    pub unsafe fn prefetch_read_data<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T0;
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    /// 预取指令 - 提前加载数据到缓存（写优化）
    ///
    /// # Safety
    ///
    /// This function is unsafe because it uses processor-specific intrinsics
    /// for memory prefetching. The caller must ensure `ptr` points to valid memory.
    #[inline(always)]
    pub unsafe fn prefetch_write_data<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_mm_prefetch;
            use std::arch::x86_64::_MM_HINT_T1;
            _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
        }
    }
}

/// 🚀 内存屏障控制
pub struct MemoryBarriers;

impl MemoryBarriers {
    /// 编译器屏障 - 防止编译器重排序
    #[inline(always)]
    pub fn compiler_barrier() {
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }

    /// 轻量级内存屏障 - 仅CPU重排序保护
    #[inline(always)]
    pub fn memory_barrier_light() {
        std::sync::atomic::fence(Ordering::Acquire);
    }

    /// 重量级内存屏障 - 全序一致性
    #[inline(always)]
    pub fn memory_barrier_heavy() {
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// 存储屏障 - 确保写入可见性
    #[inline(always)]
    pub fn store_barrier() {
        std::sync::atomic::fence(Ordering::Release);
    }

    /// 加载屏障 - 确保读取正确性
    #[inline(always)]
    pub fn load_barrier() {
        std::sync::atomic::fence(Ordering::Acquire);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_counter() {
        let counter = CacheAlignedCounter::new(0);
        assert!(counter.ensure_cache_aligned());

        assert_eq!(counter.load(), 0);
        counter.increment();
        assert_eq!(counter.load(), 1);
    }

    #[test]
    fn test_simd_memcpy() {
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut dst = [0u8; 10];

        unsafe {
            SIMDMemoryOps::memcpy_simd_optimized(dst.as_mut_ptr(), src.as_ptr(), src.len());
        }

        assert_eq!(src, dst);
    }

    #[test]
    fn test_cache_optimized_ring_buffer() {
        let buffer: CacheOptimizedRingBuffer<u64> = CacheOptimizedRingBuffer::new(16).unwrap();

        assert!(buffer.is_empty());

        // 测试推入
        assert!(buffer.try_push(42));
        assert_eq!(buffer.len(), 1);

        // 测试弹出
        assert_eq!(buffer.try_pop(), Some(42));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_simd_memcmp() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        let c = [1u8, 2, 3, 4, 6];

        unsafe {
            assert!(SIMDMemoryOps::memcmp_simd_optimized(a.as_ptr(), b.as_ptr(), a.len()));

            assert!(!SIMDMemoryOps::memcmp_simd_optimized(a.as_ptr(), c.as_ptr(), a.len()));
        }
    }
}
