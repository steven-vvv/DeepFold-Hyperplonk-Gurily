//! SIMD-optimized packed BabyBear field operations
//!
//! This module provides platform-specific SIMD implementations:
//! - AVX512: 16 elements parallel (512-bit)
//! - AVX2: 8 elements parallel (256-bit)
//! - NEON: 4 elements parallel (128-bit)
//! - Fallback: scalar operations

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::PackedBabyBearAVX512 as PackedBabyBear;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use avx2::PackedBabyBearAVX2 as PackedBabyBear;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::PackedBabyBearNeon as PackedBabyBear;

// Fallback module is always available for testing and comparison
mod fallback;
pub use fallback::PackedBabyBearScalar;

// Use fallback as PackedBabyBear when no SIMD is available
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
pub use fallback::PackedBabyBearScalar as PackedBabyBear;

/// Trait for packed field operations
pub trait PackedField: Sized + Copy + Clone + Default {
    type Scalar;
    
    /// Number of elements in the packed representation
    const WIDTH: usize;
    
    /// Create a packed value from a slice
    fn from_slice(slice: &[Self::Scalar]) -> Self;
    
    /// Convert to an array
    fn to_array(&self) -> Vec<Self::Scalar>;
    
    /// Store into a mutable slice
    fn store(&self, dst: &mut [Self::Scalar]);
    
    /// Broadcast a single value to all lanes
    fn broadcast(val: Self::Scalar) -> Self;
    
    /// Create from a function that maps index to value
    fn from_fn<F: Fn(usize) -> Self::Scalar>(f: F) -> Self;
}
