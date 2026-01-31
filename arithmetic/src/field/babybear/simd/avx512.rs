//! AVX512 SIMD implementation for PackedBabyBear
//! Processes 16 BabyBear elements in parallel using 512-bit registers

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::PackedField;
use crate::field::babybear::BabyBear;

/// Packed BabyBear using AVX512 (16 elements parallel)
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct PackedBabyBearAVX512(pub [BabyBear; 16]);

impl Default for PackedBabyBearAVX512 {
    fn default() -> Self {
        Self([BabyBear::default(); 16])
    }
}

impl PartialEq for PackedBabyBearAVX512 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedBabyBearAVX512 {}

impl PackedBabyBearAVX512 {
    pub const WIDTH: usize = 16;
    
    /// Convert to AVX512 vector
    #[inline]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    pub(crate) fn to_vector(self) -> __m512i {
        unsafe { transmute(self.0) }
    }
    
    /// Create from AVX512 vector
    #[inline]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    pub(crate) unsafe fn from_vector(vector: __m512i) -> Self {
        Self(transmute(vector))
    }
    
    /// Create a packed value with all zeros
    #[inline]
    pub fn zero() -> Self {
        Self([BabyBear::default(); 16])
    }
    
    /// Create a packed value with all ones
    #[inline]
    pub fn one() -> Self {
        Self([BabyBear::from(1u32); 16])
    }
}

impl PackedField for PackedBabyBearAVX512 {
    type Scalar = BabyBear;
    const WIDTH: usize = 16;
    
    #[inline]
    fn from_slice(slice: &[BabyBear]) -> Self {
        assert!(slice.len() >= 16, "slice too short");
        let mut arr = [BabyBear::default(); 16];
        arr.copy_from_slice(&slice[..16]);
        Self(arr)
    }
    
    #[inline]
    fn to_array(&self) -> Vec<BabyBear> {
        self.0.to_vec()
    }
    
    #[inline]
    fn store(&self, dst: &mut [BabyBear]) {
        dst[..16].copy_from_slice(&self.0);
    }
    
    #[inline]
    fn broadcast(val: BabyBear) -> Self {
        Self([val; 16])
    }
    
    #[inline]
    fn from_fn<F: Fn(usize) -> BabyBear>(f: F) -> Self {
        Self([
            f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7),
            f(8), f(9), f(10), f(11), f(12), f(13), f(14), f(15),
        ])
    }
}

// Scalar fallback implementations for non-AVX512 targets
impl Neg for PackedBabyBearAVX512 {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        Self(std::array::from_fn(|i| -self.0[i]))
    }
}

impl Add for PackedBabyBearAVX512 {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl AddAssign for PackedBabyBearAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..16 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Sub for PackedBabyBearAVX512 {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl SubAssign for PackedBabyBearAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..16 {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Mul for PackedBabyBearAVX512 {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
}

impl MulAssign for PackedBabyBearAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..16 {
            self.0[i] *= rhs.0[i];
        }
    }
}
