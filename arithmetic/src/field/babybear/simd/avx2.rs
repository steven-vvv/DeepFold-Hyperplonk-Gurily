//! AVX2 SIMD implementation for PackedBabyBear
//! Processes 8 BabyBear elements in parallel using 256-bit registers

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::PackedField;
use crate::field::babybear::BabyBear;
use crate::field::babybear::params::{PRIME, MONTY_MU};

/// Packed BabyBear using AVX2 (8 elements parallel)
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct PackedBabyBearAVX2(pub [BabyBear; 8]);

impl Default for PackedBabyBearAVX2 {
    fn default() -> Self {
        Self([BabyBear::default(); 8])
    }
}

impl PartialEq for PackedBabyBearAVX2 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedBabyBearAVX2 {}

impl PackedBabyBearAVX2 {
    pub const WIDTH: usize = 8;
    
    /// Convert to AVX2 vector
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn to_vector(self) -> __m256i {
        unsafe { transmute(self.0) }
    }
    
    /// Create from AVX2 vector
    #[inline]
    #[cfg(target_arch = "x86_64")]
    pub(crate) unsafe fn from_vector(vector: __m256i) -> Self {
        Self(transmute(vector))
    }
    
    /// Create a packed value with all zeros
    #[inline]
    pub fn zero() -> Self {
        Self([BabyBear::default(); 8])
    }
    
    /// Create a packed value with all ones
    #[inline]
    pub fn one() -> Self {
        Self([BabyBear::from(1u32); 8])
    }
}

impl PackedField for PackedBabyBearAVX2 {
    type Scalar = BabyBear;
    const WIDTH: usize = 8;
    
    #[inline]
    fn from_slice(slice: &[BabyBear]) -> Self {
        assert!(slice.len() >= 8, "slice too short");
        let mut arr = [BabyBear::default(); 8];
        arr.copy_from_slice(&slice[..8]);
        Self(arr)
    }
    
    #[inline]
    fn to_array(&self) -> Vec<BabyBear> {
        self.0.to_vec()
    }
    
    #[inline]
    fn store(&self, dst: &mut [BabyBear]) {
        dst[..8].copy_from_slice(&self.0);
    }
    
    #[inline]
    fn broadcast(val: BabyBear) -> Self {
        Self([val; 8])
    }
    
    #[inline]
    fn from_fn<F: Fn(usize) -> BabyBear>(f: F) -> Self {
        Self([f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)])
    }
}


/// AVX2 modular addition: (a + b) mod P
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_mod_reduce(a: __m256i) -> __m256i {
    let p = _mm256_set1_epi32(PRIME as i32);
    let diff = _mm256_sub_epi32(a, p);
    // If a >= p use diff; else keep a.
    _mm256_min_epu32(a, diff)
}

/// AVX2 modular addition: (a + b) mod P
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_mod_add(a: __m256i, b: __m256i) -> __m256i {
    let sum = _mm256_add_epi32(a, b);
    avx2_mod_reduce(sum)
}

/// AVX2 modular subtraction: (a - b) mod P
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_mod_sub(a: __m256i, b: __m256i) -> __m256i {
    let p = _mm256_set1_epi32(PRIME as i32);
    let diff = _mm256_sub_epi32(a, b);
    // If a < b (borrow occurred), add P
    let mask = _mm256_cmpgt_epi32(b, a);
    let correction = _mm256_and_si256(mask, p);
    _mm256_add_epi32(diff, correction)
}

/// AVX2 modular negation: -a mod P
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_mod_neg(a: __m256i) -> __m256i {
    let p = _mm256_set1_epi32(PRIME as i32);
    let zero = _mm256_setzero_si256();
    let neg = _mm256_sub_epi32(p, a);
    // If a == 0, result is 0
    let mask = _mm256_cmpeq_epi32(a, zero);
    _mm256_andnot_si256(mask, neg)
}

/// Move high 32-bit of each 64-bit pair to low position
/// [a0, a1, a2, a3, a4, a5, a6, a7] -> [a1, a1, a3, a3, a5, a5, a7, a7]
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn movehdup_epi32(a: __m256i) -> __m256i {
    // Shuffle to duplicate odd elements: indices [1,1,3,3,5,5,7,7]
    _mm256_shuffle_epi32(a, 0xF5) // 0b11_11_01_01
}

/// Blend even positions from a and odd positions from b
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn blend_evn_odd(evn: __m256i, odd: __m256i) -> __m256i {
    // evn has results in positions 0,2,4,6 (as 64-bit results, high 32 bits valid)
    // odd has results in positions 1,3,5,7 (as 64-bit results, high 32 bits valid)
    // We need to extract high 32 bits of each 64-bit lane
    let evn_hi = _mm256_srli_epi64(evn, 32);
    let odd_hi = _mm256_and_si256(odd, _mm256_set1_epi64x(!0xFFFFFFFFi64));
    _mm256_or_si256(evn_hi, odd_hi)
}

/// Montgomery multiplication for even positions (using _mm256_mul_epu32)
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn monty_mul_evn(a: __m256i, b: __m256i) -> __m256i {
    let p = _mm256_set1_epi64x(PRIME as i64);
    let mu = _mm256_set1_epi64x(MONTY_MU as i64);
    
    // Product: 32x32 -> 64 for even positions
    let prod = _mm256_mul_epu32(a, b);
    
    // q = (prod * mu) mod 2^32 (low 32 bits of each 64-bit lane)
    let q = _mm256_mul_epu32(prod, mu);
    
    // q_p = q * P
    let q_p = _mm256_mul_epu32(q, p);
    
    // result = (prod - q_p) >> 32
    let diff = _mm256_sub_epi64(prod, q_p);

    // If prod < q_p (borrow), add P << 32 before shifting.
    let borrow = _mm256_cmpgt_epi64(q_p, prod);
    let correction = _mm256_and_si256(borrow, _mm256_slli_epi64(p, 32));
    _mm256_add_epi64(diff, correction)
}

/// AVX2 Montgomery multiplication for 8 elements
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_monty_mul(lhs: __m256i, rhs: __m256i) -> __m256i {
    // Process even positions [0, 2, 4, 6]
    let evn_result = monty_mul_evn(lhs, rhs);
    
    // Move odd positions to even for processing
    let lhs_odd = movehdup_epi32(lhs);
    let rhs_odd = movehdup_epi32(rhs);
    let odd_result = monty_mul_evn(lhs_odd, rhs_odd);
    
    // Combine results and canonicalize
    let packed = blend_evn_odd(evn_result, odd_result);
    avx2_mod_reduce(packed)
}

impl Neg for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let v = self.to_vector();
            let result = avx2_mod_neg(v);
            Self::from_vector(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self([
                -self.0[0], -self.0[1], -self.0[2], -self.0[3],
                -self.0[4], -self.0[5], -self.0[6], -self.0[7],
            ])
        }
    }
}

impl Add for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a = self.to_vector();
            let b = rhs.to_vector();
            let result = avx2_mod_add(a, b);
            Self::from_vector(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self([
                self.0[0] + rhs.0[0], self.0[1] + rhs.0[1],
                self.0[2] + rhs.0[2], self.0[3] + rhs.0[3],
                self.0[4] + rhs.0[4], self.0[5] + rhs.0[5],
                self.0[6] + rhs.0[6], self.0[7] + rhs.0[7],
            ])
        }
    }
}

impl AddAssign for PackedBabyBearAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a = self.to_vector();
            let b = rhs.to_vector();
            let result = avx2_mod_sub(a, b);
            Self::from_vector(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self([
                self.0[0] - rhs.0[0], self.0[1] - rhs.0[1],
                self.0[2] - rhs.0[2], self.0[3] - rhs.0[3],
                self.0[4] - rhs.0[4], self.0[5] - rhs.0[5],
                self.0[6] - rhs.0[6], self.0[7] - rhs.0[7],
            ])
        }
    }
}

impl SubAssign for PackedBabyBearAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let a = self.to_vector();
            let b = rhs.to_vector();
            let result = avx2_monty_mul(a, b);
            Self::from_vector(result)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self([
                self.0[0] * rhs.0[0], self.0[1] * rhs.0[1],
                self.0[2] * rhs.0[2], self.0[3] * rhs.0[3],
                self.0[4] * rhs.0[4], self.0[5] * rhs.0[5],
                self.0[6] * rhs.0[6], self.0[7] * rhs.0[7],
            ])
        }
    }
}

impl MulAssign for PackedBabyBearAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::fallback::PackedBabyBearScalar;
    use rand::RngCore;
    use rand::thread_rng;
    
    #[test]
    fn test_packed_avx2_basic() {
        let a = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 1));
        let b = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 10));
        
        let sum = a + b;
        let arr = sum.to_array();
        for i in 0..8 {
            assert_eq!(arr[i], BabyBear::new((i as u32 + 1) + (i as u32 + 10)));
        }
    }
    
    #[test]
    fn test_packed_avx2_mul() {
        let a = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 1));
        let b = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 2));
        
        let prod = a * b;
        let arr = prod.to_array();
        for i in 0..8 {
            let expected = BabyBear::new(i as u32 + 1) * BabyBear::new(i as u32 + 2);
            assert_eq!(arr[i], expected, "mismatch at index {}", i);
        }
    }
    
    #[test]
    fn test_packed_avx2_sub() {
        let a = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 100));
        let b = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32 + 1));
        
        let diff = a - b;
        let arr = diff.to_array();
        for i in 0..8 {
            assert_eq!(arr[i], BabyBear::new(99));
        }
    }

    #[test]
    fn test_avx2_scalar_consistency() {
        use super::PackedField;

        let mut rng = thread_rng();
        for _ in 0..100 {
            let values_a: [BabyBear; 8] = std::array::from_fn(|_| BabyBear::new(rng.next_u32()));
            let values_b: [BabyBear; 8] = std::array::from_fn(|_| BabyBear::new(rng.next_u32()));

            let avx_a = PackedBabyBearAVX2(values_a);
            let avx_b = PackedBabyBearAVX2(values_b);
            let scalar_a = PackedBabyBearScalar(values_a);
            let scalar_b = PackedBabyBearScalar(values_b);

            assert_eq!((avx_a + avx_b).to_array(), (scalar_a + scalar_b).to_array());
            assert_eq!((avx_a - avx_b).to_array(), (scalar_a - scalar_b).to_array());
            assert_eq!((avx_a * avx_b).to_array(), (scalar_a * scalar_b).to_array());
            assert_eq!((-avx_a).to_array(), (-scalar_a).to_array());
        }
    }
}
