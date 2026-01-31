//! Scalar fallback implementation for PackedBabyBear
//! Used when no SIMD instructions are available

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::PackedField;
use crate::field::babybear::BabyBear;

/// Scalar fallback packed BabyBear - processes 8 elements sequentially
/// Maintains API compatibility with SIMD versions
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedBabyBearScalar(pub [BabyBear; 8]);

impl PackedBabyBearScalar {
    pub const WIDTH: usize = 8;
    
    /// Create a new packed value with all zeros
    #[inline]
    pub fn zero() -> Self {
        Self([BabyBear::default(); 8])
    }
    
    /// Create a new packed value with all ones
    #[inline]
    pub fn one() -> Self {
        Self([BabyBear::from(1u32); 8])
    }
}

impl PackedField for PackedBabyBearScalar {
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

impl Neg for PackedBabyBearScalar {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        Self([
            -self.0[0], -self.0[1], -self.0[2], -self.0[3],
            -self.0[4], -self.0[5], -self.0[6], -self.0[7],
        ])
    }
}

impl Add for PackedBabyBearScalar {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0], self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2], self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4], self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6], self.0[7] + rhs.0[7],
        ])
    }
}

impl AddAssign for PackedBabyBearScalar {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Sub for PackedBabyBearScalar {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] - rhs.0[0], self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2], self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4], self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6], self.0[7] - rhs.0[7],
        ])
    }
}

impl SubAssign for PackedBabyBearScalar {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Mul for PackedBabyBearScalar {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] * rhs.0[0], self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2], self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4], self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6], self.0[7] * rhs.0[7],
        ])
    }
}

impl MulAssign for PackedBabyBearScalar {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] *= rhs.0[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use rand::thread_rng;
    
    #[test]
    fn test_packed_scalar_basic() {
        let a = PackedBabyBearScalar::from_fn(|i| BabyBear::new(i as u32 + 1));
        let b = PackedBabyBearScalar::from_fn(|i| BabyBear::new(i as u32 + 10));
        
        let sum = a + b;
        let arr = sum.to_array();
        for i in 0..8 {
            assert_eq!(arr[i], BabyBear::new((i as u32 + 1) + (i as u32 + 10)));
        }
    }
    
    #[test]
    fn test_packed_scalar_mul() {
        let a = PackedBabyBearScalar::from_fn(|i| BabyBear::new(i as u32 + 1));
        let b = PackedBabyBearScalar::from_fn(|i| BabyBear::new(i as u32 + 2));
        
        let prod = a * b;
        let arr = prod.to_array();
        for i in 0..8 {
            let expected = BabyBear::new(i as u32 + 1) * BabyBear::new(i as u32 + 2);
            assert_eq!(arr[i], expected);
        }
    }
    
    #[test]
    fn test_packed_scalar_broadcast() {
        let val = BabyBear::new(42);
        let packed = PackedBabyBearScalar::broadcast(val);
        
        for elem in packed.0.iter() {
            assert_eq!(*elem, val);
        }
    }
}
