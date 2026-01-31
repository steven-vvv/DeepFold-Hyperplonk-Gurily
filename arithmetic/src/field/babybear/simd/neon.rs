//! ARM NEON SIMD implementation for PackedBabyBear
//! Processes 4 BabyBear elements in parallel using 128-bit registers

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::PackedField;
use crate::field::babybear::BabyBear;

/// Packed BabyBear using ARM NEON (4 elements parallel)
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct PackedBabyBearNeon(pub [BabyBear; 4]);

impl Default for PackedBabyBearNeon {
    fn default() -> Self {
        Self([BabyBear::default(); 4])
    }
}

impl PartialEq for PackedBabyBearNeon {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PackedBabyBearNeon {}

impl PackedBabyBearNeon {
    pub const WIDTH: usize = 4;
    
    /// Create a packed value with all zeros
    #[inline]
    pub fn zero() -> Self {
        Self([BabyBear::default(); 4])
    }
    
    /// Create a packed value with all ones
    #[inline]
    pub fn one() -> Self {
        Self([BabyBear::from(1u32); 4])
    }
}

impl PackedField for PackedBabyBearNeon {
    type Scalar = BabyBear;
    const WIDTH: usize = 4;
    
    #[inline]
    fn from_slice(slice: &[BabyBear]) -> Self {
        assert!(slice.len() >= 4, "slice too short");
        let mut arr = [BabyBear::default(); 4];
        arr.copy_from_slice(&slice[..4]);
        Self(arr)
    }
    
    #[inline]
    fn to_array(&self) -> Vec<BabyBear> {
        self.0.to_vec()
    }
    
    #[inline]
    fn store(&self, dst: &mut [BabyBear]) {
        dst[..4].copy_from_slice(&self.0);
    }
    
    #[inline]
    fn broadcast(val: BabyBear) -> Self {
        Self([val; 4])
    }
    
    #[inline]
    fn from_fn<F: Fn(usize) -> BabyBear>(f: F) -> Self {
        Self([f(0), f(1), f(2), f(3)])
    }
}

// Scalar implementations (NEON intrinsics can be added for aarch64 targets)
impl Neg for PackedBabyBearNeon {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl Add for PackedBabyBearNeon {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl AddAssign for PackedBabyBearNeon {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Sub for PackedBabyBearNeon {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl SubAssign for PackedBabyBearNeon {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Mul for PackedBabyBearNeon {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }
}

impl MulAssign for PackedBabyBearNeon {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] *= rhs.0[i];
        }
    }
}
