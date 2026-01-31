//! BabyBear quartic extension field: F_p[X] / (X^4 - 11)
//!
//! Elements are represented as a_0 + a_1*X + a_2*X^2 + a_3*X^3

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand::RngCore;

use super::BabyBear;
use super::monty::consts;
use crate::field::{Field, FftField};

/// The non-residue W such that X^4 - W is irreducible over BabyBear
/// W = 11
const W: BabyBear = BabyBear { value: consts::MONTY_W };

/// BabyBear quartic extension field element
/// Represents a_0 + a_1*X + a_2*X^2 + a_3*X^3 where X^4 = 11
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BabyBearExt4 {
    pub c: [BabyBear; 4],
}

impl BabyBearExt4 {
    /// Create a new extension field element
    #[inline]
    pub const fn new(c0: BabyBear, c1: BabyBear, c2: BabyBear, c3: BabyBear) -> Self {
        Self { c: [c0, c1, c2, c3] }
    }
    
    /// Create from base field element (embed as constant polynomial)
    #[inline]
    pub const fn from_base(x: BabyBear) -> Self {
        Self { c: [x, BabyBear { value: 0 }, BabyBear { value: 0 }, BabyBear { value: 0 }] }
    }
    
    /// Exponentiation with u128 exponent (for large exponents like p^4 - 2)
    pub fn exp_u128(&self, mut exponent: u128) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        
        while exponent != 0 {
            if (exponent & 1) == 1 {
                result *= base;
            }
            base *= base;
            exponent >>= 1;
        }
        result
    }
}

impl Neg for BabyBearExt4 {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            c: [-self.c[0], -self.c[1], -self.c[2], -self.c[3]],
        }
    }
}

impl Add for BabyBearExt4 {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            c: [
                self.c[0] + rhs.c[0],
                self.c[1] + rhs.c[1],
                self.c[2] + rhs.c[2],
                self.c[3] + rhs.c[3],
            ],
        }
    }
}

impl AddAssign for BabyBearExt4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.c[i] += rhs.c[i];
        }
    }
}

impl Sub for BabyBearExt4 {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            c: [
                self.c[0] - rhs.c[0],
                self.c[1] - rhs.c[1],
                self.c[2] - rhs.c[2],
                self.c[3] - rhs.c[3],
            ],
        }
    }
}

impl SubAssign for BabyBearExt4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.c[i] -= rhs.c[i];
        }
    }
}

impl Mul for BabyBearExt4 {
    type Output = Self;
    
    /// Multiplication in F_p[X]/(X^4 - 11) using schoolbook method
    /// (a0 + a1*X + a2*X^2 + a3*X^3) * (b0 + b1*X + b2*X^2 + b3*X^3)
    fn mul(self, rhs: Self) -> Self::Output {
        let a = &self.c;
        let b = &rhs.c;
        
        // Compute product coefficients (before reduction)
        // c0 = a0*b0 + 11*(a1*b3 + a2*b2 + a3*b1)
        // c1 = a0*b1 + a1*b0 + 11*(a2*b3 + a3*b2)
        // c2 = a0*b2 + a1*b1 + a2*b0 + 11*(a3*b3)
        // c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
        
        let c0 = a[0] * b[0] + W * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
        let c1 = a[0] * b[1] + a[1] * b[0] + W * (a[2] * b[3] + a[3] * b[2]);
        let c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + W * (a[3] * b[3]);
        let c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
        
        Self { c: [c0, c1, c2, c3] }
    }
}

impl MulAssign for BabyBearExt4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl From<u32> for BabyBearExt4 {
    #[inline]
    fn from(value: u32) -> Self {
        Self::from_base(BabyBear::new(value))
    }
}

impl From<BabyBear> for BabyBearExt4 {
    #[inline]
    fn from(value: BabyBear) -> Self {
        Self::from_base(value)
    }
}

impl Field for BabyBearExt4 {
    const NAME: &'static str = "BabyBearExt4";
    const SIZE: usize = 16; // 4 * 4 bytes
    type BaseField = BabyBear;
    
    #[inline]
    fn zero() -> Self {
        Self::from_base(BabyBear::zero())
    }
    
    #[inline]
    fn is_zero(&self) -> bool {
        self.c.iter().all(|x| x.is_zero())
    }
    
    #[inline]
    fn one() -> Self {
        Self::from_base(BabyBear::one())
    }
    
    #[inline]
    fn inv_2() -> Self {
        Self::from_base(BabyBear::inv_2())
    }
    
    fn random(mut rng: impl RngCore) -> Self {
        Self {
            c: [
                BabyBear::random(&mut rng),
                BabyBear::random(&mut rng),
                BabyBear::random(&mut rng),
                BabyBear::random(&mut rng),
            ],
        }
    }
    
    fn exp(&self, mut exponent: usize) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        
        while exponent != 0 {
            if (exponent & 1) == 1 {
                result *= base;
            }
            base *= base;
            exponent >>= 1;
        }
        result
    }
    
    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        // For degree-4 extension F_{p^4}, use Fermat's little theorem:
        // a^{-1} = a^{p^4 - 2}
        // Use u128 to handle the large exponent
        let p = super::params::PRIME as u128;
        let exp = p * p * p * p - 2;
        Some(self.exp_u128(exp))
    }
    
    #[inline]
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c: [self.c[0] + rhs, self.c[1], self.c[2], self.c[3]],
        }
    }
    
    #[inline]
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        self.c[0] += rhs;
    }
    
    #[inline]
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        Self {
            c: [
                self.c[0] * rhs,
                self.c[1] * rhs,
                self.c[2] * rhs,
                self.c[3] * rhs,
            ],
        }
    }
    
    #[inline]
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        for i in 0..4 {
            self.c[i] *= rhs;
        }
    }
    
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let read_chunk = |start: usize| -> BabyBear {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[start..start + 8]);
            let value = u64::from_le_bytes(chunk) % (super::params::PRIME as u64);
            BabyBear::new(value as u32)
        };

        Self {
            c: [
                read_chunk(0),
                read_chunk(8),
                read_chunk(16),
                read_chunk(24),
            ],
        }
    }
    
    fn serialize_into(&self, buffer: &mut [u8]) {
        for i in 0..4 {
            self.c[i].serialize_into(&mut buffer[i * 4..(i + 1) * 4]);
        }
    }
    
    fn deserialize_from(buffer: &[u8]) -> Self {
        Self {
            c: [
                BabyBear::deserialize_from(&buffer[0..4]),
                BabyBear::deserialize_from(&buffer[4..8]),
                BabyBear::deserialize_from(&buffer[8..12]),
                BabyBear::deserialize_from(&buffer[12..16]),
            ],
        }
    }
}

impl FftField for BabyBearExt4 {
    const LOG_ORDER: u32 = 27;
    const ROOT_OF_UNITY: Self = Self {
        c: [
            BabyBear { value: super::params::TWO_ADIC_ROOT_OF_UNITY },
            BabyBear { value: 0 },
            BabyBear { value: 0 },
            BabyBear { value: 0 },
        ],
    };
    type FftBaseField = BabyBear;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_ext4_basic() {
        let a = BabyBearExt4::from(12345u32);
        let b = BabyBearExt4::from(67890u32);
        
        assert_eq!((a + b) - b, a);
        assert_eq!(a * BabyBearExt4::one(), a);
    }
    
    #[test]
    fn test_ext4_mul() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let a = BabyBearExt4::random(&mut rng);
            let b = BabyBearExt4::random(&mut rng);
            let c = BabyBearExt4::random(&mut rng);
            
            // Associativity
            assert_eq!((a * b) * c, a * (b * c));
            
            // Commutativity
            assert_eq!(a * b, b * a);
            
            // Identity
            assert_eq!(a * BabyBearExt4::one(), a);
        }
    }
    
    #[test]
    fn test_ext4_inv() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let a = BabyBearExt4::random(&mut rng);
            if !a.is_zero() {
                let a_inv = a.inv().unwrap();
                assert_eq!(a * a_inv, BabyBearExt4::one());
            }
        }
    }
}
