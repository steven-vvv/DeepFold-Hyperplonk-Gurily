//! BabyBear base field implementation
//! 
//! A 31-bit prime field with p = 2^31 - 2^27 + 1 = 2013265921
//! Using Montgomery representation for efficient multiplication.

use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand::RngCore;

use super::params::PRIME;
use super::monty::{to_monty, from_monty, monty_mul, monty_add, monty_sub, monty_neg, consts};
use crate::field::{Field, FftField};

/// BabyBear field element in Montgomery representation
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct BabyBear {
    /// The value stored in Montgomery form: value = x * 2^32 mod P
    pub(crate) value: u32,
}

impl std::fmt::Debug for BabyBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BabyBear({})", from_monty(self.value))
    }
}

impl BabyBear {
    /// Create a new BabyBear element from a standard integer value
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self { value: to_monty(value % PRIME) }
    }

    /// Get the value in standard (non-Montgomery) representation
    #[inline]
    pub const fn to_canonical(&self) -> u32 {
        from_monty(self.value)
    }
    
    /// Get the raw Montgomery value
    #[inline]
    pub const fn to_monty(&self) -> u32 {
        self.value
    }
}

impl Neg for BabyBear {
    type Output = Self;
    
    #[inline]
    fn neg(self) -> Self::Output {
        Self { value: monty_neg(self.value) }
    }
}

impl Add for BabyBear {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self { value: monty_add(self.value, rhs.value) }
    }
}

impl AddAssign for BabyBear {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value = monty_add(self.value, rhs.value);
    }
}

impl Sub for BabyBear {
    type Output = Self;
    
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self { value: monty_sub(self.value, rhs.value) }
    }
}

impl SubAssign for BabyBear {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value = monty_sub(self.value, rhs.value);
    }
}

impl Mul for BabyBear {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self { value: monty_mul(self.value, rhs.value) }
    }
}

impl MulAssign for BabyBear {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.value = monty_mul(self.value, rhs.value);
    }
}

impl From<u32> for BabyBear {
    #[inline]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl Field for BabyBear {
    const NAME: &'static str = "BabyBear";
    const SIZE: usize = 4;
    type BaseField = BabyBear;
    
    #[inline]
    fn zero() -> Self {
        Self { value: consts::MONTY_ZERO }
    }
    
    #[inline]
    fn is_zero(&self) -> bool {
        self.value == 0
    }
    
    #[inline]
    fn one() -> Self {
        Self { value: consts::MONTY_ONE }
    }
    
    #[inline]
    fn inv_2() -> Self {
        Self { value: consts::MONTY_INV_TWO }
    }
    
    fn random(mut rng: impl RngCore) -> Self {
        // Generate a random u32 and reduce mod PRIME
        let raw = rng.next_u32() % PRIME;
        Self::new(raw)
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
        // Using Fermat's little theorem: a^{-1} = a^{p-2} mod p
        Some(self.exp((PRIME - 2) as usize))
    }
    
    #[inline]
    fn add_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self + rhs
    }
    
    #[inline]
    fn add_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self += rhs;
    }
    
    #[inline]
    fn mul_base_elem(&self, rhs: Self::BaseField) -> Self {
        *self * rhs
    }
    
    #[inline]
    fn mul_assign_base_elem(&mut self, rhs: Self::BaseField) {
        *self *= rhs;
    }
    
    fn from_uniform_bytes(bytes: &[u8; 32]) -> Self {
        let ptr = bytes.as_ptr() as *const u32;
        let v = unsafe { ptr.read_unaligned() } % PRIME;
        Self::new(v)
    }
    
    fn serialize_into(&self, buffer: &mut [u8]) {
        let canonical = self.to_canonical();
        buffer[..4].copy_from_slice(&canonical.to_le_bytes());
    }
    
    fn deserialize_from(buffer: &[u8]) -> Self {
        let bytes: [u8; 4] = buffer[..4].try_into().expect("buffer too small");
        let v = u32::from_le_bytes(bytes);
        assert!(v < PRIME, "value out of range");
        Self::new(v)
    }
}

impl FftField for BabyBear {
    const LOG_ORDER: u32 = 27; // 2-adicity of BabyBear
    const ROOT_OF_UNITY: Self = Self { value: super::params::TWO_ADIC_ROOT_OF_UNITY };
    type FftBaseField = BabyBear;
}
