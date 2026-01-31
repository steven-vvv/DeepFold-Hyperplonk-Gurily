//! Montgomery arithmetic utilities for BabyBear field

use super::params::{MONTY_BITS, MONTY_MASK, MONTY_MU, PRIME};

/// Convert a standard representation value to Montgomery representation
/// x -> x * 2^32 mod P
#[inline]
pub const fn to_monty(x: u32) -> u32 {
    (((x as u64) << MONTY_BITS) % (PRIME as u64)) as u32
}

/// Convert from Montgomery representation to standard representation
/// x * 2^32 mod P -> x mod P
#[inline]
pub const fn from_monty(x: u32) -> u32 {
    monty_reduce(x as u64)
}

/// Montgomery reduction
/// Input: 0 <= x < P * 2^32
/// Output: 0 <= result < P such that result ≡ x * 2^{-32} (mod P)
#[inline]
pub const fn monty_reduce(x: u64) -> u32 {
    // q = (x * mu) mod 2^32
    let q = (x.wrapping_mul(MONTY_MU as u64)) & MONTY_MASK;
    
    // q_p = q * P
    let q_p = q * (PRIME as u64);
    
    // (x - q_p) / 2^32
    // Since x ≡ q_p (mod 2^32), the low 32 bits are 0
    let (x_sub_qp, borrow) = x.overflowing_sub(q_p);
    let result = (x_sub_qp >> MONTY_BITS) as u32;
    
    // If borrow occurred, add P to correct
    if borrow {
        result.wrapping_add(PRIME)
    } else if result >= PRIME {
        result - PRIME
    } else {
        result
    }
}

/// Montgomery multiplication: compute (a * b * 2^{-32}) mod P
/// where a and b are in Montgomery representation
#[inline]
pub const fn monty_mul(a: u32, b: u32) -> u32 {
    monty_reduce((a as u64) * (b as u64))
}

/// Montgomery addition with reduction
#[inline]
pub const fn monty_add(a: u32, b: u32) -> u32 {
    let sum = a.wrapping_add(b);
    if sum >= PRIME || sum < a {
        sum.wrapping_sub(PRIME)
    } else {
        sum
    }
}

/// Montgomery subtraction with reduction
#[inline]
pub const fn monty_sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a.wrapping_add(PRIME).wrapping_sub(b)
    }
}

/// Montgomery negation
#[inline]
pub const fn monty_neg(a: u32) -> u32 {
    if a == 0 {
        0
    } else {
        PRIME - a
    }
}

/// Precomputed constants in Montgomery form
pub mod consts {
    use super::{to_monty, PRIME};
    
    /// Zero in Montgomery form (0 * 2^32 mod P = 0)
    pub const MONTY_ZERO: u32 = 0;
    
    /// One in Montgomery form (1 * 2^32 mod P)
    pub const MONTY_ONE: u32 = to_monty(1);
    
    /// Two in Montgomery form
    pub const MONTY_TWO: u32 = to_monty(2);
    
    /// Inverse of 2 in Montgomery form: (P + 1) / 2 in monty
    pub const MONTY_INV_TWO: u32 = to_monty((PRIME + 1) / 2);
    
    /// The value 11 in Montgomery form (for extension field X^4 - 11)
    pub const MONTY_W: u32 = to_monty(11);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_monty_roundtrip() {
        for x in [0u32, 1, 2, 100, 1000, PRIME - 1] {
            let m = to_monty(x);
            let r = from_monty(m);
            assert_eq!(r, x);
        }
    }
    
    #[test]
    fn test_monty_mul() {
        // Test: 3 * 5 = 15
        let a = to_monty(3);
        let b = to_monty(5);
        let c = monty_mul(a, b);
        assert_eq!(from_monty(c), 15);
    }
    
    #[test]
    fn test_monty_add() {
        // Test: 100 + 200 = 300
        let a = to_monty(100);
        let b = to_monty(200);
        let c = monty_add(a, b);
        assert_eq!(from_monty(c), 300);
        
        // Test wrap around
        let x = to_monty(PRIME - 1);
        let y = to_monty(2);
        let z = monty_add(x, y);
        assert_eq!(from_monty(z), 1);
    }
    
    #[test]
    fn test_monty_sub() {
        // Test: 300 - 100 = 200
        let a = to_monty(300);
        let b = to_monty(100);
        let c = monty_sub(a, b);
        assert_eq!(from_monty(c), 200);
        
        // Test wrap around
        let x = to_monty(1);
        let y = to_monty(2);
        let z = monty_sub(x, y);
        assert_eq!(from_monty(z), PRIME - 1);
    }
}
