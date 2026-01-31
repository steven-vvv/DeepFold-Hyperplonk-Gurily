//! BabyBear field implementation with SIMD optimization
//! 
//! BabyBear is a 31-bit prime field with p = 2^31 - 2^27 + 1
//! This module provides:
//! - Base field operations with Montgomery representation
//! - Extension field (degree 4)
//! - SIMD-optimized packed field operations (AVX2/AVX512/NEON)

mod babybear;
mod monty;
mod extension;

pub mod simd;

pub use babybear::BabyBear;
pub use extension::BabyBearExt4;
pub use monty::{to_monty, from_monty, monty_reduce};

#[cfg(feature = "simd")]
pub use simd::PackedBabyBear;

/// BabyBear field parameters
pub mod params {
    /// The prime modulus: p = 2^31 - 2^27 + 1 = 2013265921
    pub const PRIME: u32 = 0x78000001;
    
    /// Montgomery parameter: number of bits for Montgomery representation
    pub const MONTY_BITS: u32 = 32;
    
    /// Montgomery mask: 2^32 - 1
    pub const MONTY_MASK: u64 = (1u64 << MONTY_BITS) - 1;
    
    /// Montgomery multiplier: -p^{-1} mod 2^32
    pub const MONTY_MU: u32 = 0x88000001;
    
    /// 2-adicity: the largest k such that 2^k divides p-1
    pub const TWO_ADICITY: usize = 27;
    
    /// Generator of the multiplicative group
    pub const MULTIPLICATIVE_GENERATOR: u32 = 31;
    
    /// Primitive root of unity of order 2^27 (in Montgomery form)
    /// Computed as: generator^((p-1)/2^27) where generator = 31
    /// p - 1 = 2013265920 = 2^27 * 15, so we need 31^15 mod p
    /// 31^15 mod p = 440564289 (canonical form)
    /// In Montgomery form: 1476048622
    pub const TWO_ADIC_ROOT_OF_UNITY: u32 = 1476048622;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use rand::thread_rng;

    #[test]
    fn test_babybear_basic_arithmetic() {
        let a = BabyBear::new(12345);
        let b = BabyBear::new(67890);
        
        // Addition and subtraction inverse
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
        
        // Multiplication identity
        assert_eq!(a * BabyBear::one(), a);
        assert_eq!(a * BabyBear::zero(), BabyBear::zero());
        
        // Negation
        assert_eq!(a + (-a), BabyBear::zero());
    }

    #[test]
    fn test_babybear_multiplication() {
        let a = BabyBear::new(1000);
        let b = BabyBear::new(2000);
        let c = BabyBear::new(2000000); // 1000 * 2000
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_babybear_inverse() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let a = BabyBear::random(&mut rng);
            if !a.is_zero() {
                let a_inv = a.inv().unwrap();
                assert_eq!(a * a_inv, BabyBear::one());
            }
        }
    }

    #[test]
    fn test_babybear_exp() {
        let a = BabyBear::new(2);
        let a_squared = a.exp(2);
        assert_eq!(a_squared, BabyBear::new(4));
        
        let a_cubed = a.exp(3);
        assert_eq!(a_cubed, BabyBear::new(8));
    }

    #[test]
    fn test_montgomery_roundtrip() {
        for i in [0u32, 1, 2, 100, 1000, params::PRIME - 1] {
            let m = to_monty(i);
            let r = from_monty(m);
            assert_eq!(r, i, "Montgomery roundtrip failed for {}", i);
        }
    }
    
    #[test]
    fn test_two_adic_root_of_unity() {
        use crate::field::FftField;
        
        // First, compute the correct root of unity from generator
        // generator = 31, and we need generator^((p-1)/2^27) = generator^15
        let generator = BabyBear::new(params::MULTIPLICATIVE_GENERATOR);
        let correct_root = generator.exp(15); // (p-1)/2^27 = 15
        
        // Verify the computed root is correct
        let mut power = correct_root;
        for _ in 0..27 {
            power = power.square();
        }
        assert_eq!(power, BabyBear::one(), "computed root^(2^27) should be 1");
        
        // Verify it's primitive (not a 2^26-th root)
        let mut half_power = correct_root;
        for _ in 0..26 {
            half_power = half_power.square();
        }
        assert_ne!(half_power, BabyBear::one(), "computed root^(2^26) should not be 1");
        
        // Now verify the constant matches
        let g = BabyBear::ROOT_OF_UNITY;
        assert_eq!(g.to_canonical(), correct_root.to_canonical(), 
            "ROOT_OF_UNITY constant should be {} (monty: {})", 
            correct_root.to_canonical(), correct_root.to_monty());
    }
    
    #[test]
    fn test_fft_compatibility() {
        use crate::field::FftField;
        
        // Verify LOG_ORDER matches 2-adicity
        assert_eq!(BabyBear::LOG_ORDER, 27);
        
        // Verify the root of unity order
        let g = BabyBear::ROOT_OF_UNITY;
        assert!(!g.is_zero());
    }
    
    #[test]
    fn test_radix2_group_integration() {
        use crate::mul_group::Radix2Group;
        
        // Create a small Radix2Group with BabyBear
        let group: Radix2Group<BabyBear> = Radix2Group::new(4); // 2^4 = 16 elements
        
        assert_eq!(group.size(), 16);
        
        // Verify first element is 1
        assert_eq!(group.element_at(0), BabyBear::one());
        
        // Verify omega^16 = 1
        let omega = group.element_at(1);
        assert_eq!(omega.exp(16), BabyBear::one());
        
        // Verify inverses work
        let elem = group.element_at(3);
        let elem_inv = group.element_inv_at(3);
        assert_eq!(elem * elem_inv, BabyBear::one());
    }
    
    #[test]
    fn test_fft_with_babybear() {
        use crate::mul_group::Radix2Group;
        
        // Test FFT with BabyBear
        let group: Radix2Group<BabyBear> = Radix2Group::new(3); // 8 elements
        
        // Create a simple polynomial: [1, 2, 3, 4, 5, 6, 7, 8]
        let coeffs: Vec<BabyBear> = (1..=8).map(|x| BabyBear::new(x)).collect();
        let original = coeffs.clone();
        
        // FFT (takes ownership)
        let evals = group.fft(coeffs);
        
        // IFFT (takes ownership)
        let recovered = group.ifft(evals);
        
        // Should get back original
        assert_eq!(recovered, original);
    }
}
