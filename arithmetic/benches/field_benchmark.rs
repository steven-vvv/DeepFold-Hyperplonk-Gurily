//! Field operation benchmarks

use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;

use arithmetic::field::Field;
use arithmetic::field::goldilocks64::Goldilocks64;

#[cfg(feature = "babybear")]
use arithmetic::field::babybear::{BabyBear, simd::PackedBabyBearScalar};

#[cfg(all(feature = "babybear", target_arch = "x86_64", target_feature = "avx2"))]
use arithmetic::field::babybear::simd::PackedBabyBear;

fn bench_field_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_multiplication");
    
    // Goldilocks64 scalar multiplication
    group.bench_function("goldilocks_scalar", |b| {
        let mut rng = thread_rng();
        let a = Goldilocks64::random(&mut rng);
        let x = Goldilocks64::random(&mut rng);
        b.iter(|| a * x)
    });
    
    #[cfg(feature = "babybear")]
    {
        // BabyBear scalar multiplication
        group.bench_function("babybear_scalar", |b| {
            let mut rng = thread_rng();
            let a = BabyBear::random(&mut rng);
            let x = BabyBear::random(&mut rng);
            b.iter(|| a * x)
        });
        
        // BabyBear packed multiplication (scalar fallback)
        group.bench_function("babybear_packed_scalar_8x", |b| {
            use arithmetic::field::babybear::simd::PackedField;
            let a = PackedBabyBearScalar::from_fn(|i| BabyBear::from(i as u32 + 1));
            let x = PackedBabyBearScalar::from_fn(|i| BabyBear::from(i as u32 + 10));
            b.iter(|| a * x)
        });

        // BabyBear packed multiplication (AVX2 SIMD)
        #[cfg(all(feature = "babybear", target_arch = "x86_64", target_feature = "avx2"))]
        group.bench_function("babybear_packed_avx2_8x", |b| {
            use arithmetic::field::babybear::simd::PackedField;
            let a = PackedBabyBear::from_fn(|i| BabyBear::from(i as u32 + 1));
            let x = PackedBabyBear::from_fn(|i| BabyBear::from(i as u32 + 10));
            b.iter(|| a * x)
        });
    }
    
    group.finish();
}

fn bench_field_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_addition");
    let mut rng = thread_rng();
    
    group.bench_function("goldilocks_add", |b| {
        let a = Goldilocks64::random(&mut rng);
        let x = Goldilocks64::random(&mut rng);
        b.iter(|| a + x)
    });
    
    #[cfg(feature = "babybear")]
    {
        group.bench_function("babybear_add", |b| {
            let a = BabyBear::random(&mut rng);
            let x = BabyBear::random(&mut rng);
            b.iter(|| a + x)
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_field_mul, bench_field_add);
criterion_main!(benches);
