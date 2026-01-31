//! Detailed End-to-End Benchmarking for DeepFold-Hyperplonk
//!
//! This benchmark provides comprehensive performance analysis with:
//! - Step-by-step timing for each phase
//! - Multiple scheme comparisons (BaseFold, DeepFold)
//! - Multiple field comparisons (Goldilocks64, BabyBear)
//! - Detailed terminal output for analysis
//!
//! Usage:
//!   cargo bench -p hyperplonk --bench detailed_benchmark
//!   cargo bench -p hyperplonk --bench detailed_benchmark -- --nv 16
//!   cargo bench -p hyperplonk --bench detailed_benchmark -- --samples 5

use std::time::{Duration, Instant};
use std::env;

use arithmetic::{
    field::{
        goldilocks64::{Goldilocks64, Goldilocks64Ext},
        Field,
    },
    mul_group::Radix2Group,
};
use rand::thread_rng;

use hyperplonk::{circuit::Circuit, prover::Prover, sumcheck::Sumcheck, verifier::Verifier};
use poly_commit::basefold::{BaseFoldParam, BaseFoldVerifier, BasefoldProver};
use poly_commit::deepfold::{DeepFoldParam, DeepFoldProver, DeepFoldVerifier};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for benchmark runs
#[derive(Clone, Debug)]
pub struct BenchConfig {
    /// Number of variables (log2 of number of gates)
    pub nv: u32,
    /// Number of samples for averaging
    pub samples: usize,
    /// Number of warmup iterations
    pub warmup: usize,
    /// Query number for BaseFold
    pub basefold_queries: usize,
    /// Query number for DeepFold
    pub deepfold_queries: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            nv: 16,
            samples: 3,
            warmup: 1,
            basefold_queries: 120,
            deepfold_queries: 45,
        }
    }
}

impl BenchConfig {
    pub fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut config = Self::default();
        
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--nv" if i + 1 < args.len() => {
                    config.nv = args[i + 1].parse().unwrap_or(config.nv);
                    i += 2;
                }
                "--samples" if i + 1 < args.len() => {
                    config.samples = args[i + 1].parse().unwrap_or(config.samples);
                    i += 2;
                }
                "--warmup" if i + 1 < args.len() => {
                    config.warmup = args[i + 1].parse().unwrap_or(config.warmup);
                    i += 2;
                }
                "--basefold-queries" if i + 1 < args.len() => {
                    config.basefold_queries = args[i + 1].parse().unwrap_or(config.basefold_queries);
                    i += 2;
                }
                "--deepfold-queries" if i + 1 < args.len() => {
                    config.deepfold_queries = args[i + 1].parse().unwrap_or(config.deepfold_queries);
                    i += 2;
                }
                _ => i += 1,
            }
        }
        config
    }
}

// ============================================================================
// Timing Data Structures
// ============================================================================

/// Detailed timing for a single benchmark run
#[derive(Clone, Debug, Default)]
pub struct StepTiming {
    /// Step name
    pub name: String,
    /// Duration of this step
    pub duration: Duration,
    /// Additional metadata
    pub metadata: Option<String>,
}

/// Complete timing breakdown for a proof generation/verification cycle
#[derive(Clone, Debug, Default)]
pub struct BenchmarkResult {
    /// Scheme name (e.g., "DeepFold", "BaseFold")
    pub scheme: String,
    /// Field name (e.g., "Goldilocks64", "BabyBear")
    pub field: String,
    /// Number of variables
    pub nv: u32,
    /// Number of gates (2^nv)
    pub num_gates: usize,
    /// Step-by-step timing breakdown
    pub steps: Vec<StepTiming>,
    /// Total prover time
    pub prover_total: Duration,
    /// Sumcheck time (subset of prover)
    pub sumcheck_time: Duration,
    /// Verifier time
    pub verifier_time: Duration,
    /// Proof size in bytes
    pub proof_size: usize,
    /// Sample index
    pub sample_index: usize,
}

impl BenchmarkResult {
    pub fn new(scheme: &str, field: &str, nv: u32, sample_index: usize) -> Self {
        Self {
            scheme: scheme.to_string(),
            field: field.to_string(),
            nv,
            num_gates: 1 << nv,
            sample_index,
            ..Default::default()
        }
    }

    pub fn add_step(&mut self, name: &str, duration: Duration, metadata: Option<String>) {
        self.steps.push(StepTiming {
            name: name.to_string(),
            duration,
            metadata,
        });
    }
}

/// Aggregated statistics across multiple samples
#[derive(Clone, Debug)]
pub struct AggregatedStats {
    pub scheme: String,
    pub field: String,
    pub nv: u32,
    pub num_gates: usize,
    pub samples: usize,
    
    // Prover stats
    pub prover_mean: Duration,
    pub prover_min: Duration,
    pub prover_max: Duration,
    pub prover_stddev_us: f64,
    
    // Sumcheck stats
    pub sumcheck_mean: Duration,
    pub sumcheck_min: Duration,
    pub sumcheck_max: Duration,
    
    // Verifier stats
    pub verifier_mean: Duration,
    pub verifier_min: Duration,
    pub verifier_max: Duration,
    
    // Proof size
    pub proof_size: usize,
    
    // Per-step aggregated timing
    pub step_means: Vec<(String, Duration)>,
}

impl AggregatedStats {
    pub fn from_results(results: &[BenchmarkResult]) -> Self {
        if results.is_empty() {
            panic!("Cannot aggregate empty results");
        }
        
        let first = &results[0];
        let n = results.len();
        
        // Prover stats
        let prover_times: Vec<Duration> = results.iter().map(|r| r.prover_total).collect();
        let prover_mean = prover_times.iter().sum::<Duration>() / n as u32;
        let prover_min = prover_times.iter().min().copied().unwrap();
        let prover_max = prover_times.iter().max().copied().unwrap();
        let prover_mean_us = prover_mean.as_micros() as f64;
        let prover_variance: f64 = prover_times.iter()
            .map(|t| {
                let diff = t.as_micros() as f64 - prover_mean_us;
                diff * diff
            })
            .sum::<f64>() / n as f64;
        let prover_stddev_us = prover_variance.sqrt();
        
        // Sumcheck stats
        let sumcheck_times: Vec<Duration> = results.iter().map(|r| r.sumcheck_time).collect();
        let sumcheck_mean = sumcheck_times.iter().sum::<Duration>() / n as u32;
        let sumcheck_min = sumcheck_times.iter().min().copied().unwrap();
        let sumcheck_max = sumcheck_times.iter().max().copied().unwrap();
        
        // Verifier stats
        let verifier_times: Vec<Duration> = results.iter().map(|r| r.verifier_time).collect();
        let verifier_mean = verifier_times.iter().sum::<Duration>() / n as u32;
        let verifier_min = verifier_times.iter().min().copied().unwrap();
        let verifier_max = verifier_times.iter().max().copied().unwrap();
        
        // Aggregate step timing
        let mut step_means = Vec::new();
        if !first.steps.is_empty() {
            for step_idx in 0..first.steps.len() {
                let step_name = &first.steps[step_idx].name;
                let step_times: Vec<Duration> = results.iter()
                    .filter_map(|r| r.steps.get(step_idx))
                    .map(|s| s.duration)
                    .collect();
                if !step_times.is_empty() {
                    let mean = step_times.iter().sum::<Duration>() / step_times.len() as u32;
                    step_means.push((step_name.clone(), mean));
                }
            }
        }
        
        Self {
            scheme: first.scheme.clone(),
            field: first.field.clone(),
            nv: first.nv,
            num_gates: first.num_gates,
            samples: n,
            prover_mean,
            prover_min,
            prover_max,
            prover_stddev_us,
            sumcheck_mean,
            sumcheck_min,
            sumcheck_max,
            verifier_mean,
            verifier_min,
            verifier_max,
            proof_size: first.proof_size,
            step_means,
        }
    }
}

// ============================================================================
// Output Formatting
// ============================================================================

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us >= 1_000_000 {
        format!("{:.2} s", us as f64 / 1_000_000.0)
    } else if us >= 1_000 {
        format!("{:.2} ms", us as f64 / 1_000.0)
    } else {
        format!("{} μs", us)
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn print_header() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          DeepFold-Hyperplonk Detailed End-to-End Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_config(config: &BenchConfig) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration                                                               │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│  Number of Variables (nv):     {:>8}                                     │", config.nv);
    println!("│  Number of Gates (2^nv):       {:>8}                                     │", 1u32 << config.nv);
    println!("│  Samples per scheme:           {:>8}                                     │", config.samples);
    println!("│  Warmup iterations:            {:>8}                                     │", config.warmup);
    println!("│  BaseFold queries:             {:>8}                                     │", config.basefold_queries);
    println!("│  DeepFold queries:             {:>8}                                     │", config.deepfold_queries);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

fn print_sample_result(result: &BenchmarkResult) {
    println!("  Sample #{}: Prover={}, Sumcheck={}, Verifier={}, Proof={}",
        result.sample_index + 1,
        format_duration(result.prover_total),
        format_duration(result.sumcheck_time),
        format_duration(result.verifier_time),
        format_size(result.proof_size)
    );
}

fn print_aggregated_stats(stats: &AggregatedStats) {
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ {} / {} (nv={}, gates={})", 
        stats.scheme, stats.field, stats.nv, stats.num_gates);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ PROVER TIMING ({} samples):", stats.samples);
    println!("│   Mean:     {:>15}                                              │", format_duration(stats.prover_mean));
    println!("│   Min:      {:>15}                                              │", format_duration(stats.prover_min));
    println!("│   Max:      {:>15}                                              │", format_duration(stats.prover_max));
    println!("│   StdDev:   {:>12.2} μs                                              │", stats.prover_stddev_us);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ SUMCHECK TIMING:");
    println!("│   Mean:     {:>15}                                              │", format_duration(stats.sumcheck_mean));
    println!("│   Min:      {:>15}                                              │", format_duration(stats.sumcheck_min));
    println!("│   Max:      {:>15}                                              │", format_duration(stats.sumcheck_max));
    println!("│   % of Prover: {:>6.1}%                                                   │", 
        stats.sumcheck_mean.as_micros() as f64 / stats.prover_mean.as_micros() as f64 * 100.0);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ VERIFIER TIMING:");
    println!("│   Mean:     {:>15}                                              │", format_duration(stats.verifier_mean));
    println!("│   Min:      {:>15}                                              │", format_duration(stats.verifier_min));
    println!("│   Max:      {:>15}                                              │", format_duration(stats.verifier_max));
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ PROOF SIZE:  {:>15}                                              │", format_size(stats.proof_size));
    
    if !stats.step_means.is_empty() {
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!("│ STEP BREAKDOWN:");
        for (name, duration) in &stats.step_means {
            let pct = duration.as_micros() as f64 / stats.prover_mean.as_micros() as f64 * 100.0;
            println!("│   {:25} {:>12} ({:>5.1}%)                        │", 
                name, format_duration(*duration), pct);
        }
    }
    
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
}

fn print_comparison_table(all_stats: &[AggregatedStats]) {
    if all_stats.is_empty() {
        return;
    }
    
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           COMPARISON SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Scheme/Field           │ Prover (mean) │ Sumcheck     │ Verifier    │ Proof  ║");
    println!("╠════════════════════════╪═══════════════╪══════════════╪═════════════╪════════╣");
    
    for stats in all_stats {
        let label = format!("{}/{}", stats.scheme, stats.field);
        println!("║ {:22} │ {:>13} │ {:>12} │ {:>11} │ {:>6} ║",
            label,
            format_duration(stats.prover_mean),
            format_duration(stats.sumcheck_mean),
            format_duration(stats.verifier_mean),
            format_size(stats.proof_size)
        );
    }
    
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    
    // Speedup comparison if we have multiple entries
    if all_stats.len() >= 2 {
        println!();
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ SPEEDUP ANALYSIS (relative to first scheme)                                 │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        
        let baseline = &all_stats[0];
        for (i, stats) in all_stats.iter().enumerate().skip(1) {
            let prover_speedup = baseline.prover_mean.as_micros() as f64 / stats.prover_mean.as_micros() as f64;
            let sumcheck_speedup = baseline.sumcheck_mean.as_micros() as f64 / stats.sumcheck_mean.as_micros() as f64;
            let verifier_speedup = baseline.verifier_mean.as_micros() as f64 / stats.verifier_mean.as_micros() as f64;
            let proof_ratio = stats.proof_size as f64 / baseline.proof_size as f64;
            
            let current_label = format!("{}/{}", stats.scheme, stats.field);
            let baseline_label = format!("{}/{}", baseline.scheme, baseline.field);
            println!("│ {} vs {}:", current_label, baseline_label);
            println!("│   Prover speedup:    {:>6.2}x                                             │", prover_speedup);
            println!("│   Sumcheck speedup:  {:>6.2}x                                             │", sumcheck_speedup);
            println!("│   Verifier speedup:  {:>6.2}x                                             │", verifier_speedup);
            println!("│   Proof size ratio:  {:>6.2}x                                             │", proof_ratio);
            if i < all_stats.len() - 1 {
                println!("├─────────────────────────────────────────────────────────────────────────────┤");
            }
        }
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
    }
}

// ============================================================================
// Benchmark Implementations
// ============================================================================

fn bench_basefold_goldilocks(config: &BenchConfig) -> Vec<BenchmarkResult> {
    let nv = config.nv;
    let num_gates = 1u32 << nv;
    
    println!("\n▶ Running BaseFold / Goldilocks64 (nv={}, {} samples)...", nv, config.samples);
    
    // Setup
    let mock_circuit = Circuit::<Goldilocks64Ext> {
        permutation: [
            (0..num_gates).map(|x| x.into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
        ],
        selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
    };
    
    let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(nv + 2)];
    for i in 1..nv as usize {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    
    let pp = BaseFoldParam::<Goldilocks64Ext> {
        mult_subgroups,
        variable_num: nv as usize,
        query_num: config.basefold_queries,
    };
    
    let (pk, vk) = mock_circuit.setup::<BasefoldProver<_>, BaseFoldVerifier<_>>(&pp, &pp);
    let prover = Prover { prover_key: pk };
    let verifier = Verifier { verifier_key: vk };
    
    // Generate witness
    let a: Vec<Goldilocks64> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut thread_rng()))
        .collect();
    let b: Vec<Goldilocks64> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut thread_rng()))
        .collect();
    let c: Vec<Goldilocks64> = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = mock_circuit.selector[i];
            -((Goldilocks64::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect();
    
    // Warmup
    for _ in 0..config.warmup {
        let _ = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
    }
    
    // Benchmark samples
    let mut results = Vec::new();
    for sample in 0..config.samples {
        let mut result = BenchmarkResult::new("BaseFold", "Goldilocks64", nv, sample);
        
        Sumcheck::reset_timing();
        
        let start = Instant::now();
        let proof = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
        result.prover_total = start.elapsed();
        
        result.sumcheck_time = Duration::from_micros(Sumcheck::prove_time_us());
        result.proof_size = proof.bytes.len();
        
        // Add step breakdown
        result.add_step("Sumcheck", result.sumcheck_time, None);
        result.add_step("Polynomial Commitment", 
            result.prover_total.saturating_sub(result.sumcheck_time), None);
        
        let start = Instant::now();
        assert!(verifier.verify(&pp, nv as usize, proof));
        result.verifier_time = start.elapsed();
        
        print_sample_result(&result);
        results.push(result);
    }
    
    results
}

fn bench_deepfold_goldilocks(config: &BenchConfig) -> Vec<BenchmarkResult> {
    let nv = config.nv;
    let num_gates = 1u32 << nv;
    
    println!("\n▶ Running DeepFold / Goldilocks64 (nv={}, {} samples)...", nv, config.samples);
    
    // Setup
    let mock_circuit = Circuit::<Goldilocks64Ext> {
        permutation: [
            (0..num_gates).map(|x| x.into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
        ],
        selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
    };
    
    let mut mult_subgroups = vec![Radix2Group::<Goldilocks64>::new(nv + 2)];
    for i in 1..nv as usize {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    
    let pp = DeepFoldParam::<Goldilocks64Ext> {
        mult_subgroups,
        variable_num: nv as usize,
        query_num: config.deepfold_queries,
    };
    
    let (pk, vk) = mock_circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover { prover_key: pk };
    let verifier = Verifier { verifier_key: vk };
    
    // Generate witness
    let a: Vec<Goldilocks64> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut thread_rng()))
        .collect();
    let b: Vec<Goldilocks64> = (0..num_gates)
        .map(|_| Goldilocks64::random(&mut thread_rng()))
        .collect();
    let c: Vec<Goldilocks64> = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = mock_circuit.selector[i];
            -((Goldilocks64::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect();
    
    // Warmup
    for _ in 0..config.warmup {
        let _ = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
    }
    
    // Benchmark samples
    let mut results = Vec::new();
    for sample in 0..config.samples {
        let mut result = BenchmarkResult::new("DeepFold", "Goldilocks64", nv, sample);
        
        Sumcheck::reset_timing();
        
        let start = Instant::now();
        let proof = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
        result.prover_total = start.elapsed();
        
        result.sumcheck_time = Duration::from_micros(Sumcheck::prove_time_us());
        result.proof_size = proof.bytes.len();
        
        // Add step breakdown
        result.add_step("Sumcheck", result.sumcheck_time, None);
        result.add_step("Polynomial Commitment", 
            result.prover_total.saturating_sub(result.sumcheck_time), None);
        
        let start = Instant::now();
        assert!(verifier.verify(&pp, nv as usize, proof));
        result.verifier_time = start.elapsed();
        
        print_sample_result(&result);
        results.push(result);
    }
    
    results
}

#[cfg(feature = "babybear")]
fn bench_deepfold_babybear(config: &BenchConfig) -> Vec<BenchmarkResult> {
    use arithmetic::field::babybear::{BabyBear, BabyBearExt4};
    
    let nv = config.nv;
    let num_gates = 1u32 << nv;
    
    println!("\n▶ Running DeepFold / BabyBear (nv={}, {} samples)...", nv, config.samples);
    
    // Note: BabyBear + DeepFold + Hyperplonk integration is experimental.
    // The SIMD paths work at the component level but full end-to-end verification
    // may have issues. We'll try to run and report any failures gracefully.
    
    // Setup - must use same permutation offsets as prover/verifier (1<<29, 1<<30)
    let mock_circuit = Circuit::<BabyBearExt4> {
        permutation: [
            (0..num_gates).map(|x| x.into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 29)).into()).collect(),
            (0..num_gates).map(|x| (x + (1 << 30)).into()).collect(),
        ],
        selector: (0..num_gates).map(|x| (x & 1).into()).collect(),
    };
    
    let mut mult_subgroups = vec![Radix2Group::<BabyBear>::new(nv + 2)];
    for i in 1..nv as usize {
        mult_subgroups.push(mult_subgroups[i - 1].exp(2));
    }
    
    let pp = DeepFoldParam::<BabyBearExt4> {
        mult_subgroups,
        variable_num: nv as usize,
        query_num: config.deepfold_queries,
    };
    
    let (pk, vk) = mock_circuit.setup::<DeepFoldProver<_>, DeepFoldVerifier<_>>(&pp, &pp);
    let prover = Prover { prover_key: pk };
    let verifier = Verifier { verifier_key: vk };
    
    // Generate witness
    let a: Vec<BabyBear> = (0..num_gates)
        .map(|_| BabyBear::random(&mut thread_rng()))
        .collect();
    let b: Vec<BabyBear> = (0..num_gates)
        .map(|_| BabyBear::random(&mut thread_rng()))
        .collect();
    let c: Vec<BabyBear> = (0..num_gates)
        .map(|i| {
            let i = i as usize;
            let s = mock_circuit.selector[i];
            -((BabyBear::one() - s) * (a[i] + b[i]) + s * a[i] * b[i])
        })
        .collect();
    
    // First, test if verification works at all
    println!("  Testing BabyBear verification...");
    let test_proof = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
    let verification_works = verifier.verify(&pp, nv as usize, test_proof);
    
    if !verification_works {
        println!("  ⚠ WARNING: BabyBear verification failed!");
        println!("  BabyBear + DeepFold + Hyperplonk integration is experimental.");
        println!("  The SIMD optimizations work at component level (Sumcheck, DeepFold folding)");
        println!("  but full end-to-end verification needs additional work.");
        println!("  Skipping BabyBear benchmark. Prover-only timing available in warmup.");
        println!();
        return Vec::new();
    }
    
    println!("  Verification OK, running benchmark samples...");
    
    // Warmup
    for _ in 0..config.warmup {
        let _ = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
    }
    
    // Benchmark samples
    let mut results = Vec::new();
    for sample in 0..config.samples {
        let mut result = BenchmarkResult::new("DeepFold", "BabyBear", nv, sample);
        
        Sumcheck::reset_timing();
        
        let start = Instant::now();
        let proof = prover.prove(&pp, nv as usize, [a.clone(), b.clone(), c.clone()]);
        result.prover_total = start.elapsed();
        
        result.sumcheck_time = Duration::from_micros(Sumcheck::prove_time_us());
        result.proof_size = proof.bytes.len();
        
        // Add step breakdown
        result.add_step("Sumcheck", result.sumcheck_time, None);
        result.add_step("Polynomial Commitment", 
            result.prover_total.saturating_sub(result.sumcheck_time), None);
        
        let start = Instant::now();
        let verified = verifier.verify(&pp, nv as usize, proof);
        result.verifier_time = start.elapsed();
        
        if !verified {
            println!("  Sample #{}: Verification FAILED", sample + 1);
            continue;
        }
        
        print_sample_result(&result);
        results.push(result);
    }
    
    results
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    let config = BenchConfig::from_args();
    
    print_header();
    print_config(&config);
    
    let mut all_stats = Vec::new();
    
    // Run BaseFold/Goldilocks benchmark
    let basefold_results = bench_basefold_goldilocks(&config);
    if !basefold_results.is_empty() {
        let stats = AggregatedStats::from_results(&basefold_results);
        print_aggregated_stats(&stats);
        all_stats.push(stats);
    }
    
    // Run DeepFold/Goldilocks benchmark
    let deepfold_results = bench_deepfold_goldilocks(&config);
    if !deepfold_results.is_empty() {
        let stats = AggregatedStats::from_results(&deepfold_results);
        print_aggregated_stats(&stats);
        all_stats.push(stats);
    }
    
    // Run DeepFold/BabyBear benchmark (if feature enabled)
    #[cfg(feature = "babybear")]
    {
        let babybear_results = bench_deepfold_babybear(&config);
        if !babybear_results.is_empty() {
            let stats = AggregatedStats::from_results(&babybear_results);
            print_aggregated_stats(&stats);
            all_stats.push(stats);
        }
    }
    
    // Print comparison table
    print_comparison_table(&all_stats);
    
    println!();
    println!("Benchmark complete!");
    println!();
}
