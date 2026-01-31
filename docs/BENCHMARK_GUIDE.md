# DeepFold-Hyperplonk 基准测试指南

本文档详细说明 DeepFold-Hyperplonk 项目的端到端基准测试系统，包括当前实现状态、执行方法、指标含义和自定义参数方法。

---

## 1. 项目当前实现概述

### 1.1 项目架构

```
DeepFold-Hyperplonk/
├── arithmetic/           # 基础算术库
│   └── src/
│       ├── field/        # 有限域实现
│       │   ├── goldilocks64.rs   # 64位Goldilocks域
│       │   ├── bn_254.rs         # BN254配对友好域
│       │   └── babybear/         # 31位BabyBear域 (SIMD优化)
│       ├── mul_group.rs  # 乘法群 (FFT支持)
│       └── poly.rs       # 多项式运算
├── hyperplonk/           # Hyperplonk协议实现
│   ├── src/
│   │   ├── circuit.rs    # 电路定义
│   │   ├── prover.rs     # 证明者
│   │   ├── verifier.rs   # 验证者
│   │   ├── sumcheck.rs   # Sumcheck协议
│   │   └── prod_eq_check.rs  # 乘积等式检查
│   └── benches/          # 基准测试
│       ├── detailed_benchmark.rs  # 详细端到端基准测试
│       ├── basefold.rs   # BaseFold方案基准
│       ├── deepfold.rs   # DeepFold方案基准
│       └── piop.rs       # PIOP基准
├── poly_commit/          # 多项式承诺方案
│   └── src/
│       ├── basefold.rs   # BaseFold实现
│       ├── deepfold.rs   # DeepFold实现 (SIMD优化)
│       └── kzg.rs        # KZG承诺
└── util/                 # 工具库
    └── src/
        ├── fiat_shamir.rs    # Fiat-Shamir变换
        └── merkle_tree.rs    # Merkle树
```

### 1.2 支持的方案

| 方案 | 有限域 | SIMD支持 | 说明 |
|------|--------|----------|------|
| **BaseFold** | Goldilocks64 | ❌ | 标准BaseFold多项式承诺 |
| **DeepFold** | Goldilocks64 | ❌ | 改进的折叠方案 |
| **DeepFold** | BabyBear | ✅ AVX2 | SIMD优化版本 |

### 1.3 当前优化状态

根据阶段性技术报告 (2026-02-01)：

- **BabyBear SIMD核心**：已修复并验证 AVX2 Montgomery乘法
- **Sumcheck SIMD路径**：已在 `fold_next_domain` 中启用
- **DeepFold SIMD路径**：已在 `evaluate_next_domain` 中启用
- **Feature对齐**：所有crate支持统一的 feature flags

---

## 2. 执行基准测试

### 2.1 详细端到端基准测试（推荐）

新的详细基准测试提供全面的性能分析，包括步骤级时间分解和多方案对比。

```bash
# 使用默认参数运行（nv=16, 3个样本）
cargo bench -p hyperplonk --bench detailed_benchmark

# 指定变量数量（控制电路规模）
cargo bench -p hyperplonk --bench detailed_benchmark -- --nv 18

# 指定样本数量（更多样本=更准确的统计）
cargo bench -p hyperplonk --bench detailed_benchmark -- --samples 5

# 组合参数
cargo bench -p hyperplonk --bench detailed_benchmark -- --nv 16 --samples 5 --warmup 2

# 使用特定feature运行
cargo bench -p hyperplonk --bench detailed_benchmark --features "babybear simd avx2"
```

### 2.2 传统基准测试

```bash
# DeepFold方案
cargo bench -p hyperplonk --bench deepfold

# BaseFold方案
cargo bench -p hyperplonk --bench basefold

# PIOP基准
cargo bench -p hyperplonk --bench piop

# 域运算基准
cargo bench -p arithmetic --bench field_benchmark
```

### 2.3 启用SIMD优化

```bash
# 启用BabyBear + SIMD + AVX2
cargo bench -p hyperplonk --bench detailed_benchmark --features "babybear simd avx2"

# 仅使用Goldilocks（禁用BabyBear）
cargo bench -p hyperplonk --bench detailed_benchmark --features "goldilocks"
```

---

## 3. 基准测试输出解读

### 3.1 示例输出

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          DeepFold-Hyperplonk Detailed End-to-End Benchmark                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ Configuration                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Number of Variables (nv):           16                                     │
│  Number of Gates (2^nv):          65536                                     │
│  Samples per scheme:                  3                                     │
│  Warmup iterations:                   1                                     │
│  BaseFold queries:                  120                                     │
│  DeepFold queries:                   45                                     │
└─────────────────────────────────────────────────────────────────────────────┘

▶ Running BaseFold / Goldilocks64 (nv=16, 3 samples)...
  Sample #1: Prover=245.32 ms, Sumcheck=89.45 ms, Verifier=12.34 ms, Proof=156.42 KB
  Sample #2: Prover=243.18 ms, Sumcheck=88.92 ms, Verifier=12.21 ms, Proof=156.42 KB
  Sample #3: Prover=244.56 ms, Sumcheck=89.12 ms, Verifier=12.28 ms, Proof=156.42 KB

┌─────────────────────────────────────────────────────────────────────────────┐
│ BaseFold / Goldilocks64 (nv=16, gates=65536)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROVER TIMING (3 samples):
│   Mean:         244.35 ms                                              │
│   Min:          243.18 ms                                              │
│   Max:          245.32 ms                                              │
│   StdDev:         1072.45 μs                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ SUMCHECK TIMING:
│   Mean:          89.16 ms                                              │
│   Min:           88.92 ms                                              │
│   Max:           89.45 ms                                              │
│   % of Prover:    36.5%                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ VERIFIER TIMING:
│   Mean:          12.28 ms                                              │
│   Min:           12.21 ms                                              │
│   Max:           12.34 ms                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ PROOF SIZE:       156.42 KB                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ STEP BREAKDOWN:
│   Sumcheck                       89.16 ms ( 36.5%)                     │
│   Polynomial Commitment         155.19 ms ( 63.5%)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 对比摘要

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           COMPARISON SUMMARY                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Scheme/Field           │ Prover (mean) │ Sumcheck     │ Verifier    │ Proof  ║
╠════════════════════════╪═══════════════╪══════════════╪═════════════╪════════╣
║ BaseFold/Goldilocks64  │     244.35 ms │     89.16 ms │    12.28 ms │ 156 KB ║
║ DeepFold/Goldilocks64  │     198.42 ms │     72.34 ms │     8.92 ms │  89 KB ║
║ DeepFold/BabyBear      │     156.78 ms │     58.23 ms │     7.12 ms │  62 KB ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ SPEEDUP ANALYSIS (relative to first scheme)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ DeepFold/Goldilocks64 vs BaseFold/Goldilocks64:
│   Prover speedup:      1.23x                                             │
│   Sumcheck speedup:    1.23x                                             │
│   Verifier speedup:    1.38x                                             │
│   Proof size ratio:    0.57x                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ DeepFold/BabyBear vs BaseFold/Goldilocks64:
│   Prover speedup:      1.56x                                             │
│   Sumcheck speedup:    1.53x                                             │
│   Verifier speedup:    1.72x                                             │
│   Proof size ratio:    0.40x                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 指标详解

### 4.1 核心时间指标

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **Prover Total** | 证明者完整执行时间 | 🔴 最重要的性能指标 |
| **Sumcheck Time** | Sumcheck协议执行时间 | 🟡 热点分析关键指标 |
| **Verifier Time** | 验证者执行时间 | 🟢 通常远小于Prover |

### 4.2 Prover时间分解

```
Prover Total = Sumcheck + Polynomial Commitment

其中:
- Sumcheck: 包含多轮求和检查和域折叠
- Polynomial Commitment: 包含FFT、Merkle树构建、查询生成
```

### 4.3 统计指标

| 指标 | 含义 |
|------|------|
| **Mean** | 所有样本的算术平均值 |
| **Min** | 最快的执行时间（最佳情况） |
| **Max** | 最慢的执行时间（最坏情况） |
| **StdDev** | 标准差，衡量结果的稳定性 |
| **% of Prover** | 该步骤占Prover总时间的百分比 |

### 4.4 Proof Size

证明大小取决于：
- **变量数量 (nv)**: 更多变量 → 更大证明
- **查询数量**: 更多查询 → 更大证明（但更高安全性）
- **域元素大小**: Goldilocks64 (8字节) vs BabyBear (4字节)

### 4.5 加速比解读

| 加速比 | 含义 |
|--------|------|
| > 1.0x | 新方案更快 |
| = 1.0x | 性能相同 |
| < 1.0x | 新方案更慢（回归） |

---

## 5. 自定义基准测试参数

### 5.1 命令行参数

```bash
cargo bench -p hyperplonk --bench detailed_benchmark -- [OPTIONS]

选项:
  --nv <NUM>              变量数量 (默认: 16)
                          电路大小 = 2^nv 个门
                          建议范围: 12-22

  --samples <NUM>         每个方案的样本数 (默认: 3)
                          更多样本 = 更准确的统计
                          建议: 3-10

  --warmup <NUM>          预热迭代次数 (默认: 1)
                          避免首次运行的冷启动效应

  --basefold-queries <NUM>  BaseFold查询数 (默认: 120)
                            影响安全级别和证明大小

  --deepfold-queries <NUM>  DeepFold查询数 (默认: 45)
                            DeepFold需要更少的查询
```

### 5.2 Feature Flags

在 `Cargo.toml` 或命令行中配置：

```toml
# hyperplonk/Cargo.toml
[features]
default = ["babybear", "simd"]
babybear = ["arithmetic/babybear", "poly_commit/babybear"]
goldilocks = ["arithmetic/goldilocks", "poly_commit/goldilocks"]
simd = ["arithmetic/simd", "poly_commit/simd"]
avx2 = ["arithmetic/avx2", "poly_commit/avx2", "simd"]
avx512 = ["arithmetic/avx512", "poly_commit/avx512", "simd"]
neon = ["arithmetic/neon", "poly_commit/neon", "simd"]
```

```bash
# 示例：仅Goldilocks
cargo bench --bench detailed_benchmark --no-default-features --features "goldilocks"

# 示例：BabyBear + AVX2
cargo bench --bench detailed_benchmark --features "babybear simd avx2"
```

### 5.3 修改代码参数

编辑 `hyperplonk/benches/detailed_benchmark.rs` 中的 `BenchConfig`：

```rust
impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            nv: 16,                    // 变量数量
            samples: 3,                 // 样本数
            warmup: 1,                  // 预热次数
            basefold_queries: 120,      // BaseFold查询数
            deepfold_queries: 45,       // DeepFold查询数
        }
    }
}
```

---

## 6. 样本含义与解读

### 6.1 为什么需要多个样本？

单次运行可能受到以下因素影响：
- **CPU频率波动**: 热节流、睿频加速
- **缓存状态**: 冷缓存 vs 热缓存
- **系统负载**: 后台进程干扰
- **内存分配**: 碎片化程度

多样本统计可以提供：
- **均值 (Mean)**: 典型性能
- **最小值 (Min)**: 最佳可达性能
- **最大值 (Max)**: 最差情况
- **标准差 (StdDev)**: 结果稳定性

### 6.2 建议的样本数

| 目的 | 建议样本数 |
|------|-----------|
| 快速检查 | 1-2 |
| 开发迭代 | 3-5 |
| 正式评测 | 10+ |
| 论文数据 | 20+ |

### 6.3 预热 (Warmup) 的作用

预热运行会：
- 加载代码到指令缓存
- 预热分支预测器
- 分配并初始化内存池
- JIT优化（如果有）

**建议**：至少1次预热，避免首次运行的异常高耗时。

---

## 7. 常见问题

### Q1: 为什么SIMD版本没有明显加速？

**可能原因**：
1. **数据量不足**: SIMD优势在大规模数据处理时更明显
2. **内存瓶颈**: 内存带宽而非计算成为瓶颈
3. **非SIMD热点**: 其他步骤（如Merkle树）未优化

**解决方案**：
- 增大 `nv` 参数
- 检查 feature flags 是否正确启用
- 使用 profiler 定位实际热点

### Q2: 如何对比不同机器的结果？

**建议**：
1. 记录CPU型号和频率
2. 记录内存大小和速度
3. 使用相同的 `nv` 和 `samples`
4. 对比**相对加速比**而非绝对时间

### Q3: 证明大小为什么重要？

证明大小影响：
- **链上成本**: 更小的证明 = 更低的gas费用
- **带宽需求**: 网络传输开销
- **存储成本**: 长期存储的费用

### Q4: 如何添加新的基准测试方案？

在 `detailed_benchmark.rs` 中添加新函数：

```rust
fn bench_new_scheme(config: &BenchConfig) -> Vec<BenchmarkResult> {
    // 1. 设置电路和参数
    // 2. 创建Prover和Verifier
    // 3. 生成witness
    // 4. 运行warmup
    // 5. 收集样本数据
    // 6. 返回结果
}
```

然后在 `main()` 中调用并添加到 `all_stats`。

---

## 8. 技术背景

### 8.1 Hyperplonk协议流程

```
1. Setup Phase
   ├── 生成电路描述 (permutation, selector)
   └── 生成多项式承诺参数

2. Prover Phase
   ├── 计算witness多项式
   ├── 生成多项式承诺
   ├── Sumcheck协议 (多轮交互)
   │   ├── 每轮发送多项式系数
   │   └── 接收随机挑战
   ├── ProdEqCheck (乘积等式验证)
   └── 多项式开放证明

3. Verifier Phase
   ├── 验证Sumcheck证明
   ├── 验证ProdEqCheck
   └── 验证多项式开放
```

### 8.2 DeepFold vs BaseFold

| 特性 | BaseFold | DeepFold |
|------|----------|----------|
| 折叠轮数 | 标准 | 优化 |
| 查询数量 | 120 | 45 |
| 证明大小 | 较大 | 较小 |
| 验证复杂度 | 较高 | 较低 |

### 8.3 BabyBear SIMD优势

- **31位域**: 适合32位SIMD操作
- **AVX2**: 同时处理8个域元素
- **Montgomery表示**: 快速模乘
- **适用场景**: Sumcheck折叠、DeepFold域运算

---

## 9. 相关文档

- `docs/BABYBEAR_SIMD_MIGRATION_PLAN.md` - BabyBear SIMD迁移计划
- `docs/TECHNICAL_ANALYSIS.md` - 技术深度解析
- `docs/STAGE_TECH_REPORT_2026-02-01.md` - 阶段性技术报告

---

*文档版本: 2026-02-01*
*作者: Cascade*
