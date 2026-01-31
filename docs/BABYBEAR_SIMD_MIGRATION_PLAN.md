# BabyBear SIMD 优化迁移计划

## 概述

本文档详细分析 Plonky3 中 BabyBear 域的 SIMD 并行指令集优化方案，并制定将其迁移到当前 DeepFold-Hyperplonk 项目的具体计划，替换现有的 Goldilocks64 实现。

---

## 1. 当前项目架构分析

### 1.1 项目结构
```
DeepFold-Hyperplonk/
├── arithmetic/           # 基础算术库
│   └── src/
│       ├── field/        # 有限域实现
│       │   ├── goldilocks64.rs   # 当前64位域
│       │   └── bn_254.rs         # BN254配对域
│       ├── mul_group.rs  # 乘法群
│       └── poly.rs       # 多项式
├── hyperplonk/           # Hyperplonk协议
│   └── src/
│       ├── sumcheck.rs   # Sumcheck协议
│       ├── prover.rs     # 证明者
│       └── verifier.rs   # 验证者
├── poly_commit/          # 多项式承诺
│   └── src/
│       ├── deepfold.rs   # DeepFold承诺
│       └── basefold.rs   # BaseFold承诺
└── util/                 # 工具库
    └── src/
        ├── fiat_shamir.rs
        └── merkle_tree.rs
```

### 1.2 当前 Goldilocks64 实现特点

**域参数：**
- 模数: `P = 2^64 - 2^32 + 1 = 18446744069414584321`
- 存储: 64位整数
- 乘法: 使用128位中间结果进行约简
- FFT阶: `LOG_ORDER = 32` (2-adicity)

**现有优化：**
- x86_64 汇编优化的加法 (`add_no_canonicalize_trashing_input`)
- 分支提示优化 (`branch_hint`)
- 128位约简优化

**局限性：**
- 没有 SIMD 向量化
- 单元素运算，无法利用现代 CPU 的并行能力
- 64位域运算本身开销较大

---

## 2. Plonky3 BabyBear SIMD 优化方案分析

### 2.1 BabyBear 域参数

```rust
// 31位素数域
const PRIME: u32 = 0x78000001;  // 2^31 - 2^27 + 1

// Montgomery参数
const MONTY_BITS: u32 = 32;
const MONTY_MU: u32 = 0x88000001;  // P^-1 mod 2^32

// 2-adicity = 27 (高于Goldilocks)
const TWO_ADICITY: usize = 27;
```

### 2.2 Montgomery 表示法

BabyBear 使用 Montgomery 表示法加速乘法：
- 标准值 `x` 存储为 `x * R mod P`，其中 `R = 2^32`
- 乘法约简只需单次操作，无需完整除法

**Montgomery 约简核心算法：**
```rust
// 输入: 0 <= C < P * B (B = 2^32)
// 输出: 0 <= R < P, R = C * B^-1 (mod P)
fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) & MONTY_MASK;
    let u = t * (PRIME as u64);
    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MONTY_BITS) as u32;
    let corr = if over { PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}
```

### 2.3 SIMD 并行化架构

Plonky3 提供三级 SIMD 支持：

| 平台 | 向量宽度 | 并行元素数 | 类型 |
|------|----------|------------|------|
| AVX2 | 256-bit | 8 个 u32 | `PackedMontyField31AVX2` |
| AVX512 | 512-bit | 16 个 u32 | `PackedMontyField31AVX512` |
| NEON (ARM) | 128-bit | 4 个 u32 | `PackedMontyField31Neon` |

### 2.4 AVX2 核心优化实现

**向量化乘法（8元素并行）：**
```rust
fn mul<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    // 分离奇偶位置元素
    let lhs_evn = lhs;
    let rhs_evn = rhs;
    let lhs_odd = movehdup_epi32(lhs);  // 复制奇数位到偶数位
    let rhs_odd = movehdup_epi32(rhs);
    
    // 并行Montgomery乘法
    let d_evn = monty_mul::<MPAVX2>(lhs_evn, rhs_evn);
    let d_odd = monty_mul::<MPAVX2>(lhs_odd, rhs_odd);
    
    // 混合结果
    blend_evn_odd(d_evn, d_odd)
}
```

**向量化Montgomery约简：**
```rust
fn partial_monty_red_unsigned_to_signed<MPAVX2>(input: __m256i) -> __m256i {
    unsafe {
        let q = _mm256_mul_epu32(input, MPAVX2::PACKED_MU);
        let q_p = _mm256_mul_epu32(q, MPAVX2::PACKED_P);
        _mm256_sub_epi32(input, q_p)
    }
}
```

**向量化加法：**
```rust
fn add(self, rhs: Self) -> Self {
    let lhs = self.to_vector();
    let rhs = rhs.to_vector();
    let res = mm256_mod_add(lhs, rhs, PMP::PACKED_P);
    unsafe { Self::from_vector(res) }
}
```

### 2.5 优化点积实现

Plonky3 针对点积运算有特殊优化，延迟Montgomery约简：

```rust
fn dot_product_2<PMP>(lhs: [LHS; 2], rhs: [RHS; 2]) -> __m256i {
    // 累加乘积到64位中间值，延迟约简
    let dot_evn = _mm256_add_epi64(mul_evn0, mul_evn1);
    let dot_odd = _mm256_add_epi64(mul_odd0, mul_odd1);
    
    // 单次Montgomery约简
    let red_evn = partial_monty_red_unsigned_to_signed::<PMP>(dot_evn);
    let red_odd = partial_monty_red_unsigned_to_signed::<PMP>(dot_odd);
    
    blend_evn_odd(red_evn, red_odd)
}
```

这种方法将约简次数从 2N 降低到 2，显著提升性能。

---

## 3. 性能对比分析

### 3.1 理论性能比较

| 操作 | Goldilocks64 (当前) | BabyBear SIMD (目标) | 加速比 |
|------|---------------------|---------------------|--------|
| 加法 | 1 op/cycle | 8-16 op/cycle | 8-16x |
| 乘法 | ~10 cycles | ~3 cycles (8并行) | ~25x |
| 点积(N) | N次乘法+约简 | N/8乘法 + 1次约简 | >10x |
| FFT蝶形 | 逐元素 | 向量化 | 8-16x |

### 3.2 为什么选择 BabyBear

1. **31位域适合SIMD**: 
   - 32位整数可直接用 `_mm256_mul_epu32` 操作
   - 64位中间结果不溢出

2. **Montgomery友好**:
   - `MONTY_BITS = 32` 对齐机器字
   - 约简只需位移和减法

3. **高2-adicity (27)**:
   - 支持大规模FFT (2^27点)
   - 比Goldilocks的32略低但足够

4. **扩展域支持**:
   - 支持4/5/8次扩展
   - `BinomialExtensionData` 预计算

---

## 4. 迁移实施计划

### 阶段一：基础架构 (1-2周)

#### 4.1 创建新的域模块结构
```
arithmetic/src/field/
├── mod.rs              # 统一导出
├── traits.rs           # Field/FftField trait定义
├── goldilocks64.rs     # 保留（兼容）
├── bn_254.rs           # 保留
└── babybear/           # 新增
    ├── mod.rs          # 模块定义
    ├── babybear.rs     # 基础实现
    ├── extension.rs    # 扩展域
    ├── monty.rs        # Montgomery工具
    └── simd/           # SIMD优化
        ├── mod.rs
        ├── avx2.rs     # AVX2实现
        ├── avx512.rs   # AVX512实现
        └── neon.rs     # ARM NEON实现
```

#### 4.2 定义核心 Trait

```rust
// arithmetic/src/field/traits.rs
pub trait MontyParameters: Copy + Default + Debug + Eq + Send + Sync + 'static {
    const PRIME: u32;
    const MONTY_BITS: u32;
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;
}

pub trait PackedMontyParameters: MontyParameters {
    #[cfg(target_arch = "x86_64")]
    type PackedType;
}

pub trait FieldParameters: PackedMontyParameters {
    const MONTY_ZERO: MontyField31<Self>;
    const MONTY_ONE: MontyField31<Self>;
    const MONTY_GEN: MontyField31<Self>;
}
```

### 阶段二：BabyBear 核心实现 (2-3周)

#### 4.3 实现 BabyBear 基础域

```rust
// arithmetic/src/field/babybear/babybear.rs
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct BabyBear {
    pub(crate) value: u32,  // Montgomery表示
}

impl BabyBearParameters {
    pub const PRIME: u32 = 0x78000001;
    pub const MONTY_BITS: u32 = 32;
    pub const MONTY_MU: u32 = 0x88000001;
}

impl BabyBear {
    #[inline(always)]
    pub const fn new(value: u32) -> Self {
        Self { value: to_monty(value) }
    }
    
    #[inline(always)]
    pub(crate) const fn new_monty(value: u32) -> Self {
        Self { value }
    }
}
```

#### 4.4 实现 Montgomery 运算

```rust
// arithmetic/src/field/babybear/monty.rs
#[inline]
pub const fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % PRIME as u64) as u32
}

#[inline]
pub const fn from_monty(x: u32) -> u32 {
    monty_reduce(x as u64)
}

#[inline]
pub const fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) & MONTY_MASK;
    let u = t * (PRIME as u64);
    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> 32) as u32;
    if over { x_sub_u_hi.wrapping_add(PRIME) } else { x_sub_u_hi }
}
```

### 阶段三：SIMD 向量化实现 (3-4周)

#### 4.5 AVX2 打包类型

```rust
// arithmetic/src/field/babybear/simd/avx2.rs
use core::arch::x86_64::*;

const WIDTH: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedBabyBearAVX2(pub [BabyBear; WIDTH]);

impl PackedBabyBearAVX2 {
    #[inline]
    pub(crate) fn to_vector(self) -> __m256i {
        unsafe { transmute(self) }
    }

    #[inline]
    pub(crate) unsafe fn from_vector(vector: __m256i) -> Self {
        transmute(vector)
    }
}
```

#### 4.6 向量化算术运算

```rust
impl Add for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.to_vector();
            let rhs = rhs.to_vector();
            let sum = _mm256_add_epi32(lhs, rhs);
            let sub = _mm256_sub_epi32(sum, PACKED_P);
            let res = _mm256_min_epu32(sum, sub);
            Self::from_vector(res)
        }
    }
}

impl Mul for PackedBabyBearAVX2 {
    type Output = Self;
    
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let lhs = self.to_vector();
            let rhs = rhs.to_vector();
            let res = simd_mul_avx2(lhs, rhs);
            Self::from_vector(res)
        }
    }
}
```

### 阶段四：数据拆分与批处理优化 (2-3周)

#### 4.7 数据拆分策略

针对 Sumcheck 和 DeepFold 的核心循环，实现数据拆分并行：

```rust
// 原始循环（单元素）
for i in 0..n {
    result[i] = a[i] * b[i] + c[i];
}

// SIMD优化（8元素批处理）
fn process_batch_avx2(
    a: &[BabyBear],
    b: &[BabyBear],
    c: &[BabyBear],
    result: &mut [BabyBear]
) {
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = PackedBabyBearAVX2::from_slice(&a[idx..]);
        let b_vec = PackedBabyBearAVX2::from_slice(&b[idx..]);
        let c_vec = PackedBabyBearAVX2::from_slice(&c[idx..]);
        let res = a_vec * b_vec + c_vec;
        res.store(&mut result[idx..]);
    }
    // 处理尾部
    for i in (chunks * 8)..a.len() {
        result[i] = a[i] * b[i] + c[i];
    }
}
```

#### 4.8 Sumcheck 优化

```rust
// hyperplonk/src/sumcheck.rs 优化版本
impl Sumcheck {
    pub fn prove_simd<F: PackedField>(
        mut evals: [Vec<F>; N],
        transcript: &mut Transcript,
    ) -> (Vec<F>, [F; N]) {
        // 使用PackedField批量处理
        let packed_evals: Vec<F::Packing> = evals
            .iter()
            .map(|v| F::Packing::pack_slice(v))
            .collect();
        
        // 向量化外推和求和
        // ...
    }
}
```

### 阶段五：集成与测试 (2周)

#### 4.9 适配现有接口

```rust
// arithmetic/src/field/mod.rs
#[cfg(feature = "babybear")]
pub use babybear::{BabyBear, BabyBearExt, PackedBabyBear};

#[cfg(not(feature = "babybear"))]
pub use goldilocks64::{Goldilocks64, Goldilocks64Ext};
```

#### 4.10 测试用例

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_babybear_basic() {
        let a = BabyBear::new(12345);
        let b = BabyBear::new(67890);
        assert_eq!((a + b) - b, a);
        assert_eq!(a * BabyBear::one(), a);
    }
    
    #[test]
    fn test_packed_mul() {
        let a = PackedBabyBearAVX2::from_fn(|i| BabyBear::new(i as u32));
        let b = PackedBabyBearAVX2::from_fn(|i| BabyBear::new((i + 1) as u32));
        let c = a * b;
        for i in 0..8 {
            assert_eq!(c.0[i], BabyBear::new(i as u32) * BabyBear::new((i + 1) as u32));
        }
    }
}
```

---

## 5. 项目结构重组

### 5.1 最终目录结构

```
DeepFold-Hyperplonk/
├── arithmetic/
│   ├── Cargo.toml          # 添加 SIMD feature flags
│   └── src/
│       ├── lib.rs
│       ├── field/
│       │   ├── mod.rs
│       │   ├── traits.rs           # 统一trait定义
│       │   ├── goldilocks64.rs     # 兼容保留
│       │   ├── bn_254.rs
│       │   └── babybear/
│       │       ├── mod.rs
│       │       ├── babybear.rs     # 核心31位域
│       │       ├── extension.rs    # 扩展域 (4/5/8次)
│       │       ├── monty.rs        # Montgomery工具
│       │       ├── two_adic.rs     # 2-adic数据
│       │       └── simd/
│       │           ├── mod.rs      # 条件编译入口
│       │           ├── avx2.rs     # x86_64 AVX2
│       │           ├── avx512.rs   # x86_64 AVX512
│       │           ├── neon.rs     # ARM NEON
│       │           └── fallback.rs # 无SIMD回退
│       ├── mul_group.rs
│       └── poly.rs
├── hyperplonk/
│   └── src/
│       ├── sumcheck.rs         # 添加SIMD路径
│       └── ...
├── poly_commit/
│   └── src/
│       ├── deepfold.rs         # 添加SIMD路径
│       └── ...
└── docs/
    └── BABYBEAR_SIMD_MIGRATION_PLAN.md
```

### 5.2 Cargo.toml 配置

```toml
# arithmetic/Cargo.toml
[package]
name = "arithmetic"
version = "0.1.0"
edition = "2021"

[features]
default = ["babybear", "simd"]
babybear = []
goldilocks = []
simd = []
avx2 = ["simd"]
avx512 = ["simd"]
neon = ["simd"]

[dependencies]
rand = "0.8"

[target.'cfg(target_arch = "x86_64")'.dependencies]
# x86 intrinsics are in std

[target.'cfg(target_arch = "aarch64")'.dependencies]
# ARM intrinsics are in std
```

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SIMD不兼容 | 部分平台无法使用 | 提供fallback实现 |
| 精度差异 | 证明可能不兼容 | 完整的正确性测试 |
| 性能回退 | 某些场景变慢 | 基准测试覆盖 |
| API变更 | 现有代码需修改 | 保持trait兼容 |

---

## 7. 时间线

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 基础架构设计 | 1-2周 |
| 2 | BabyBear核心实现 | 2-3周 |
| 3 | SIMD向量化 | 3-4周 |
| 4 | 数据拆分优化 | 2-3周 |
| 5 | 集成测试 | 2周 |
| **总计** | | **10-14周** |

---

## 8. 参考资源

1. [Plonky3 源码](https://github.com/Plonky3/Plonky3)
2. [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
3. [Modern Computer Arithmetic - Montgomery算法](https://members.loria.fr/PZimmermann/mca/mca-cup-0.5.9.pdf)
4. [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics)

---

## 附录：关键代码参考

### A. Plonky3 Montgomery乘法 (AVX2)

```rust
// 源码位置: ref/Plonky3/monty-31/src/x86_64_avx2/packing.rs
fn mul<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    let lhs_evn = lhs;
    let rhs_evn = rhs;
    let lhs_odd = movehdup_epi32(lhs);
    let rhs_odd = movehdup_epi32(rhs);

    let d_evn = monty_mul::<MPAVX2>(lhs_evn, rhs_evn);
    let d_odd = monty_mul::<MPAVX2>(lhs_odd, rhs_odd);

    blend_evn_odd(d_evn, d_odd)
}
```

### B. 当前Goldilocks64乘法

```rust
// 源码位置: arithmetic/src/field/goldilocks64.rs
impl std::ops::Mul for Goldilocks64 {
    type Output = Goldilocks64;
    fn mul(self, rhs: Self) -> Self::Output {
        reduce128((self.v as u128) * (rhs.v as u128))
    }
}
```
