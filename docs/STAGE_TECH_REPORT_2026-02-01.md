# 阶段性技术报告（2026-02-01）

## 1. 概要
本阶段完成 BabyBear SIMD (AVX2) 的核心路径修复与集成，并在 Sumcheck / DeepFold 两条关键路径中增加 BabyBear SIMD 快速路径。所有改动在 AVX2 环境下完成单元测试验证，并完成基准测试以对比标量与 SIMD 版本性能。

## 2. 主要交付
### 2.1 BabyBear SIMD 核心修复
- 修复 AVX2 Montgomery 乘法的借位纠正与结果规范化
- 修复 AVX2 模加约简逻辑
- 新增 AVX2 vs 标量回退一致性测试

### 2.2 BabyBear 扩展域修复
- 修复 `from_uniform_bytes` 的字节拆分逻辑，使其真实使用输入 32 字节

### 2.3 Sumcheck SIMD 路径
- 在 `fold_next_domain` 中增加 BabyBear SIMD 路径，并通过 `TypeId` 动态分流到 AVX2。

### 2.4 DeepFold SIMD 路径
- 在 `evaluate_next_domain` 中增加 BabyBear SIMD 路径，按 PackedBabyBear 批处理。

### 2.5 Feature 对齐
- 为 `arithmetic / poly_commit / hyperplonk` 增加一致的 feature flags（babybear/simd/avx2/avx512/neon）。

### 2.6 生命周期约束修复
- Sumcheck 的 `F: 'static` 约束已级联到 `ProdEqCheck` 与 `Prover` 以满足 SIMD TypeId 分流需要。

## 3. 关键文件变更（摘录）
- `arithmetic/src/field/babybear/simd/avx2.rs`
- `arithmetic/src/field/babybear/extension.rs`
- `hyperplonk/src/sumcheck.rs`
- `poly_commit/src/deepfold.rs`
- `arithmetic/Cargo.toml`, `poly_commit/Cargo.toml`, `hyperplonk/Cargo.toml`
- `arithmetic/benches/field_benchmark.rs`

## 4. 测试结果（AVX2 环境）
已执行：
- `cargo test -p arithmetic babybear::simd::avx2` ✅
- `cargo test -p poly_commit` ✅
- `cargo test -p hyperplonk` ✅

## 5. 基准测试结果（摘要）
`cargo bench -p arithmetic --bench field_benchmark`
- `goldilocks_scalar`: ~1.76 ns
- `babybear_scalar`: ~0.24 ns
- `babybear_packed_scalar_8x`: ~1.82 ns
- `babybear_packed_avx2_8x`: ~2.09 ns

结论：当前 benchmark 下 SIMD 8x 未体现明显优势，单 lane 折算约 0.26ns，和标量接近。后续建议提高批量规模（例如每次迭代执行多次 packed 运算或在协议级基准中对比）。

## 6. 已知警告（未影响功能）
- nil/basefold/shuffle 等模块存在历史 unused 警告
- goldilocks64 与 babybear 中部分未使用常量/函数（需后续清理）

## 7. 风险与注意事项
- TypeId 分流依赖 `F: 'static`，已在调用链上补充
- SIMD 路径仅在启用 `babybear + simd + avx2` 时生效

## 8. 下一阶段建议
- Sumcheck 外推（extrapolation）逻辑 SIMD 化
- DeepFold/Hyperplonk 端到端 benchmark 对比（basefold/deepfold/piop）
- 清理历史 unused 警告

---
报告人：Cascade
