# 基准测试分析报告 (2026-02-01)

## 1. 测试环境

- **测试日期**: 2026-02-01
- **测试参数**: 
  - `nv = 16` (65,536 个门)
  - `samples = 3`
  - `warmup = 1`
  - BaseFold queries: 120
  - DeepFold queries: 45

---

## 2. 测试结果摘要

| 方案 | Prover (平均) | Sumcheck | Verifier | Proof Size |
|------|---------------|----------|----------|------------|
| **BaseFold / Goldilocks64** | 872.86 ms | 607.32 ms (69.6%) | 5.45 ms | 294.81 KB |
| **DeepFold / Goldilocks64** | 779.05 ms | 517.03 ms (66.4%) | 4.13 ms | 150.89 KB |
| **DeepFold / BabyBear** | 1.06 s | 692.26 ms (65.5%) | 4.87 ms | 146.37 KB |

---

## 3. 性能分析

### 3.1 DeepFold vs BaseFold (Goldilocks64)

| 指标 | 加速比/比率 | 说明 |
|------|-------------|------|
| **Prover** | 1.12x | DeepFold比BaseFold快12% |
| **Sumcheck** | 1.17x | Sumcheck协议快17% |
| **Verifier** | 1.32x | 验证速度提升32% |
| **Proof Size** | 0.51x | 证明大小减少约一半 |

### 3.2 BabyBear vs Goldilocks64 (DeepFold)

| 指标 | 比率 | 说明 |
|------|------|------|
| **Prover** | 0.74x | BabyBear比Goldilocks64慢26% |
| **Sumcheck** | 0.75x | Sumcheck慢25% |
| **Verifier** | 0.85x | 验证慢15% |
| **Proof Size** | 0.97x | 证明大小相近 |

### 3.3 BabyBear 性能分析

**为什么 BabyBear 当前比 Goldilocks64 慢？**

1. **扩展域开销**: BabyBearExt4 是4次扩展域，每次乘法需要16次基域乘法 + 多次加法，而 Goldilocks64Ext 是2次扩展
2. **Montgomery 表示转换**: BabyBear 使用 Montgomery 形式，需要额外的转换开销
3. **SIMD 路径未完全覆盖**: 当前 SIMD 优化仅在 `fold_next_domain` 和 `evaluate_next_domain` 中启用，Sumcheck 的外推逻辑仍是标量计算
4. **扩展域 FFT**: BabyBearExt4 的 FFT 使用基域根但在扩展域上运算，效率较低

**优化机会**:
- Sumcheck 外推逻辑 SIMD 化 (占66%时间)
- 使用 BabyBear 基域直接计算（某些场景可避免扩展域）
- 多项式承诺中的批量 SIMD 处理

---

## 4. 时间分解

| 步骤 | BaseFold/Gold | DeepFold/Gold | DeepFold/Baby |
|------|---------------|---------------|---------------|
| **Sumcheck** | 607 ms (69.6%) | 517 ms (66.4%) | 692 ms (65.5%) |
| **Poly Commit** | 266 ms (30.4%) | 262 ms (33.6%) | 365 ms (34.5%) |

**关键发现**: Sumcheck协议占Prover总时间的 **65-70%**，是主要的性能瓶颈。

---

## 5. 结论与建议

### 5.1 DeepFold 优势明确
- **性能提升**: Prover快12%，Verifier快32%
- **证明更小**: 减少约50%的证明大小
- **更少查询**: 45 vs 120 查询，降低通信开销

### 5.2 BabyBear 集成已完成 ✅
- **端到端验证**: 已修复并通过
- **问题根因**: 置换偏移量需要与 prover/verifier 硬编码值 `1<<29`, `1<<30` 匹配
- **当前性能**: 比 Goldilocks64 慢，需要进一步优化

### 5.3 优化建议

1. **高优先级**:
   - Sumcheck 外推逻辑 SIMD 化 (占66%时间，8x并行潜力)
   - 考虑使用 BabyBear 基域而非扩展域（某些协议变体）

2. **中优先级**:
   - 多项式承诺的 Merkle 树并行化
   - 批量 Montgomery 转换

3. **长期**:
   - AVX-512 支持 (16元素并行)
   - 探索更低次扩展域 (如 BabyBear 2次扩展)

---

## 6. 原始数据

### BaseFold / Goldilocks64
```
Sample #1: Prover=882.67 ms, Sumcheck=612.96 ms, Verifier=5.64 ms, Proof=294.81 KB
Sample #2: Prover=865.05 ms, Sumcheck=603.43 ms, Verifier=5.26 ms, Proof=294.81 KB
Sample #3: Prover=870.86 ms, Sumcheck=605.58 ms, Verifier=5.45 ms, Proof=294.81 KB
Mean: 872.86 ms
```

### DeepFold / Goldilocks64
```
Sample #1: Prover=781.26 ms, Sumcheck=518.91 ms, Verifier=4.37 ms, Proof=150.89 KB
Sample #2: Prover=778.12 ms, Sumcheck=516.25 ms, Verifier=3.89 ms, Proof=150.89 KB
Sample #3: Prover=777.76 ms, Sumcheck=515.94 ms, Verifier=4.13 ms, Proof=150.89 KB
Mean: 779.05 ms
```

### DeepFold / BabyBear ✅
```
Sample #1: Prover=1.05 s, Sumcheck=688.69 ms, Verifier=4.13 ms, Proof=146.37 KB
Sample #2: Prover=1.05 s, Sumcheck=692.89 ms, Verifier=4.41 ms, Proof=146.37 KB
Sample #3: Prover=1.08 s, Sumcheck=695.21 ms, Verifier=6.08 ms, Proof=146.37 KB
Mean: 1.06 s
```

---

## 7. 后续工作

- [x] 调试 BabyBear + Hyperplonk 端到端验证 ✅
- [ ] Sumcheck SIMD 外推优化 (最大优化潜力)
- [ ] 添加更大规模测试 (nv=18, 20, 22)
- [ ] 探索 BabyBear 基域直接计算方案

---

*报告生成: Cascade*
*测试命令: `cargo bench -p hyperplonk --bench detailed_benchmark`*
