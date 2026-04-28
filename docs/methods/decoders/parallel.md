# Parallel 并行解码策略


## 算法逻辑精要

Parallel并行解码在Vanilla基础上引入两种自适应解码机制以突破每步固定解码数量的限制：threshold-based模式直接选择所有置信度不低于预设阈值的token进行批量并行解码，factor-based模式则自适应地计算最优解码数量n，使得(n+1)×(1−第n高置信度)小于设定因子。两种机制均允许每步解码可变数量的token，在保持生成质量的同时显著提升解码速率。

## 概述

Parallel（并行）解码策略源自 Fast-dLLM 论文，是一种基于置信度阈值的并行解码方法。该策略通过设置置信度阈值，允许每步解码多个高置信度的 token，从而显著提升生成速度，同时保持生成质量。

## 算法原理

### 核心思想

传统 Vanilla 解码每步只解码固定数量的 token（通常为 1 个），这导致生成速度较慢。Parallel 解码的核心洞察是：

> **高置信度的 token 可以安全地并行解码，因为它们不太可能依赖其他未解码的 token。**

### 两种实现方式

#### 1. Threshold-based（基于阈值）

```python
# 选择所有置信度 >= threshold 的 token
selected_tokens = confidence >= threshold
```

- 直观且易于调节
- 每步解码的 token 数量动态变化
- 适合需要稳定质量保证的场景

#### 2. Factor-based（基于因子）

```python
# 找到最大的 n，使得 (n+1) * (1 - nth_confidence) < factor
for n in range(1, num_unmasked_tokens + 1):
    if (n + 1) * (1 - sorted_conf[n - 1]) >= factor:
        break
selected_tokens = top-n tokens
```

- 自适应调整解码数量
- 基于理论分析的速度-质量权衡
- 更适合追求最大加速比的场景

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallel 解码流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   单步解码过程                            │    │
│  │                                                         │    │
│  │  1. 模型前向传播                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  input: [prompt + MASK + ... + MASK]  │           │    │
│  │     │  output: logits for all positions     │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  2. 计算置信度                                           │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  confidence[i] = max(softmax(logits)) │           │    │
│  │     │                                       │           │    │
│  │     │  示例: [0.95, 0.72, 0.88, 0.45, 0.91] │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  3. 并行选择 (threshold=0.9)                             │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  confidence >= 0.9?                   │           │    │
│  │     │  [✓, ✗, ✗, ✗, ✓]                      │           │    │
│  │     │                                       │           │    │
│  │     │  选择位置: [0, 4]                      │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  4. 更新 Frame                                           │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  解码位置 0, 4 的 token                │           │    │
│  │     │  保留位置 1, 2, 3 为 MASK              │           │    │
│  │     └───────────────────────────────────────┘           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  重复直到所有位置解码完成                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Token 依赖关系分析

```
原始序列:    [A] [B] [C] [D] [E]
置信度:      0.95 0.72 0.88 0.45 0.91
阈值:        0.9

选择结果:
┌─────────────────────────────────────────┐
│ 位置 0 (A): ✓ 选中 (0.95 >= 0.9)        │
│ 位置 1 (B): ✗ 未选中 (0.72 < 0.9)       │
│ 位置 2 (C): ✗ 未选中 (0.88 < 0.9)       │
│ 位置 3 (D): ✗ 未选中 (0.45 < 0.9)       │
│ 位置 4 (E): ✓ 选中 (0.91 >= 0.9)        │
└─────────────────────────────────────────┘

解码后:      [A] [MASK] [MASK] [MASK] [E]
```

## 核心参数

### Threshold 模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | None | 置信度阈值，所有 >= 该值的 token 将被解码 |
| `num_transfer_tokens` | int | 1 | 最少解码的 token 数量（保底机制） |

### Factor 模式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `factor` | float | None | 速度因子，控制并行解码的激进程度 |

### 共享参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列总长度 |
| `block_length` | int | - | 块大小 |
| `temperature` | float | 0.0 | 采样温度 |
| `alg` | str | "maskgit_plus" | 置信度计算算法 |

## 详细代码流程分析

Parallel 解码的实现位于 `src/generation/vanilla.py` 的 `confidence_unmasking` 函数（L142-L269）中，通过 `threshold` 或 `factor` 参数激活。调用链路：`vanilla_generate` → `unmasking_fn`（闭包）→ `confidence_unmasking`。

### threshold-based 并行解码

```python
# 源文件: src/generation/vanilla.py L199-L211
if threshold is not None:
    # 1.a select all tokens whose confidence is above the threshold
    col_indices = torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]
    counts = torch.sum(confidence >= threshold, dim=-1).cpu().tolist()
    transfer_index = list(torch.split(col_indices, counts))
    # check if there are too many tokens to be decoded in any sequence
    # in this case, we only keep the top-k tokens with highest confidence
    # to do so, we simply clear the transfer_index for those sequences and fall back to top-k selection later
    for i, t in enumerate(transfer_index):
        if t.numel() > max_transfer_tokens[i]:
            transfer_index[i] = torch.tensor([])
            num_transfer_tokens[i] = max_transfer_tokens[i]
```

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L200 | `torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]` — 找出所有 batch 中置信度达到阈值的列索引，`[:, 1]` 取位置维，得到跨 batch 拼接的一维张量 |
| L201 | `torch.sum(confidence >= threshold, dim=-1).cpu().tolist()` — 统计每个 batch 中满足条件的数量 |
| L202 | `torch.split(col_indices, counts)` — 按每个 batch 的数量将索引切分回各序列 |
| L204-L206 | 循环检查：若某序列选中数量超过 `max_transfer_tokens[i]`（当前块内可转移位置总数），清空该序列的 transfer_index 并将 `num_transfer_tokens` 上调为上限值，后续由 top-k 回退（L245-L267）补足 |

### factor-based 并行解码

```python
# 源文件: src/generation/vanilla.py L212-L224
elif factor is not None:
    # 1.b unmask top-n* tokens, where n* = argmax_{n} (n + 1)(1 - nth_largest_conf) < factor
    num_unmasked_tokens = torch.sum(transfer_index_mask, dim=-1, keepdim=True)
    for i in range(batch_size):
        sorted_conf, _ = torch.sort(
            confidence[i][transfer_index_mask[i]],
            dim=-1,
            descending=True,
        )
        for n in range(1, num_unmasked_tokens[i] + 1):
            if (n + 1) * (1 - sorted_conf[n - 1]) >= factor:
                break
        transfer_index[i] = torch.topk(confidence[i], min(n - 1, int(max_transfer_tokens[i].item())), dim=-1).indices
```

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L214 | `num_unmasked_tokens` — 每个序列当前块内可转移位置的数量 |
| L215-L218 | 对该序列可转移位置的置信度降序排序得到 `sorted_conf` |
| L219-L221 | 从 n=1 递增，当 `(n+1)*(1-sorted_conf[n-1]) >= factor` 时停止，`sorted_conf[n-1]` 是第 n 高的置信度 |
| L224 | `torch.topk(confidence[i], min(n-1, max_transfer_tokens), dim=-1)` — 取 top-(n-1) 个索引，用 `max_transfer_tokens` 做上限保护 |

### 保底回退机制

```python
# 源文件: src/generation/vanilla.py L239-L267
num_transfer_tokens = torch.clamp(
    num_transfer_tokens,
    min=min_transfer_tokens,
    max=max_transfer_tokens,
)

if fallback_indices := [
    i for i, t in enumerate(transfer_index) if t.numel() < num_transfer_tokens[i]
]:
    confidence_subset = confidence[fallback_indices]
    topk_transfer_index = [
        torch.topk(
            confidence_subset[i],
            int(
                torch.min(
                    transfer_index_mask[fallback_indices[i]].sum(),
                    num_transfer_tokens[fallback_indices[i]],
                )
            ),
            dim=-1,
        ).indices
        for i in range(confidence_subset.size(0))
    ]
    source_iter = iter(topk_transfer_index)
    transfer_index = [
        next(source_iter) if i in fallback_indices else t
        for i, t in enumerate(transfer_index)
    ]
```

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L239-L243 | `torch.clamp` 确保 `num_transfer_tokens` 介于 `min_transfer_tokens` 和 `max_transfer_tokens` 之间 |
| L245-L247 | `fallback_indices` — 收集 transfer_index 数量不足 `num_transfer_tokens` 的序列索引 |
| L248-L258 | 对需要回退的序列，取其置信度中 top-k 个最高分位置（k = `num_transfer_tokens[i]` 且不超过可转移位置总数） |
| L263-L266 | 合并结果：fallback_indices 中的位置取 topk_transfer_index 的对应值，其余保持原样 |

## Token 选择策略

### Threshold 模式选择逻辑

```
置信度分布示例:
位置:     0     1     2     3     4     5     6     7
置信度: 0.95  0.72  0.88  0.45  0.91  0.33  0.89  0.96
阈值:    0.9

选择结果:
         ✓     ✗     ✗     ✗     ✓     ✗     ✗     ✓
         
解码位置: [0, 4, 7]
```

### Factor 模式选择逻辑

```
因子计算示例 (factor=1.0):

位置:     0     1     2     3     4
置信度: 0.95  0.88  0.72  0.45  0.33
排序后: 0.95  0.88  0.72  0.45  0.33

计算 (n+1) * (1 - nth_conf):
n=1: (1+1) * (1-0.95) = 0.10 < 1.0 ✓
n=2: (2+1) * (1-0.88) = 0.36 < 1.0 ✓
n=3: (3+1) * (1-0.72) = 1.12 >= 1.0 ✗

选择: top-2 tokens (位置 0, 1)
```

## 使用示例

### 配置文件

```yaml
# configs/generation/vanilla.yaml (Parallel 模式)
strategy: vanilla

# Threshold 模式
threshold: 0.9

# 或 Factor 模式
factor: null

alg: "maskgit_plus"
gen_length: null
block_length: null
num_transfer_tokens: 1
temperature: 0.0
stop_until_eos: false
```

### 命令行使用

```bash
# Threshold 模式
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    generation.num_transfer_tokens=1 \
    generation.gen_length=256 \
    generation.block_length=32 \
    generation.threshold=0.9 \
    model=llada-inst

# Factor 模式
accelerate launch eval.py \
    generation=vanilla \
    generation.factor=1.0 \
    generation.gen_length=256 \
    generation.block_length=32 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.vanilla import vanilla_generate

# Threshold 模式
result = vanilla_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    threshold=0.9,  # 并行解码阈值
    num_transfer_tokens=1,
    mask_token_id=tokenizer.mask_token_id,
)

# Factor 模式
result = vanilla_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    factor=1.0,  # 速度因子
    mask_token_id=tokenizer.mask_token_id,
)
```

## 性能特点

### 优势

1. **显著加速**：每步可解码多个 token，大幅减少迭代次数
2. **质量保持**：置信度阈值确保只解码"安全"的 token
3. **自适应**：根据模型置信度动态调整解码数量
4. **理论支撑**：基于 token 依赖关系的理论分析

### 劣势

1. **参数敏感**：threshold/factor 的选择对性能影响较大
2. **潜在风险**：过早解码可能破坏 token 间的依赖关系
3. **质量波动**：不同样本的解码数量差异可能导致质量不稳定

### 速度提升分析

```
假设 gen_length=256, block_length=32

Vanilla (num_transfer_tokens=1):
- 每块需要 32 次迭代
- 总迭代次数: 8 * 32 = 256 次

Parallel (threshold=0.9, 平均每步解码 4 个 token):
- 每块约需要 8 次迭代
- 总迭代次数: 8 * 8 = 64 次
- 加速比: 256 / 64 = 4x
```

### 适用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 追求速度 | `threshold=0.8`, `factor=1.5` | 更激进的并行解码 |
| 平衡模式 | `threshold=0.9`, `factor=1.0` | 速度与质量的平衡 |
| 质量优先 | `threshold=0.95`, `factor=0.5` | 更保守的解码策略 |
| 配合 KV Cache | `threshold=0.9` | 最佳实践 |

## 与 KV Cache 的协同

Parallel 解码与 KV Cache 结合可实现最佳性能：

```python
# 使用 D2Cache 加速
from src.cache import D2Cache

result = vanilla_generate(
    model=model,
    input_ids=input_ids,
    threshold=0.9,
    cache_cls=D2Cache,  # 启用 KV Cache
)
```

KV Cache 的作用：
- 缓存已解码 token 的 KV 状态
- 避免重复计算，进一步加速
- 支持近似缓存，容忍少量精度损失

## 参数调优指南

### Threshold 调优

| Threshold | 预期效果 | 适用场景 |
|-----------|----------|----------|
| 0.95-1.0 | 保守，质量高 | 关键任务、代码生成 |
| 0.85-0.95 | 平衡 | 通用场景 |
| 0.7-0.85 | 激进，速度快 | 实时应用、批量处理 |

### Factor 调优

| Factor | 预期效果 | 适用场景 |
|--------|----------|----------|
| 0.5-1.0 | 保守 | 质量敏感任务 |
| 1.0-1.5 | 平衡 | 通用场景 |
| 1.5-2.0 | 激进 | 速度优先场景 |

## 参考文献

- [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618) - Fast-dLLM 论文
