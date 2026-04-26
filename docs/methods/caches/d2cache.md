# d2Cache 缓存方法详解

## 算法逻辑精要

d2Cache 通过三种互补机制智能地选择每步需要计算的位置：其一，置信度密度（Certainty Density）利用高斯核评估已生成区域的确定性分布，据此选出高置信度的掩码位置优先解码；其二，Attention Rollout 在各层累积注意力流以追踪全局语义重要性，借助 `nucleus_select` 按累积概率筛选高注意力位置，确保语义关键区域不被遗漏；其三，窗口膨胀（`inflate_w`）自动填充选中位置之间的小间隙，避免因过度稀疏选择导致生成不连贯。其核心调度是：在 `on_step_end` 中综合置信度密度分数与注意力全局重要性构建 `q_mask`，在 `model_forward` 中通过 `top_up_mask` 实现批次内位置数均衡，在 `attention` 中按掩码更新 KV 缓存并逐层累积注意力展开矩阵。

## 概述

d2Cache 是 d2Cache 项目中最先进、最复杂的缓存实现。它在 PrefixCache 和 dLLM-Cache 的基础上，引入了基于注意力权重分析和置信度密度的智能选择机制。d2Cache 通过 Attention Rollout 算法追踪全局注意力流，结合置信度密度评估，动态选择最优的计算位置，从而在保证生成质量的同时实现最高的计算效率。

## 算法原理和理论基础

### 核心创新

d2Cache 的核心创新包括三个关键机制：

1. **Attention Rollout**：通过累积注意力权重追踪全局注意力流，识别语义重要的位置
2. **置信度密度（Certainty Density）**：使用高斯核评估已生成区域的置信度分布，指导选择策略
3. **智能位置选择**：结合注意力重要性和置信度密度，动态选择最优的计算位置

### 理论基础

#### Attention Rollout 算法

Attention Rollout 源自论文 "Quantifying Attention Flow in Transformers"。其核心思想是：

- 在 Transformer 中，信息通过注意力机制在层间流动
- 通过累积各层的注意力权重，可以追踪从输入到输出的信息流
- 高注意力流的位置对最终输出影响更大，应该优先刷新

数学表示：
```
A_rollout^(l) = (A^(l) + I) @ A_rollout^(l-1)
```
其中 `A^(l)` 是第 l 层的平均注意力权重矩阵，`I` 是单位矩阵。

#### 置信度密度计算

置信度密度使用高斯核通过 FFT 计算局部置信度分布：

```
density(x) = Σ confidence(y) * G(x - y, σ) / Σ G(x - y, σ)
```

其中 `G(x, σ)` 是标准差为 σ 的高斯核。高置信度密度区域表示周围已生成内容较为确定。

### 工作流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           d2Cache 完整工作流程                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         步骤开始 (on_step_end)                            │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     1. 更新置信度缓存 (_conf_cache)                       │  │
│  │                     2. 计算剩余掩码位置                                    │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     3. 确定搜索范围 (search_mask)                         │  │
│  │                        基于 current_k 参数                                 │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     4. 计算置信度密度分数                                  │  │
│  │                     scores = conf * certainty_density(~remaining)        │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     5. 选择 top-k 高分位置 (selected_mask)                │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     6. 添加新生成标记位置 (transfer tokens)               │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     7. 基于 Attention Rollout 选择全局位置               │  │
│  │                     q_mask |= nucleus_select(rollout, rollout_p)         │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     8. 窗口膨胀 (inflate_w)                               │  │
│  │                     填充选中位置之间的间隙                                  │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     9. 更新 _full_q_mask 和 _global_importance            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         前向传播 (model_forward)                          │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     10. top_up_mask 确保批次均衡                          │  │
│  │                     优先选择高重要性位置                                    │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     11. 选择性计算 Q、K、V                                 │  │
│  └─────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     12. 累积 Attention Rollout                            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 核心数据结构和参数说明

### 类定义

```python
class d2Cache(dCache):
    def __init__(
        self,
        model_config,
        rollout_p: float = 0.1,
        current_k: int = 32,
        sigma: float = 10.0,
        inflate_w: int = 4,
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache: torch.Tensor | None = None
        self._full_q_mask: torch.Tensor | None = None
        self._density_score: torch.Tensor
        self._global_importance: torch.Tensor
        self.rollout_p = rollout_p
        self.current_k = current_k
        self.sigma = sigma
        self.inflate_w = inflate_w
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_config` | dict | 必需 | 模型配置对象 |
| `rollout_p` | float | 0.1 | 基于 Attention Rollout 选择的位置比例 (0-1) |
| `current_k` | int | 32 | 每步选择的高置信度位置数量 |
| `sigma` | float | 10.0 | 置信度密度计算的高斯核标准差 |
| `inflate_w` | int | 4 | 窗口膨胀大小，填充选中位置间的间隙 |

### 核心数据结构

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `key_cache` | list[Tensor] | (B, T, head_dim) | 每层 Key 状态缓存 |
| `value_cache` | list[Tensor] | (B, T, head_dim) | 每层 Value 状态缓存 |
| `_conf_cache` | Tensor | (B, G) | 置信度分数缓存（G = 生成长度） |
| `_full_q_mask` | Tensor | (B, T) | 完整的查询位置掩码 |
| `_density_score` | Tensor | (B, G) | 置信度密度分数 |
| `_global_importance` | Tensor | (B, T) | 全局注意力重要性分数 |
| `_attn_rollout` | Tensor | (B, T, T) | 累积的注意力展开矩阵 |

### 参数调优指南

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              参数调优指南                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  rollout_p (Attention Rollout 选择比例):                                         │
│  ├── 较大值 (0.2-0.5): 选择更多全局重要位置，生成更稳定但计算更多               │
│  ├── 中等值 (0.1-0.2): 平衡性能和稳定性，推荐默认值                             │
│  └── 较小值 (0.0-0.1): 最小化计算，依赖置信度密度选择                           │
│                                                                                 │
│  current_k (每步选择位置数):                                                     │
│  ├── 较大值 (64-128): 每步生成更多内容，生成速度更快但计算更多                  │
│  ├── 中等值 (32-64): 平衡生成速度和计算效率，推荐默认值                         │
│  └── 较小值 (8-32): 更精细的生成控制，适合质量敏感场景                          │
│                                                                                 │
│  sigma (置信度密度标准差):                                                       │
│  ├── 较大值 (15-30): 更平滑的密度分布，考虑更大范围的上下文                     │
│  ├── 中等值 (5-15): 平衡局部和全局信息，推荐默认值                              │
│  └── 较小值 (1-5): 更局部的密度评估，更敏感于邻近位置                           │
│                                                                                 │
│  inflate_w (窗口膨胀大小):                                                       │
│  ├── 较大值 (8-16): 填充更大的间隙，生成更连贯但计算更多                        │
│  ├── 中等值 (2-8): 平衡连贯性和效率，推荐默认值                                 │
│  └── 较小值 (0-2): 最小化填充，更精确的位置选择                                 │
│      └── inflate_w=0 时完全禁用窗口膨胀                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 详细代码流程分析

以下按源码文件 [`src/cache/d2cache.py`](file:///Users/lier/codes/d2Cache/src/cache/d2cache.py) 的模块顺序，逐方法展开分析。

### `__init__` — 初始化

```python
# 源文件: src/cache/d2cache.py L18-L36
def __init__(
    self,
    model_config,
    rollout_p: float = 0.1,
    current_k: int = 32,
    sigma: float = 10.0,
    inflate_w: int = 4,
):
    super().__init__(model_config)
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    self._conf_cache: torch.Tensor | None = None
    self._full_q_mask: torch.Tensor | None = None
    self._density_score: torch.Tensor
    self._global_importance: torch.Tensor
    self.rollout_p = rollout_p
    self.current_k = current_k
    self.sigma = sigma
    self.inflate_w = inflate_w
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L18-L25 | `def __init__(self, model_config, rollout_p, current_k, sigma, inflate_w):` | 构造函数。`rollout_p` 控制基于 Attention Rollout 的 nucleus 选择比例；`current_k` 控制每步基于置信度密度选择的高置信度位置数；`sigma` 为置信度密度高斯核的标准差；`inflate_w` 为窗口膨胀大小。 |
| L26 | `super().__init__(model_config)` | 调用父类 `dCache` 初始化基础属性。 |
| L27-L28 | `self.key_cache/value_cache = []` | KV 缓存列表，每个元素形状 `(B, T, head_dim)`，按层索引。 |
| L29 | `self._conf_cache: torch.Tensor \| None = None` | 置信度分数缓存，形状 `(B, G)`，G 为生成 token 数量。累积记录各位置在各步的置信度。 |
| L30 | `self._full_q_mask: torch.Tensor \| None = None` | 完整的 Q 位置掩码，形状 `(B, T)`，由 `on_step_end` 产生并被 `model_forward` 消费。 |
| L31 | `self._density_score: torch.Tensor` | 置信度密度分数，形状 `(B, G)`，在 `on_step_end` 中计算，供 `top_up_mask` 使用。 |
| L32 | `self._global_importance: torch.Tensor` | 全局注意力重要性，形状 `(B, T)`，由 `_attn_rollout.sum(dim=1)` 产生，供 `top_up_mask` 使用。 |
| L34-L36 | `self.rollout_p/current_k/sigma/inflate_w = ...` | 保存用户配置的超参数。 |

---

### `model_forward` — 模型前向传播上下文管理器

```python
# 源文件: src/cache/d2cache.py L38-L55
@contextmanager
def model_forward(self, x: torch.Tensor):
    with super().model_forward(x=x) as ctx:
        B, T, C = x.shape
        if self._full_q_mask is not None:
            self.active_q_mask = self.top_up_mask(
                self._full_q_mask[self.active_seq_mask]
            )
            ctx.x = x[self.active_q_mask].view(B, -1, C)
        yield ctx

        if self._full_q_mask is not None:
            assert ctx.logits is not None and self.active_q_mask is not None
            ctx.logits = torch.zeros(
                (B, T, ctx.logits.size(-1)),
                dtype=ctx.logits.dtype,
                device=ctx.logits.device,
            ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L38 | `@contextmanager` | 上下文管理器装饰器。 |
| L39 | `def model_forward(self, x):` | 输入 `x` 形状 `(B, T, C)`。 |
| L40 | `with super().model_forward(x=x) as ctx:` | 进入父类前向传播上下文。 |
| L41 | `B, T, C = x.shape` | 提取维度。 |
| L42-L44 | `if self._full_q_mask is not None:` | 若 `_full_q_mask` 已初始化，调用 `top_up_mask` 对活跃序列的掩码做批次均衡填充，得到 `active_q_mask`。这是 d2Cache 与 PrefixCache 的关键区别：d2Cache 在 `model_forward` 中动态调整掩码以确保批次内位置数一致。 |
| L44-L46 | `self.active_q_mask = self.top_up_mask(self._full_q_mask[self.active_seq_mask])` | 数据流转：`_full_q_mask` (B_total, T) → 取活跃行 (B_active, T) → `top_up_mask` 填充 → `active_q_mask` (B_active, T)。 |
| L47 | `ctx.x = x[self.active_q_mask].view(B, -1, C)` | 裁剪输入，形状 `(B, T, C) → (B, T', C)`。 |
| L48 | `yield ctx` | 交出控制权。 |
| L50-L55 | `ctx.logits = torch.zeros(...).masked_scatter_(...)` | 将 logits 散布回完整形状 `(B, T, vocab_size)`，与 PrefixCache 相同。 |

---

### `attention` — 注意力层上下文管理器

```python
# 源文件: src/cache/d2cache.py L57-L128
@contextmanager
def attention(
    self,
    layer_idx: int,
    x: torch.Tensor,
    attn_norm: nn.Module,
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
):
    with super().attention(
        layer_idx,
        x,
        attn_norm,
        q_proj,
        k_proj,
        v_proj,
        attention_mask,
        position_ids,
    ) as ctx:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(ctx.k)
            self.value_cache.append(ctx.v)
        else:
            assert self.active_q_mask is not None
            if layer_idx == 0:
                active_seq_idx = torch.where(self.active_seq_mask)[0]
                m_nonzero = self.active_q_mask.nonzero(as_tuple=False)
                self._active_q_indices = (
                    active_seq_idx[m_nonzero[:, 0]],
                    m_nonzero[:, 1],
                )

            self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)
            self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(
                0, 1
            )
            ctx.k = self.key_cache[layer_idx][self.active_seq_mask]
            ctx.v = self.value_cache[layer_idx][self.active_seq_mask]

        if layer_idx == 0:
            self._q_position_ids, self._kv_position_ids = (
                AttentionContext.select_position_ids(
                    position_ids, self.active_q_mask
                )
            )
            self._attention_mask = AttentionContext.convert_attention_mask(
                attention_mask,
                dtype=ctx.k.dtype,
                query_length=ctx.q.shape[1],
                key_value_length=self.value_cache[layer_idx].shape[1],
            )

        ctx.q_position_ids = self._q_position_ids
        ctx.kv_position_ids = self._kv_position_ids
        ctx.attention_mask = self._attention_mask
        yield ctx

        assert (
            ctx.attn_weight is not None
        ), 'The attention weights must be outputed, make sure you\'ve set attn_implementation="eager"'

        if layer_idx == 0:
            self._attn_rollout = torch.eye(
                self.key_cache[layer_idx].size(1), device=x.device, dtype=x.dtype
            ).expand(x.size(0), -1, -1)
        self.accumulate_attn_rollout(ctx.attn_weight)
```

**逐行解释——第一部分：KV 缓存管理 (L57-L98)：**

KV 缓存逻辑与 PrefixCache 高度相似，关键区别在 L79-L84 的 `else` 分支对索引的精确处理。

| 行号 | 代码 | 说明 |
|------|------|------|
| L57 | `@contextmanager` | 上下文管理器装饰器。 |
| L58-L68 | `def attention(self, layer_idx, x, ...):` | 参数与 PrefixCache 完全相同。 |
| L69-L78 | `with super().attention(...) as ctx:` | 调用父类完成 LayerNorm 和 Q/K/V 投影。父类返回的 `ctx.k`/`ctx.v` 为 `(B_active, T', H)`（仅 active_q_mask 位置参与计算）。 |
| L79-L82 | `if len(self.key_cache) <= layer_idx:` | 首次前向传播，追加完整缓存。 |
| L83-L96 | `else:` | 非首次：增量更新。与 PrefixCache 相同，在 `layer_idx == 0` 时计算全局索引 `_active_q_indices`，然后按索引更新和返回缓存。d2Cache 仅有首次和常规两种缓存状态，无 `_new_block_start` 分支。 |
| L87-L91 | `if layer_idx == 0: ...` | 计算 `_active_q_indices`（行映射 + 列索引），与 PrefixCache 相同。 |
| L93-L96 | `self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)` | `ctx.k` 从 `(B_active, T', H)` flatten 为 `(B_active*T', H)` 后赋值给缓存的指定位置。 |
| L97-L98 | `ctx.k = self.key_cache[layer_idx][self.active_seq_mask]` | 从缓存取出完整 KV 赋值回 `ctx`，恢复形状 `(B_active, T, H)`。 |

**逐行解释——第二部分：层间共享变量与 yield 后逻辑 (L100-L128)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L100-L101 | `if layer_idx == 0:` | 仅在第 0 层设置层间共享变量。 |
| L102-L106 | `AttentionContext.select_position_ids(position_ids, self.active_q_mask)` | 筛选 Q 的 position_ids，形状从 `(B, T)` 缩为 `(B, T')`。 |
| L107-L112 | `AttentionContext.convert_attention_mask(...)` | 转换 attention_mask 为加性掩码，KV 长度取完整缓存的 dim=1。 |
| L114-L116 | `ctx.q/kv_position_ids/attention_mask = ...` | 赋值到 ctx。 |
| L117 | `yield ctx` | 交出控制权，模型在此完成注意力计算并填充 `ctx.attn_weight` 和 `ctx.o`。 |
| L119-L121 | `assert ctx.attn_weight is not None` | d2Cache 的核心依赖：必须输出注意力权重。要求模型使用 `attn_implementation="eager"`。 |
| L123-L127 | `if layer_idx == 0: self._attn_rollout = torch.eye(T).expand(B, -1, -1)` | 在第 0 层初始化 `_attn_rollout` 为 `(B, T, T)` 的单位矩阵，作为 Attention Rollout 的初始值。 |
| L128 | `self.accumulate_attn_rollout(ctx.attn_weight)` | 逐层累积注意力流，`ctx.attn_weight` 形状 `(B, n_heads, T', T)`。 |

---

### `top_up_mask` — 批次均衡填充

```python
# 源文件: src/cache/d2cache.py L130-L145
def top_up_mask(self, q_mask: torch.Tensor):
    q_mask = q_mask.clone()
    num_selected_per_seq = q_mask.sum(dim=-1)
    _, G = self._density_score.shape
    if torch.any(num_selected_per_seq != num_selected_per_seq.max()):
        combined_scores = torch.where(
            q_mask, -torch.inf, self._global_importance[self.active_seq_mask]
        )
        combined_scores[:, -G:] += (
            combined_scores.max() + self._density_score[self.active_seq_mask]
        )

        top_up_mask_(q_mask, int(num_selected_per_seq.max()), combined_scores)
    return q_mask
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L130 | `def top_up_mask(self, q_mask):` | 输入 `q_mask` 形状 `(B_active, T)`，为活跃序列的 Q 掩码。 |
| L131 | `q_mask = q_mask.clone()` | 克隆避免原地修改上游张量。 |
| L132 | `num_selected_per_seq = q_mask.sum(dim=-1)` | 统计每条序列已选中的位置数，形状 `(B_active,)`。 |
| L133 | `_, G = self._density_score.shape` | 获取生成 token 数量 G。 |
| L134 | `if torch.any(num_selected_per_seq != num_selected_per_seq.max()):` | 仅当存在序列选中数少于最大值时才执行填充。这是批次均衡的核心条件。 |
| L135-L137 | `combined_scores = torch.where(q_mask, -torch.inf, self._global_importance[...])` | 构造复合分数：已选中位置设为 `-inf`（不可再选），未选中位置使用全局注意力重要性。形状 `(B_active, T)`。 |
| L138-L141 | `combined_scores[:, -G:] += combined_scores.max() + self._density_score[...]` | 对 Response 区域（最后 G 个位置）的分数叠加置信度密度值和一个全局偏置（`combined_scores.max()`），确保 Response 区域的掩码位置优先被选中。 |
| L144 | `top_up_mask_(q_mask, int(num_selected_per_seq.max()), combined_scores)` | 调用工具函数 `top_up_mask_`，将 `q_mask` 填充至每序列 `max_count` 个 `True`，按 `combined_scores` 从高到低选择新增位置。 |
| L145 | `return q_mask` | 返回填充后的掩码，所有序列选中位置数一致。 |

---

### `accumulate_attn_rollout` — Attention Rollout 累积

```python
# 源文件: src/cache/d2cache.py L147-L174
def accumulate_attn_rollout(self, attn_scores: torch.Tensor):
    """
    Computes one step of the Attention Rollout for attention maps.
    In this setup, only a subset of tokens act as queries.

    Args:
        attn_scores (torch.Tensor):
            Attention scores for the current layer, with shape of (B, num_heads, q_len, seq_len).
    """
    B, n_heads, q_len, seq_len = attn_scores.shape
    device, dtype = attn_scores.device, attn_scores.dtype

    if self.active_q_mask is None:
        effective_attn = attn_scores.mean(dim=1)
    else:
        effective_attn = torch.eye(seq_len, device=device, dtype=dtype).repeat(
            B, 1, 1
        )
        effective_attn[self.active_q_mask] = attn_scores.mean(dim=1).reshape(
            -1, seq_len
        )

    residual_attn = effective_attn + torch.eye(seq_len, device=device, dtype=dtype)
    residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)

    self._attn_rollout = residual_attn @ self._attn_rollout
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L148-L155 | 文档字符串 | 说明 Attention Rollout 的核心思想：累积各层注意力权重以追踪信息流。 |
| L156 | `B, n_heads, q_len, seq_len = attn_scores.shape` | 注意力分数形状 `(B, n_heads, q_len, seq_len)`。`q_len` = 选中 Q 位置数（T'），`seq_len` = 完整序列长度（T）。这是矩形注意力矩阵（非方阵）。 |
| L157 | `device, dtype = attn_scores.device, attn_scores.dtype` | 保存设备和类型。 |
| L159-L161 | `if self.active_q_mask is None: effective_attn = attn_scores.mean(dim=1)` | 首次前向传播（所有位置都是 Q）：沿 head 维度平均，得到 `(B, T, T)` 方阵。 |
| L162-L165 | `else: effective_attn = torch.eye(seq_len).repeat(B, 1, 1)` | 非首次：先创建 `(B, T, T)` 单位矩阵作为骨架。 |
| L166-L168 | `effective_attn[self.active_q_mask] = attn_scores.mean(dim=1).reshape(-1, seq_len)` | 将矩形注意力 `(B, T', T)` 的每一行注入到 `effective_attn` 矩阵中 `active_q_mask=True` 的对应行。这样未选中位置的注意力保留为单位向量（自关注）。关键数据流转：`(B, n_heads, T', T) → mean → (B, T', T) → 注入 (B, T, T) 骨架`。 |
| L170 | `residual_attn = effective_attn + torch.eye(seq_len, ...)` | 添加残差连接（加单位矩阵），对应公式中的 `A^(l) + I`。 |
| L172 | `residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)` | 按行归一化，使每行概率和为 1。 |
| L174 | `self._attn_rollout = residual_attn @ self._attn_rollout` | 矩阵乘法累积：`A_rollout^(l) = (A^(l) + I) @ A_rollout^(l-1)`。形状始终为 `(B, T, T)`。 |

---

### `on_step_end` — 步骤结束回调

这是 d2Cache 最核心的方法，实现基于置信度密度、Attention Rollout 和窗口膨胀的智能位置选择。按功能分为八个子步骤逐步分析。

```python
# 源文件: src/cache/d2cache.py L176-L298
def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
    confidence = delta.confidence
    assert confidence is not None
    B, P = frame.prompts.shape
    B_active, G = confidence.shape
    T = G + P
    block_mask = block_mask[self.active_seq_mask]
    new_frame = frame.apply_delta(delta)
    device = confidence.device

    if self._conf_cache is None:
        self._conf_cache = confidence

    remaining_mask = (
        new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
    )
    if self.active_q_mask is not None:
        valid_mask = (
            self.active_q_mask[:, P:] & frame.generated_tokens[self.active_seq_mask]
            == self.mask_token_id
        )
        self._conf_cache[self.active_seq_mask][valid_mask] = confidence[valid_mask]

    block_size = block_mask.sum(dim=1, keepdim=True)

    meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k
    min_search_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)
    min_search_end[~meets_target.any(dim=1, keepdim=True)] = G - 1

    search_end = (((min_search_end // block_size) + 1) * block_size) - 1

    block_start_indices = torch.argmax(block_mask.int(), dim=1, keepdim=True)
    col_indices = torch.arange(G, device=device)
    search_mask = (col_indices >= block_start_indices) & (col_indices <= search_end)

    scores = self._conf_cache[self.active_seq_mask] * certainty_density(
        ~remaining_mask, self.sigma
    )

    scores[block_mask] += scores.max()
    _, indices = torch.topk(
        torch.where(search_mask & remaining_mask, scores, -torch.inf),
        k=min(self.current_k, remaining_mask.size(-1)),
        dim=-1,
    )
    selected_mask = (
        torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(
            1, indices, True
        )
        & remaining_mask
    )

    if is_adapted_from_ar(self.model_config):
        response_mask = F.pad(selected_mask[:, 1:], (0, 1), value=False)
    else:
        response_mask = selected_mask

    transfer_src_index = (
        delta.transfer_src_index
        if delta.transfer_src_index is not None
        else delta.transfer_index
    )
    lengths = torch.tensor(
        [ti.numel() for ti in transfer_src_index if ti.numel() > 0], device=device
    )
    row_indices = torch.repeat_interleave(
        torch.arange(B_active, device=confidence.device), lengths
    )
    col_indices = torch.cat(transfer_src_index)
    response_mask[row_indices, col_indices] = True

    q_mask = F.pad(response_mask, (P, 0), value=False)

    global_importance = self._attn_rollout.sum(dim=1)
    q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)

    if is_adapted_from_ar(self.model_config):
        q_mask[:, P - 1] = selected_mask[:, 0]

    if self.inflate_w > 0:
        arange_t = torch.arange(T, device=device).expand(B_active, -1)

        masked_indices_next = torch.where(q_mask, arange_t, T)
        next_selected_indices = torch.cummin(
            torch.flip(masked_indices_next, dims=[-1]), dim=-1
        ).values
        next_selected_indices = torch.flip(next_selected_indices, dims=[-1])
        dist_to_next_true = next_selected_indices - arange_t

        masked_indices_prev = torch.where(q_mask, arange_t, -1)
        prev_selected_indices = torch.cummax(masked_indices_prev, dim=-1).values
        dist_to_prev_true = arange_t - prev_selected_indices

        gap_len = dist_to_next_true + dist_to_prev_true
        q_mask |= (
            (gap_len <= self.inflate_w)
            & (prev_selected_indices >= 0)
            & (next_selected_indices < T)
        )

    if self._full_q_mask is None:
        self._full_q_mask = q_mask
        self._global_importance = global_importance
        self._density_score = scores
    else:
        self._full_q_mask[self.active_seq_mask] = q_mask
        self._global_importance[self.active_seq_mask] = global_importance
        self._density_score[self.active_seq_mask] = scores
```

**逐行解释——步骤 1：参数解析与置信度更新 (L176-L202)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L176 | `def on_step_end(self, block_mask, frame, delta):` | 核心回调，`delta` 包含本步的 `confidence` 和 `transfer_src_index`。 |
| L177 | `confidence = delta.confidence` | 获取置信度张量，形状 `(B_active, G)`。 |
| L179 | `B, P = frame.prompts.shape` | B=总批次大小，P=prompt 长度。 |
| L180 | `B_active, G = confidence.shape` | B_active=活跃序列数，G=生成 token 数量。 |
| L181 | `T = G + P` | 总序列长度 = 生成 token 数 + prompt 长度。 |
| L182 | `block_mask = block_mask[self.active_seq_mask]` | 筛选活跃序列的 block_mask，形状 `(B_active, G)`。 |
| L183 | `new_frame = frame.apply_delta(delta)` | 模拟应用 delta 后的新帧状态，用于获取更新后的 `generated_tokens`。 |
| L184 | `device = confidence.device` | 保存设备引用。 |
| L186-L187 | `if self._conf_cache is None: self._conf_cache = confidence` | 首次调用时，用当前置信度初始化 `_conf_cache`，形状 `(B_active, G)`。 |
| L190-L192 | `remaining_mask = new_frame.generated_tokens[...] == self.mask_token_id` | 构造剩余掩码位置标记，形状 `(B_active, G)`。`True` 表示该位置仍是 `[MASK]` token。 |
| L193-L202 | `if self.active_q_mask is not None:` | 若非首次步骤，仅更新上一步选中且仍为掩码位置的置信度（其他位置的置信度可能已过时或无效）。 |
| L195-L197 | `valid_mask = self.active_q_mask[:, P:] & frame.generated_tokens[...] == mask_token_id` | valid_mask 形状 `(B_active, G)`，标记上一步选中且当时为掩码的位置。 |
| L198-L202 | `self._conf_cache[self.active_seq_mask][valid_mask] = confidence[valid_mask]` | 仅更新 valid_mask 位置的置信度值到缓存中。 |

**逐行解释——步骤 2：确定搜索范围 (L204-L219)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L204 | `block_size = block_mask.sum(dim=1, keepdim=True)` | 计算每序列的块大小，形状 `(B_active, 1)`。 |
| L207 | `meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k` | 从左到右累计剩余掩码数，标记何时达到 `current_k` 个候选。形状 `(B_active, G)`。 |
| L208 | `min_search_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)` | 找到每序列最早满足 `≥ current_k` 个掩码候选的列索引，形状 `(B_active, 1)`。 |
| L209 | `min_search_end[~meets_target.any(dim=1, keepdim=True)] = G - 1` | 对掩码候选不足的序列，搜索终点设为 G-1。 |
| L212 | `search_end = (((min_search_end // block_size) + 1) * block_size) - 1` | 将最小搜索终点向上取整到下一个 block 边界。 |
| L214 | `block_start_indices = torch.argmax(block_mask.int(), dim=1, keepdim=True)` | 获取当前 block 的起始位置，形状 `(B_active, 1)`。 |
| L215-L216 | `search_mask = (col_indices >= block_start_indices) & (col_indices <= search_end)` | 构造搜索窗口掩码，形状 `(B_active, G)`。 |

**逐行解释——步骤 3：置信度密度评分与 top-k 选择 (L218-L234)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L218-L220 | `scores = self._conf_cache[self.active_seq_mask] * certainty_density(~remaining_mask, self.sigma)` | 计算置信度密度分数。`certainty_density` 对 `~remaining_mask`（已生成区域，`True`）应用高斯核，高密度区域获得高分。与置信度相乘得到综合分数，形状 `(B_active, G)`。 |
| L223 | `scores[block_mask] += scores.max()` | 对 block 内位置添加全局偏置，确保每个 block 至少有一个位置被选中。 |
| L224-L228 | `indices = torch.topk(..., k=min(current_k, G))` | 在 `search_mask & remaining_mask`（搜索窗口内且仍为掩码）范围内，按 `scores` 选择 top-k 最高分的位置。返回索引形状 `(B_active, k)`。 |
| L229-L234 | `selected_mask = zeros.scatter_(1, indices, True) & remaining_mask` | 将 top-k 索引转为布尔掩码，并与 `remaining_mask` 取交集确保只选中掩码位置。形状 `(B_active, G)`。 |

**逐行解释——步骤 4-5：AR 适配与 transfer token 处理 (L236-L253)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L236-L239 | `if is_adapted_from_ar(...): response_mask = F.pad(selected_mask[:, 1:], ...)` | Dream 模型（AR 适配）需要保留选中位置的前驱 token：将 `selected_mask` 右移一位。 |
| L241-L253 | `transfer_src_index = delta.transfer_src_index or delta.transfer_index` | 获取本步从 `[MASK]` 转为具体 token 的位置索引（不等长列表）。 |
| L246-L253 | `row_indices/col_indices/response_mask[...] = True` | 将 transfer 位置也标记为 `True`，确保这些新生成的 token 在下一步参与计算。 |

**逐行解释——步骤 6-8：Rollout 选择、窗口膨胀与状态更新 (L255-L298)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L255 | `q_mask = F.pad(response_mask, (P, 0), value=False)` | 左侧 padding Prompt 区域，形状从 `(B_active, G)` 扩展为 `(B_active, T)`。 |
| L258 | `global_importance = self._attn_rollout.sum(dim=1)` | 沿 dim=1 求和，将 `(B_active, T, T)` 的 rollout 矩阵压缩为 `(B_active, T)`，每个位置的值反映其全局注意力流入量。 |
| L259 | `q_mask \|= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)` | 基于 Attention Rollout 的全局重要性，用 nucleus 选择法选取未选中但注意力重要的位置，合并到 `q_mask`。 |
| L261-L262 | `if is_adapted_from_ar(...): q_mask[:, P - 1] = selected_mask[:, 0]` | AR 适配模型需要额外选中 prompt 最后一个 token（位置 P-1），作为第一个生成 token 的前驱。 |
| L264-L267 | `if self.inflate_w > 0:` | 窗口膨胀：将选中的孤立位置周围小间隙（≤ inflate_w）也填充为选中。 |
| L269 | `arange_t = torch.arange(T, ...).expand(B_active, -1)` | 创建位置索引张量，形状 `(B_active, T)`。 |
| L271-L276 | `masked_indices_next = ... → dist_to_next_true` | 计算每个位置到其右侧最近选中位置的距离：先用 `torch.where` 将未选中位置替换为 T，再用 `cummin` + `flip` 取右侧最近选中索引。 |
| L278-L280 | `masked_indices_prev = ... → dist_to_prev_true` | 计算每个位置到其左侧最近选中位置的距离：用 `torch.where` 将未选中位置替换为 -1，再用 `cummax` 取左侧最近选中索引。 |
| L283-L291 | `gap_len = dist_to_next_true + dist_to_prev_true; q_mask \|= (gap_len <= inflate_w) & ...` | 若左右选中位置之间的间隙 ≤ `inflate_w`，则将中间位置也标记为选中。 |
| L295-L298 | `if self._full_q_mask is None: ... else: ...` | 更新全批次 `_full_q_mask`、`_global_importance` 和 `_density_score`。首次直接赋值，后续按 `active_seq_mask` 索引增量更新。 |

## 关键函数和上下文管理器说明

### 关键辅助函数

#### `top_up_mask_`

填充布尔掩码以确保每行有目标数量的 True 值：

```python
def top_up_mask_(
    mask: torch.Tensor, target_count: int, scores: torch.Tensor
) -> torch.Tensor:
    """
    填充布尔掩码以确保每行有目标数量的 True 值。
    基于 scores 选择哪些 False 位置翻转为 True。
    """
    B, _ = mask.shape
    device = mask.device

    num_selected_per_seq = mask.sum(dim=-1)
    num_to_pad_per_seq = (target_count - num_selected_per_seq).clamp(min=0)

    if num_to_pad_per_seq.sum() == 0:
        return mask

    max_num_to_pad = int(num_to_pad_per_seq.max())
    scores = torch.where(mask, -torch.inf, scores)

    _, indices = torch.topk(scores, k=max_num_to_pad, dim=-1)

    pad_indices = indices.masked_select(
        torch.arange(max_num_to_pad, device=device).expand(B, -1)
        < num_to_pad_per_seq.unsqueeze(-1)
    )

    row_indices = torch.repeat_interleave(
        torch.arange(B, device=device), num_to_pad_per_seq.long()
    )
    mask[row_indices, pad_indices] = True

    return mask
```

## 使用示例和参数配置

### 基本使用

```python
from src.cache import d2Cache
from transformers import AutoConfig

# 加载模型配置
model_config = AutoConfig.from_pretrained("path/to/model")

# 创建 d2Cache 实例
cache = d2Cache(
    model_config,
    rollout_p=0.1,     # 10% 的位置基于 Attention Rollout 选择
    current_k=32,      # 每步选择 32 个高置信度位置
    sigma=10.0,        # 置信度密度高斯核标准差
    inflate_w=4,       # 窗口膨胀大小
)

# 在生成过程中使用
# 注意：需要设置 attn_implementation="eager" 以输出注意力权重
model = AutoModel.from_pretrained(
    "path/to/model",
    attn_implementation="eager"
)

for step in range(num_steps):
    with cache.model_forward(hidden_states) as ctx:
        # 模型处理...
        pass
```

### 配置文件示例

```yaml
# configs/cache/d2cache.yaml
_target_: src.cache.d2Cache
rollout_p: 0.1
current_k: 32
sigma: 10.0
inflate_w: 4
```

### 不同场景的参数配置

```yaml
# 高精度配置（适合需要高质量生成的场景）
rollout_p: 0.2
current_k: 64
sigma: 15.0
inflate_w: 8

# 高效率配置（适合需要快速生成的场景）
rollout_p: 0.05
current_k: 16
sigma: 5.0
inflate_w: 2

# 平衡配置（默认推荐）
rollout_p: 0.1
current_k: 32
sigma: 10.0
inflate_w: 4
```

### 与模型集成

```python
from transformers import AutoModelForCausalLM

class MyDiffusionModel(nn.Module):
    def __init__(self, model_name):
        # 必须使用 eager 注意力实现以输出注意力权重
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
    
    def forward(self, input_ids, cache=None):
        hidden_states = self.model.embed_tokens(input_ids)
        
        if cache is not None:
            with cache.model_forward(hidden_states) as ctx:
                for layer_idx, layer in enumerate(self.model.layers):
                    with cache.attention(
                        layer_idx, ctx.x,
                        layer.input_layernorm,
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                        attention_mask, position_ids
                    ) as attn_ctx:
                        # 注意力计算必须返回注意力权重
                        attn_output, attn_weights = layer.self_attn(
                            attn_ctx.q, attn_ctx.k, attn_ctx.v,
                            attn_ctx.attention_mask,
                            output_attentions=True
                        )
                        attn_ctx.attn_weight = attn_weights
                        attn_ctx.o = layer.self_attn.o_proj(attn_output)
                        ctx.x = attn_ctx.o + attn_ctx.residual
                    
                    with cache.ffn(layer_idx, ctx.x) as ffn_ctx:
                        ffn_ctx.ffn_out = layer.mlp(ffn_ctx.x)
                        ctx.x = ffn_ctx.ffn_out + ffn_ctx.residual
                
                ctx.logits = self.model.lm_head(ctx.x)
        
        return ctx.logits if cache else self.standard_forward(hidden_states)
```

## 性能特点和适用场景

### 性能特点

| 特点 | 说明 |
|------|------|
| **智能位置选择** | 结合注意力流和置信度密度，动态选择最优计算位置 |
| **全局注意力追踪** | Attention Rollout 追踪全局信息流，识别语义重要位置 |
| **置信度密度评估** | 高斯核密度评估局部确定性，指导选择策略 |
| **窗口膨胀** | 自动填充选中位置间的间隙，保证生成连贯性 |
| **批次均衡** | 确保批次中所有序列选择相同数量的位置 |

### 计算效率分析

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    计算效率对比 (假设序列长度 512, 10步生成)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  无缓存方法:                                                                     │
│  ████████████████████████████████████████████████████████████████████ 100%      │
│  (每步计算所有位置)                                                              │
│                                                                                 │
│  PrefixCache:                                                                   │
│  ████████████████████████████████████████████████████████████████████ 100%      │
│  (首次)                                                                          │
│  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8%       │
│  (后续, 假设每步更新 10% 位置)                                                   │
│                                                                                 │
│  dLLM-Cache (kp=50, kr=2, rou=0.25):                                           │
│  ████████████████████████████████████████████████████████████████████ 100%      │
│  (首次)                                                                          │
│  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%       │
│  (常规步骤)                                                                      │
│                                                                                 │
│  d2Cache (rollout_p=0.1, current_k=32, inflate_w=4):                           │
│  ████████████████████████████████████████████████████████████████████ 100%      │
│  (首次)                                                                          │
│  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3%       │
│  (后续, 智能选择 + 窗口膨胀)                                                     │
│                                                                                 │
│  图例: █ 需要计算   ░ 复用缓存                                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 适用场景

1. **高质量生成**
   - 通过注意力流追踪确保语义重要位置得到及时刷新
   - 置信度密度评估保证生成连贯性

2. **长序列生成**
   - 智能位置选择减少不必要的计算
   - 窗口膨胀保证长距离依赖的建模

3. **批量推理**
   - 批次均衡确保 GPU 利用率最大化
   - 自动处理不同序列的进度差异

### 与其他缓存方法的对比

| 特性 | PrefixCache | dLLM-Cache | d2Cache |
|------|-------------|------------|---------|
| KV 缓存 | ✓ | ✓ | ✓ |
| 选择性更新 | ✓ | ✓ | ✓ |
| 周期性刷新 | ✗ | ✓ | ✗ |
| 自适应刷新 | ✗ | ✓ | ✗ |
| Attention Rollout | ✗ | ✗ | ✓ |
| 置信度密度 | ✗ | ✗ | ✓ |
| 窗口膨胀 | ✗ | ✗ | ✓ |
| 批次均衡 | 部分 | ✓ | ✓ |
| 注意力权重输出 | 不需要 | 不需要 | 必需 |
| 实现复杂度 | 低 | 中 | 高 |
| 计算效率提升 | 中 | 高 | 最高 |
| 生成质量保证 | 中 | 高 | 最高 |

### 局限性

1. **注意力权重依赖**：需要模型输出注意力权重，可能影响某些优化（如 Flash Attention）
2. **额外计算开销**：Attention Rollout 和置信度密度计算有额外开销
3. **参数敏感性**：四个参数需要根据场景调优，参数选择对性能影响较大
4. **内存开销**：需要存储 `_attn_rollout` 矩阵 (B, T, T)

## 总结

d2Cache 是 d2Cache 项目中最先进的缓存实现，它通过 Attention Rollout 和置信度密度分析，实现了智能的位置选择策略。相比 PrefixCache 和 dLLM-Cache，d2Cache 能够更准确地识别语义重要的位置，在保证生成质量的同时实现最高的计算效率。

d2Cache 特别适合以下场景：
- 需要高质量生成的任务
- 长序列生成任务
- 对计算效率有较高要求的批量推理场景

通过合理配置 `rollout_p`、`current_k`、`sigma` 和 `inflate_w` 参数，可以在计算效率、生成质量和生成连贯性之间找到最佳平衡点。
