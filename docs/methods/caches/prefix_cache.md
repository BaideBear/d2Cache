# PrefixCache 缓存方法详解

## 算法逻辑精要

在扩散语言模型生成过程中，PrefixCache 只计算 `active_q_mask` 标记位置的 Q/K/V 投影，其余位置直接复用已缓存的 KV 状态，从而大幅减少投影计算量。首次前向传播时缓存全部位置的 Key 和 Value，后续步骤则根据掩码有选择地更新指定位置的缓存，并将模型输出的 logits 通过 `masked_scatter_` 散布回原始序列形状。其核心调度逻辑是：在 `model_forward` contextmanager 中裁剪输入张量，在 `attention` 中管理 KV 缓存的增/改/查，在 `on_step_end` 中初始化下一步的 `active_q_mask`，在 `on_block_start` 中重置缓存状态。

## 概述

PrefixCache 是 d2Cache 项目中最基础的缓存实现，它为扩散语言模型（Diffusion Language Models）提供了一种简单而高效的 KV-Cache 复用机制。该缓存方法的核心思想是：在生成过程中，只计算需要更新的位置的 Query、Key 和 Value，而复用其他位置的缓存状态，从而显著减少计算开销。

## 算法原理和理论基础

### 核心思想

在扩散语言模型的解码过程中，并非所有位置都需要在每个步骤中重新计算。PrefixCache 的设计基于以下观察：

1. **选择性计算**：在每个生成步骤中，只有部分位置需要重新计算其 Query、Key 和 Value
2. **缓存复用**：已经计算过的状态可以被缓存并在后续步骤中复用
3. **批量处理**：支持批量序列处理，提高 GPU 利用率

### 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    PrefixCache 工作流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 首次前向传播  │───▶│  存储 KV 缓存 │───▶│ 后续前向传播  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌──────────────┐                        ┌──────────────┐      │
│  │ 计算所有位置  │                        │ 选择性计算    │      │
│  │ 的 Q、K、V   │                        │ active_q_mask │      │
│  └──────────────┘                        │ 指定的位置    │      │
│                                          └──────────────┘      │
│                                                 │              │
│                                                 ▼              │
│                                          ┌──────────────┐      │
│                                          │ 更新缓存位置  │      │
│                                          │ 返回完整 KV   │      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心数据结构和参数说明

### 类定义

```python
class PrefixCache(dCache):
    def __init__(self, model_config, use_dual: bool = False):
        super().__init__(model_config)
        self.use_dual = use_dual
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.active_q_mask: torch.Tensor | None = None
        self._new_block_start = False
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_config` | dict | 必需 | 模型配置对象，包含模型架构信息 |
| `use_dual` | bool | False | 是否使用双模式（用于特殊生成策略） |

### 核心数据结构

| 属性 | 类型 | 说明 |
|------|------|------|
| `key_cache` | list[torch.Tensor] | 每层 Key 状态的缓存列表，形状为 `(batch_size, seq_len, head_dim)` |
| `value_cache` | list[torch.Tensor] | 每层 Value 状态的缓存列表，形状为 `(batch_size, seq_len, head_dim)` |
| `active_q_mask` | torch.Tensor \| None | 布尔掩码，指示当前步骤需要计算的位置 |
| `_new_block_start` | bool | 内部标志，标记是否为新块的开始 |

## 详细代码流程分析

以下按源码文件 [`src/cache/prefix_cache.py`](file:///Users/lier/codes/d2Cache/src/cache/prefix_cache.py) 的模块顺序，逐方法展开分析。

### `__init__` — 初始化

```python
# 源文件: src/cache/prefix_cache.py L14-L20
def __init__(self, model_config, use_dual: bool = False):
    super().__init__(model_config)
    self.use_dual = use_dual
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    self.active_q_mask: torch.Tensor | None = None
    self._new_block_start = False
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L14 | `def __init__(self, model_config, use_dual: bool = False):` | 构造函数。`model_config` 传递给父类；`use_dual` 控制是否启用双模式（用于半自回归等特殊生成策略）。 |
| L15 | `super().__init__(model_config)` | 调用父类 `dCache` 的初始化，完成 `mask_token_id`、`active_seq_mask` 等基础属性的设置。 |
| L16 | `self.use_dual = use_dual` | 保存双模式标志，影响 `on_step_end` 中 `q_mask` 的构建方式。 |
| L18-L19 | `self.key_cache/value_cache: list[torch.Tensor] = []` | 初始化 KV 缓存列表。这是一个按层索引的列表，每个元素为形状 `(B, T, head_dim)` 的张量。 |
| L20 | `self.active_q_mask: torch.Tensor \| None = None` | 布尔掩码，形状 `(B, T)`，标记当前步骤需要计算 Q/K/V 投影的位置。初始为 `None`，表示首次前向传播时计算全部位置。 |
| L21 | `self._new_block_start = False` | 内部标志位，标记是否进入新块（batch 内新增序列等场景），用于控制缓存整体替换逻辑。 |

---

### `model_forward` — 模型前向传播上下文管理器

```python
# 源文件: src/cache/prefix_cache.py L22-L41
@contextmanager
def model_forward(self, x: torch.Tensor):
    with super().model_forward(x=x) as ctx:
        B, T, C = x.shape
        if self.active_q_mask is not None:
            if B != self.active_q_mask.size(0):
                self.active_q_mask = self.active_q_mask[0].expand(B, -1)
            ctx.x = x[self.active_q_mask].view(B, -1, C)

        yield ctx

        if self.active_q_mask is not None:
            assert ctx.logits is not None
            ctx.logits = torch.zeros(
                (B, T, ctx.logits.size(-1)),
                dtype=ctx.logits.dtype,
                device=ctx.logits.device,
            ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L22 | `@contextmanager` | 将方法标记为上下文管理器（generator），支持 `with` 语句。 |
| L23 | `def model_forward(self, x: torch.Tensor):` | 输入 `x` 为模型 embedding 输出，形状 `(B, T, C)`（B=batch, T=序列长度, C=隐藏维度）。 |
| L24 | `with super().model_forward(x=x) as ctx:` | 进入父类 `dCache` 的 `model_forward` 上下文，父类负责基础的前向传播框架和 `logits` 容器创建。 |
| L25 | `B, T, C = x.shape` | 提取 batch 大小 B、总序列长度 T、隐藏维度 C。 |
| L26 | `if self.active_q_mask is not None:` | 如果 `active_q_mask` 已初始化（非首次前向），则只保留掩码指定的位置以节省计算。首次时 `active_q_mask` 为 `None`，直接使用完整 `x`。 |
| L27-L28 | `if B != self.active_q_mask.size(0): ...` | 批次大小可能变化（某些序列已结束）。若 B 变化，取第一条序列的掩码并扩展到当前 batch 大小。 |
| L29-L31 | `ctx.x = x[self.active_q_mask].view(B, -1, C)` | 使用布尔索引从 `x` (B, T, C) 中选出 `active_q_mask=True` 的位置，再 `view` 回 `(B, -1, C)`。关键数据流转：`(B, T, C) → (B, T', C)`，其中 `T'` 为每序列选中位置数。 |
| L33 | `yield ctx` | 交出控制权给模型执行 Transformer 层的计算。 |
| L35 | `if self.active_q_mask is not None:` | yield 返回后，若存在掩码则需恢复 logits 形状。 |
| L37-L41 | `ctx.logits = torch.zeros(...).masked_scatter_(...)` | 创建全零张量 `(B, T, vocab_size)`，通过 `masked_scatter_` 将模型实际计算出的 logits（仅 `T'` 个位置）散布回原始 `T` 个位置。这样下游代码看到的 logits 形状始终为 `(B, T, vocab_size)`。 |

---

### `attention` — 注意力层上下文管理器

```python
# 源文件: src/cache/prefix_cache.py L43-L107
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
        elif self._new_block_start:
            self.key_cache[layer_idx][self.active_seq_mask] = ctx.k
            self.value_cache[layer_idx][self.active_seq_mask] = ctx.v
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
            self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(0, 1)
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
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L43 | `@contextmanager` | 上下文管理器装饰器。 |
| L44-L53 | `def attention(self, layer_idx, x, ...):` | 参数包含当前层索引 `layer_idx`、隐藏状态 `x`、norm 层和 Q/K/V 投影线性层、attention_mask、position_ids。 |
| L55-L64 | `with super().attention(...) as ctx:` | 调用父类 `attention` 上下文管理器，父类完成 LayerNorm、Q/K/V 投影，将结果存入 `ctx.q`、`ctx.k`、`ctx.v`，形状均为 `(B, num_selected, num_heads*head_dim)`（仅在 `active_q_mask` 为 `None` 时或首次时计算全部位置）。 |
| L65-L68 | `if len(self.key_cache) <= layer_idx:` | **首次前向传播**：当前层的 key/value 缓存尚不存在，直接将父类计算出的 `ctx.k`、`ctx.v` 作为初始缓存追加到列表中。 |
| L66-L68 | `self.key_cache.append(ctx.k)` | `ctx.k` 形状 `(B, T_kv, head_dim)`，追加到 `key_cache` 末尾，索引对应 `layer_idx`。 |
| L69-L71 | `elif self._new_block_start:` | **新块开始**：整体替换活动序列的 KV 缓存。将新计算的 `ctx.k`/`ctx.v` 写入 `key_cache[layer_idx][active_seq_mask]` 位置。 |
| L72-L87 | `else:` | **常规步骤**：增量更新部分位置的缓存。 |
| L73 | `assert self.active_q_mask is not None` | 确保 `active_q_mask` 已初始化（非首次）。 |
| L74-L80 | `if layer_idx == 0: ...` | 仅在第 0 层计算 `_active_q_indices`（全局索引，供所有层共用）：  |
| L75 | `active_seq_idx = torch.where(self.active_seq_mask)[0]` | 获取活跃序列在原始 batch 中的索引，形状 `(B_active,)`。 |
| L76 | `m_nonzero = self.active_q_mask.nonzero(as_tuple=False)` | 将 `active_q_mask` (B_active, T) 中 `True` 的位置转为 `(N, 2)` 的索引矩阵。 |
| L77-L80 | `self._active_q_indices = (...)` | 构建 `(row_indices, col_indices)` 二元组，分别映射到原始 batch 维和序列维。`row_indices` 将局部 batch 索引映射回全局 batch 索引。 |
| L82-L84 | `self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)` | 将 `ctx.k` 展平后，按 `_active_q_indices` 索引赋值到缓存的对应位置，完成增量更新。`ctx.k` 从 `(B_active, T', H)` flatten 为 `(B_active*T', H)`。 |
| L85-L86 | `self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(0, 1)` | 同上，更新 value 缓存。 |
| L87-L88 | `ctx.k = self.key_cache[layer_idx][self.active_seq_mask]` | 从缓存中取出完整 KV（含未更新位置），赋值给 `ctx.k`/`ctx.v`，使注意力计算能看到完整序列。形状恢复为 `(B_active, T, H)`。 |
| L89-L101 | `if layer_idx == 0: ...` | 仅在第 0 层计算并在所有层间共享以下变量： |
| L91-L95 | `AttentionContext.select_position_ids(position_ids, self.active_q_mask)` | 根据 `active_q_mask` 筛选 Q 的位置 ID。若 `position_ids` 形状为 `(B, T)` 且 `active_q_mask` 为 `(B, T)` 布尔掩码，则 `q_position_ids` 形状变为 `(B, T')`。 |
| L96-L101 | `AttentionContext.convert_attention_mask(...)` | 将原始 attention mask 转换为加性掩码格式（0 表示保留，-inf 表示掩码），同时根据缓存的 KV 长度和当前 Q 长度调整掩码形状。 |
| L103-L105 | `ctx.q_position_ids = self._q_position_ids` | 将缓存的层间共享变量赋值到 `ctx`，供注意力计算使用。 |
| L107 | `yield ctx` | 交出控制权，让模型完成注意力计算。 |

**缓存三态决策总结：**

| 场景 | 条件 | 操作 |
|------|------|------|
| 首次前向 | `len(key_cache) <= layer_idx` | 追加新缓存 |
| 新块开始 | `_new_block_start == True` | 整行替换 `active_seq_mask` 位置 |
| 常规步骤 | 其他 | 仅更新 `active_q_mask` 指定位置 |

---

### `on_step_end` — 步骤结束回调

```python
# 源文件: src/cache/prefix_cache.py L109-L120
def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
    if self.active_q_mask is None:
        q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)
        if not self.use_dual:
            block_start = int(block_mask[0].int().argmax() + 1)
            q_mask[:, frame.prompts.size(-1) + block_start :] = True

        if is_adapted_from_ar(self.model_config):
            q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)

        self.active_q_mask = q_mask
    self._new_block_start = False
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L109 | `def on_step_end(self, block_mask, frame, delta):` | 每个生成步骤结束时调用。`block_mask` 形状 `(B, G)` 标记当前块内待处理 token；`frame` 包含 prompts 和 generated_tokens；`delta` 记录本步的状态变更。 |
| L110 | `if self.active_q_mask is None:` | 仅首次调用时初始化 `active_q_mask`，后续步骤保持不变。 |
| L111 | `q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)` | 将 `block_mask` (B, G) 左侧 padding P 个位置（prompt 区域），得到形状 `(B, P+G)` 即 `(B, T)`。左侧 padding 值为 `False`，即初始不选中 prompt 区域。 |
| L112-L114 | `if not self.use_dual: ...` | 非双模式下，将第一个 block 之后的所有位置设为 `True`（半自回归策略：从第一个 block 开始往后的所有位置都参与计算）。`block_start` 为第一个 block 结束位置。 |
| L116-L117 | `if is_adapted_from_ar(self.model_config): ...` | 如果模型适配自 AR（自回归）架构，将 `q_mask` 右移一位（左侧补 `False`），使得每个选中 token 对应其前一个位置也参与计算（Dream 模型需要前驱 token 信息）。 |
| L119 | `self.active_q_mask = q_mask` | 保存初始化完成的 `q_mask`，形状 `(B, T)`。 |
| L120 | `self._new_block_start = False` | 重置新块标志。 |

---

### `on_block_start` — 块开始回调

```python
# 源文件: src/cache/prefix_cache.py L122-L126
def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
    self._new_block_start = True
    self.active_q_mask = None
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L122 | `def on_block_start(self, block_mask, frame):` | 当新 block 开始时（如 batch 中新增序列）调用。 |
| L124 | `self._new_block_start = True` | 设置内部标志位。不直接清空缓存列表，因为缓存的 batch 维度必须保持。下一次 `attention` 调用时会用 `active_seq_mask` 整体替换对应行的 KV 缓存。 |
| L126 | `self.active_q_mask = None` | 清空 `active_q_mask`，触发 `on_step_end` 中重新初始化 `q_mask`。 |

## 关键函数和上下文管理器说明

### AttentionContext 数据类

```python
@dataclass
class AttentionContext:
    q: torch.Tensor                    # Query 张量
    k: torch.Tensor                    # Key 张量
    v: torch.Tensor                    # Value 张量
    residual: torch.Tensor             # 残差连接
    o: torch.Tensor | None = None      # 注意力输出（由模型赋值）
    attn_weight: torch.Tensor | None = None  # 注意力权重
    q_position_ids: torch.Tensor | None = None
    kv_position_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
```

### 关键静态方法

#### `select_position_ids`

选择 Query 和 Key-Value 的位置 ID：

```python
@classmethod
def select_position_ids(
    cls,
    position_ids: torch.Tensor | None = None,
    q_mask: torch.Tensor | None = None,
    kv_mask: torch.Tensor | None = None,
):
    q_position_ids, kv_position_ids = position_ids, position_ids
    if position_ids is not None:
        if q_mask is not None:
            q_position_ids = position_ids[q_mask].view(q_mask.size(0), -1)
        if kv_mask is not None:
            kv_position_ids = position_ids[kv_mask].view(kv_mask.size(0), -1)
    return q_position_ids, kv_position_ids
```

#### `convert_attention_mask`

将注意力掩码转换为注意力内核期望的格式：

```python
@classmethod
def convert_attention_mask(
    cls,
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
    query_length: int | None = None,
    key_value_length: int | None = None,
):
    # 布尔掩码: True 表示保留（关注），False 表示掩码
    # 转换为加性掩码：0 表示保留，-inf 表示掩码
    ...
```

## 使用示例和参数配置

### 基本使用

```python
from src.cache import PrefixCache
from transformers import AutoConfig

# 加载模型配置
model_config = AutoConfig.from_pretrained("path/to/model")

# 创建 PrefixCache 实例
cache = PrefixCache(model_config, use_dual=False)

# 在生成过程中使用
with cache.model_forward(input_embeddings) as ctx:
    # 模型处理 ctx.x
    ctx.logits = model_output
```

### 配置文件示例

```yaml
# configs/cache/prefix.yaml
_target_: src.cache.PrefixCache
use_dual: false
```

### 与模型集成

```python
class MyModel(nn.Module):
    def forward(self, input_ids, cache=None):
        hidden_states = self.embed(input_ids)
        
        if cache is not None:
            with cache.model_forward(hidden_states) as ctx:
                for layer_idx, layer in enumerate(self.layers):
                    with cache.attention(
                        layer_idx, ctx.x, 
                        layer.attn_norm, layer.q_proj, 
                        layer.k_proj, layer.v_proj,
                        attention_mask, position_ids
                    ) as attn_ctx:
                        attn_ctx.o = layer.attn(attn_ctx)
                        ctx.x = attn_ctx.o + attn_ctx.residual
                    
                    with cache.ffn(layer_idx, ctx.x) as ffn_ctx:
                        ffn_ctx.ffn_out = layer.ffn(ffn_ctx.x)
                        ctx.x = ffn_ctx.ffn_out + ffn_ctx.residual
                
                ctx.logits = self.lm_head(ctx.x)
        else:
            # 无缓存的标准前向传播
            ...
```

## 性能特点和适用场景

### 性能特点

| 特点 | 说明 |
|------|------|
| **计算效率** | 只计算需要更新的位置，减少约 50%-90% 的投影计算 |
| **内存效率** | 维护完整的 KV 缓存，内存占用与序列长度线性相关 |
| **批量支持** | 完全支持批量处理，适合 GPU 并行计算 |
| **实现简洁** | 代码结构清晰，易于理解和扩展 |

### 适用场景

1. **标准扩散生成**
   - 适用于大多数扩散语言模型的生成任务
   - 特别适合需要逐步更新部分位置的场景

2. **半自回归生成**
   - 配合 `use_dual=False` 使用
   - 支持块级别的生成策略

3. **批量推理**
   - 支持多序列并行处理
   - 自动处理序列结束的情况

### 性能对比

```
┌─────────────────────────────────────────────────────────────┐
│                    计算量对比示意图                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  无缓存方法:                                                 │
│  ████████████████████████████████████████████ 100%          │
│                                                             │
│  PrefixCache (首次):                                        │
│  ████████████████████████████████████████████ 100%          │
│                                                             │
│  PrefixCache (后续步骤, 假设 10% 位置需要更新):              │
│  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10%           │
│                                                             │
│  图例: █ 需要计算   ░ 复用缓存                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 局限性

1. **内存占用**：需要存储完整的 KV 缓存，对于超长序列可能有内存压力
2. **更新粒度**：只能选择性地更新位置，无法进行更细粒度的优化
3. **无自适应机制**：不像 dLLM-Cache 和 d2Cache 那样具有自适应刷新策略

## 与其他缓存方法的对比

| 特性 | PrefixCache | dLLM-Cache | d2Cache |
|------|-------------|------------|---------|
| KV 缓存 | ✓ | ✓ | ✓ |
| 选择性更新 | ✓ | ✓ | ✓ |
| 自适应刷新 | ✗ | ✓ | ✓ |
| 注意力权重分析 | ✗ | ✗ | ✓ |
| 置信度密度 | ✗ | ✗ | ✓ |
| 实现复杂度 | 低 | 中 | 高 |
| 推荐场景 | 基础生成 | 均衡性能 | 最佳性能 |

## 总结

PrefixCache 是 d2Cache 项目的基础缓存实现，它通过选择性计算和缓存复用机制，有效减少了扩散语言模型生成过程中的计算开销。虽然功能相对简单，但它为更高级的缓存方法（如 dLLM-Cache 和 d2Cache）奠定了基础架构。对于不需要复杂自适应策略的场景，PrefixCache 是一个高效且易于使用的选择。
