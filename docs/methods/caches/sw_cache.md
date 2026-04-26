# SlidingWindowCache 介绍

## 概述

它在 PrefixCache 的基础上，将每步需要重新计算的 Query 集合限制为一个固定大小的滑动窗口——只有当前步骤中最靠左的 `W` 个 [MASK] token 被选中，随着 token 逐个被解码，窗口自动向右滑动，始终追踪"解码前沿"。这种方法在不依赖 eager attention 的前提下，实现了可控的计算量缩减和天然的准左到右解码偏置。

---

## 1. d2Cache 项目架构

### 1.1 Frame / FrameDelta / DecodeRecord

扩散语言模型（Diffusion Language Models）的生成过程被抽象为一系列 **Frame** 和 **FrameDelta**：

- **Frame**：某一时刻的生成状态快照，包含：
  - `prompts`：提示词 token 序列，形状 `(batch_size, P)`
  - `generated_tokens`：生成 token 序列，形状 `(batch_size, G)`，其中未解码的位置为 `[MASK]`
  - `confidence`：每个位置的置信度分数
  - `steps`：每个位置的解码步数记录

- **FrameDelta**：记录从当前 Frame 到下一步 Frame 的变化，包含：
  - `transfer_index`：被解码（从 [MASK] 变为具体 token）的位置索引
  - `decoded_tokens`：这些位置的解码结果

- **DecodeRecord**：整个解码轨迹的完整记录，包含初始 Frame 和 T-1 个 FrameDelta。

生成过程就是反复执行 `frame = frame.apply_delta(delta)` 的过程。

### 1.2 生成循环

对于生成长度 `G`，扩散模型通常将生成分为多个block，每个块内迭代解码。当 `block_length = gen_length` 时，整个生成就是一个大块（SlidingWindowCache 的模式）；当 `block_length < gen_length` 时，采用 Semi-AR 模式（块间顺序、块内并行）。

vanilla 生成循环的伪代码：

```python
for block_idx in range(num_blocks):
    cache.on_block_start(block_mask, frame)   # (1) 块开始
    while True:
        cache.on_step_start(block_mask, frame) # (2) 步开始
        delta = generate_step(...)             # (3) 模型前向 + 解码
        if delta is None: break
        cache.on_step_end(block_mask, frame, delta) # (4) 步结束
        frame = frame.apply_delta(delta)       # (5) 应用 delta
    cache.on_block_end(...)                    # (6) 块结束
```

### 1.3 dCache 基类与上下文管理器

所有缓存方法都继承自 `dCache`（位于 `src/cache/base.py`），它通过三个 Python 上下文管理器拦截模型计算：

```
@model_forward  →  包装整个模型的前向传播，控制哪些 token 进入计算
@attention      →  在每个 Transformer 层的自注意力中拦截 Q/K/V
@ffn            →  在每个 Transformer 层的 FFN 中拦截
```

**核心机制**：只有被选中的少数 token（由 `active_q_mask` 标记）需要重新计算 Q/K/V，其余 token 的 KV 状态从缓存中获取。最终 logits 通过 `masked_scatter_` 恢复到完整的 `(B, T, vocab_size)` 形状。

### 1.4 active_q_mask 与 active_seq_mask

两个关键掩码控制缓存的行为：

| 掩码 | 形状 | 含义 |
|------|------|------|
| `active_seq_mask` | `(batch_size,)` | 标记哪些序列仍在活跃生成（产生新 token） |
| `active_q_mask` | `(batch_size, P+G)` | 标记当前步骤中哪些 **位置** 需要重新计算 Query |

在 `generate_step` 中（[vanilla.py](file:///Users/lier/codes/d2Cache/src/generation/vanilla.py#L76-L81)），`active_q_mask` 的生成部分被用于限制 `transfer_index_mask`，即只有窗口内的 [MASK] token 才能被选中解码：

```python
if past_key_values is not None and past_key_values.active_q_mask is not None:
    valid_mask = past_key_values.active_q_mask[:, prompt_length:]
    transfer_index_mask[active_seq_idx].logical_and_(valid_mask)
```

### 1.5 KV Cache 的矩形 Attention

缓存方法的核心优化来自对 attention 计算的改造。设完整序列长度为 `T = P + G`（prompt + 生成），缓存的 K、V 形状为 `(B_active, T, head_dim)`，而 Q 只来自 `active_q_mask` 选中的位置（设 `|Q| = W`）：

```
Q: (B_active, W, head_dim)        ← 只有窗口内的 token
K: (B_active, T, head_dim)        ← 完整缓存
V: (B_active, T, head_dim)        ← 完整缓存

attention_weights: (B_active, W, T)   ← 矩形注意力矩阵
```

这意味着 Q token 可以 attend 到所有历史 token（包括 prompt 和已解码的响应），只需要计算 `W` 个 token 的 Q 投影而非全部 `T` 个。

---

## 2. 算法原理和理论基础

### 2.1 核心思想

SlidingWindowCache 的核心理念非常直观：

> 在扩散语言模型的逐步解码中，我们只需要关注"解码前沿"——即最靠左侧的若干个仍未解码的 [MASK] token。随着 token 被逐个解码，这个"窗口"自然向右滑动。

形式化定义：

设生成序列长度为 `G`，窗口大小为 `W`。在第 t 步，令 `M^{(t)}` 为所有仍为 [MASK] 的位置集合，则第 t 步的 Query 集合（只考虑生成部分）为：

```
Q^{(t)}_{response} = 前 W 个最小的位置 i ∈ M^{(t)}
```

即：**窗口始终精确覆盖最靠左的 W 个 [MASK] token**。被解码的 token 立即移出窗口，窗口自动纳入下一个 [MASK] token。窗口内的 [MASK] 位置可能不连续——中间会跳过已解码的 token。

### 2.2 直观图示

```
整个生成序列 (G=64, 不再分块):
Pos:  [ 0] [ 1] [ 2] [ 3] [ 4] [ 5] [ 6] [ 7] [ 8] [ 9] [10] [11] [12] [13] ...
      └───────────────── 单一大块 (block = gen_length) ───────────────────────┘

Step 0 (初始):
  Tokens:  [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Window:  [▓] [▓] [▓] [▓] [▓] [▓] [▓] [▓] ...............................
           └─── W=8 个 mask ──┘
  Q = {0,1,2,3,4,5,6,7}  ← 最靠左的 8 个 mask（连续，因全是 [MASK]）

Step t1 (一些 token 被解码):
  Tokens:  [A] [M] [M] [B] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Mask位置:{   1,  2,      4,  5,  6,  7,  8,  9, 10, 11, 12, 13, ...}
  Window:       [▓] [▓]     [▓] [▓] [▓] [▓] [▓] [▓]
               └──── 前 8 个 mask（跳过已解码的 3）────┘
  Q = {1,2, 4,5,6,7,8,9}  ← 0 和 3 被驱逐，8,9 新接入

Step t2:
  Tokens:  [A] [C] [M] [B] [D] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Mask位置:{        2,           5,  6,  7,  8,  9, 10, 11, 12, 13, ...}
  Window:            [▓]        [▓] [▓] [▓] [▓] [▓] [▓] [▓]
                    └────── 前 8 个 mask ──────┘
  Q = {2, 5,6,7,8,9,10,11}  ← 1,4 被驱逐，10,11 新接入

...继续滑动直到全部解码...
```

## 3. 核心数据结构

### 3.1 类定义

```python
class SlidingWindowCache(dCache):

    def __init__(self, model_config, window_size: int = 32):
        super().__init__(model_config)
        self.window_size = window_size
        self.key_cache: list[torch.Tensor] = []      # 每层 Key 缓存
        self.value_cache: list[torch.Tensor] = []    # 每层 Value 缓存
        self.active_q_mask: torch.Tensor | None = None
        self._new_block_start = False
```

### 3.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_config` | `PretrainedConfig` | 必需 | 模型配置对象，用于判断是否为 AR-adapted 模型 |
| `window_size` | `int` | 32 | 滑动窗口大小 W，即每步最多重新计算的 token 数量 |

### 3.3 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `key_cache` | `list[torch.Tensor]` | 每层 Key 状态缓存，形状 `(B_active, P+G, head_dim)` |
| `value_cache` | `list[torch.Tensor]` | 每层 Value 状态缓存，形状 `(B_active, P+G, head_dim)` |
| `active_q_mask` | `torch.Tensor \| None` | 当前步骤的 Q 掩码，`True` 的位置需要重新计算 |
| `_new_block_start` | `bool` | 内部标志，标记是否为新块的第一个步骤 |

### 3.4 继承自 dCache 的属性

| 属性 | 说明 |
|------|------|
| `active_seq_mask` | 由生成循环在每步前设置，标记当前批次中哪些序列仍在活跃生成 |
| `mask_token_id` | [MASK] token 的 ID，从环境变量 `MASK_TOKEN_ID` 中获取 |

---

## 4. 详细代码流程分析

SlidingWindowCache 覆写了基类的三个方法：`model_forward`、`attention` 和 `on_step_end`、`on_block_start`。其中 `model_forward` 和 `attention` 的实现与 PrefixCache 几乎完全相同，复用了成熟的 KV Cache 增量更新机制。唯一的区别在 `on_step_end` 中 `active_q_mask` 的计算策略。

### 4.1 model_forward：选择性子集前向传播

```python
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
            ctx.logits = torch.zeros(
                (B, T, ctx.logits.size(-1)),
                dtype=ctx.logits.dtype,
                device=ctx.logits.device,
            ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)
```

**流程解析**：

1. **进入阶段**：若 `active_q_mask` 不为 None，则通过布尔索引 `x[self.active_q_mask]` 选出窗口内的 token（形状 `(B, W, C)`），只将这些 token 送入模型各层计算。

2. **退出阶段**：模型输出 logits 形状为 `(B, W, vocab_size)`，通过 `masked_scatter_` 散布回 `(B, T, vocab_size)` 的零张量中。未计算位置的 logits 保持为 0，在后续 `unmasking_fn` 中会被 `transfer_index_mask` 过滤掉。

3. **批次大小匹配**：当某些序列提前结束时，`B` 会减少。此时将 `active_q_mask` 从第一条序列扩展以匹配新的批次大小。

```
输入 x (B, T, C)
       │
       ▼
┌──────────────────┐
│ active_q_mask    │──── None ────▶ 直接使用完整 x（首次前向）
│ 是否存在？       │
└──────────────────┘
       │ Yes
       ▼
┌──────────────────┐
│ 选择窗口内 token │
│ x[mask] → (B, W, C)
└──────────────────┘
       │
       ▼
   模型各层计算
   (Q: W, KV: T)
       │
       ▼
┌──────────────────┐
│ 散布回原始形状   │
│ zeros + masked   │
│ scatter 恢复     │
└──────────────────┘
       │
       ▼
输出 logits (B, T, vocab_size)
```

### 4.2 attention：KV Cache 增量更新

```python
@contextmanager
def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
              attention_mask=None, position_ids=None):
    with super().attention(...) as ctx:
        if len(self.key_cache) <= layer_idx:
            # 情况 1: 首次前向传播，完整缓存
            self.key_cache.append(ctx.k)
            self.value_cache.append(ctx.v)
        elif self._new_block_start:
            # 情况 2: 新块开始，覆盖缓存
            self.key_cache[layer_idx][self.active_seq_mask] = ctx.k
            self.value_cache[layer_idx][self.active_seq_mask] = ctx.v
        else:
            # 情况 3: 增量更新，只写入窗口位置
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

        # 在第 0 层缓存 position_ids 和 attention_mask
        if layer_idx == 0:
            self._q_position_ids, self._kv_position_ids = \
                AttentionContext.select_position_ids(position_ids, self.active_q_mask)
            self._attention_mask = AttentionContext.convert_attention_mask(
                attention_mask, dtype=ctx.k.dtype,
                query_length=ctx.q.shape[1],
                key_value_length=self.value_cache[layer_idx].shape[1],
            )

        ctx.q_position_ids = self._q_position_ids
        ctx.kv_position_ids = self._kv_position_ids
        ctx.attention_mask = self._attention_mask
        yield ctx
```

**三种缓存更新策略**：

```
attention 被调用
       │
       ▼
┌─────────────────────────┐
│ len(key_cache) <=       │
│     layer_idx?          │── Yes ──▶ 首次前向：追加完整 K/V 到缓存
└─────────────────────────┘
       │ No
       ▼
┌─────────────────────────┐
│ _new_block_start?       │── Yes ──▶ 新块开始：用当前完整 K/V 覆盖缓存
└─────────────────────────┘
       │ No
       ▼
┌─────────────────────────────────────────────────────┐
│ 增量更新：                                            │
│  1. 只计算窗口 token 的 Q/K/V (由 model_forward 保证) │
│  2. 将新 K/V 写入缓存的对应位置                       │
│  3. 将完整的缓存 K/V 赋给 ctx.k/ctx.v                │
│     → 实现矩形的 (W, T) attention                    │
└─────────────────────────────────────────────────────┘
```

**关键细节**：

- **`layer_idx == 0` 时计算 `_active_q_indices`**：由于所有层的 `active_q_mask` 相同，只需在第 0 层计算一次索引映射（从 `(B, T)` 的 mask 映射到 `(B_active, T)` 的缓存坐标），后续层直接复用。

- **`position_ids` 处理**：Q 的 position_ids 只包含窗口内 token 的位置（由 `active_q_mask` 筛选），KV 的 position_ids 保持完整序列。这确保了模型知道每个 KV token 在序列中的绝对位置。

- **`attention_mask` 适配**：`query_length = |Q| = W`，`key_value_length = T`（完整序列长度），生成矩形的 causal/block attention mask。

### 4.3 on_step_end：窗口位置计算

这是 SlidingWindowCache 与 PrefixCache 唯一的区别所在：

```python
def on_step_end(self, block_mask, frame, delta):
    new_frame = frame.apply_delta(delta)
    P = frame.prompts.size(-1)
    G = new_frame.generated_tokens.size(-1)
    device = new_frame.generated_tokens.device

    # 1. 找出每个活跃序列中仍为 [MASK] 的位置
    remaining_mask = (
        new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
    )
    B_active = remaining_mask.size(0)

    # 2. 为每个序列选择前 W 个 [MASK] 位置
    response_q = torch.zeros((B_active, G), dtype=torch.bool, device=device)
    for i in range(B_active):
        positions = torch.where(remaining_mask[i])[0]
        if len(positions) == 0:
            continue
        window_positions = positions[:self.window_size]  # 取前 W 个
        response_q[i, window_positions] = True

    # 3. 拼接 prompt 部分（全部为 False）
    q_mask = F.pad(response_q, (P, 0), value=False)

    # 4. 适配 AR-adapted 模型
    if is_adapted_from_ar(self.model_config):
        q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)
        q_mask[:, P - 1] = q_mask[:, P:].any(dim=-1)

    self.active_q_mask = q_mask
    self._new_block_start = False
```

**逐步骤解析**：

```
输入：frame + delta → new_frame（应用了最新解码结果的帧）
                          │
                          ▼
┌────────────────────────────────────────┐
│ 1. 找出仍为 [MASK] 的位置              │
│    remaining_mask = (tokens == MASK)   │
│    形状: (B_active, G)                 │
└────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────┐
│ 2. 逐序列处理                          │
│    for each sequence:                  │
│      positions = where(remaining[i])   │
│      window = positions[:W]            │  ← 只取最靠左的 W 个
│      response_q[i, window] = True      │
└────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────┐
│ 3. 拼接 prompt 部分                    │
│    q_mask = pad(response_q, (P, 0))    │
│    prompt 部分全部为 False            │
│    形状: (B_active, P+G)              │
└────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────┐
│ 4. AR-adapted 模型特殊处理             │
│    将 mask 左移一位，P-1 位置           │
│    在响应非空时激活                     │
└────────────────────────────────────────┘
                          │
                          ▼
               self.active_q_mask = q_mask
```

**关键设计决策**：

- **Prompt 部分不加入 Q 集**：Prompt token 的 KV 已在首次前向传播中缓存，无需重复计算。将 prompt 加入 Q 集会浪费计算且无意义。通过 attention mask，窗口内 token 可以正常 attend 到缓存的 prompt KV。

- **AR-adapted 模型处理**：对于从自回归模型（如 LLaMA）适配而来的扩散模型，需要特殊处理位置编码。`q_mask[:, 1:]` 左移一位后用 `(0, 1)` 右侧补 False，确保位置对齐。同时 `P - 1`（最后一个 prompt 位置）在响应非空时被激活，使模型能正确解码第一个生成 token。

- **窗口可能不连续**：由于跳过已解码 token，`response_q[i]` 中 `True` 的位置可能不连续。这完全正确——只有仍为 [MASK] 的位置才需要重新计算。

### 4.4 on_block_start：块开始重置

```python
def on_block_start(self, block_mask, frame):
    self._new_block_start = True
    self.active_q_mask = None
```

块开始时重置两个标志：

- **`_new_block_start = True`**：告知 `attention` 方法在下一次计算时执行情况 2（覆盖缓存），而不是情况 3（增量更新）。因为新块中完整序列需要重新计算 KV。
- **`active_q_mask = None`**：告知 `model_forward` 不做子集选择，所有 `T` 个 token 全部送入模型。这确保新块的第一个步骤进行完整的 KV 初始化。

### 4.5 与生成循环的交互全景

结合第 1.2 节的生成循环，SlidingWindowCache 的完整生命周期：

```
Step 0 (块内第一步):
  on_block_start: _new_block_start=True, active_q_mask=None
  model_forward:  active_q_mask 为空 → 全量前向传播 (B, T, C)
  attention:      _new_block_start=True → 覆盖式缓存完整 K/V
  on_step_end:    根据解码后状态计算窗口 → W 个 [MASK] token
                  _new_block_start=False

Step 1 (块内第二步):
  model_forward:  active_q_mask 非空 → 只前向传播 W 个 token
  attention:      _new_block_start=False → 增量更新缓存
                  ctx.k/v 从缓存取出完整 K/V → 矩形 attention (W, T)
  on_step_end:    重新计算窗口 → 可能滑入新的 [MASK] token

Step 2...N:
  重复 Step 1 的流程，窗口不断滑动直到全部解码完成
```

---
