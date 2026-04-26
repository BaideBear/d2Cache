# SlidingWindow Cache 设计方案（修正版）

## 1. 想法形式化

### 1.1 你的原始想法

> 我们要更新的 Q 集合是一个窗口，窗口的大小固定，每当一个 mask 被解码之后，被解码的 token 会被驱逐出窗口，然后从后方接入新的 mask，直到全部解码完成。

### 1.2 精确理解与形式化

**关键理解**：窗口本身就是"块"（block）。即：

* `block_size = gen_length`（整个生成长度就是一个大块，不分小段）

* `window_size` 是固定的查询子集大小（等价于原来的 `block_size`）

* 窗口作为一个"解码前沿"（decoding front），在整个生成序列上从左向右滑动

设生成响应长度为 $G$（`gen_length`），窗口大小为 $W$（`window_size`）。在第 $t$ 步时：

* $\mathcal{M}^{(t)} = \{ i \in [0, G) \mid \text{token}_i = \text{[MASK]} \}$ 为第 $t$ 步所有仍为 mask 的位置

* 第一个 mask 位置 $s^{(t)} = \min \mathcal{M}^{(t)}$（如果 $\mathcal{M}^{(t)}$ 非空）

则第 $t$ 步的 Q 集合（需要重新计算的 token 位置，只考虑响应部分）定义为：

$$\mathcal{Q}_{\text{response}}^{(t)} = \text{前 } W \text{ 个最小的 } i \in \mathcal{M}^{(t)}$$

即：**窗口始终精确覆盖最靠左的 $W$ 个 mask token**。被解码的 token 立即移出窗口，因此窗口内的 mask 位置可能不连续——中间会跳过已解码的 token。当最靠左的 mask token 被解码后，它被驱逐，窗口自动纳入下一个 mask token。

### 1.3 直观图示

```
整个生成序列 (G=64, 不再分块):
Pos:  [ 0] [ 1] [ 2] [ 3] [ 4] [ 5] [ 6] [ 7] [ 8] [ 9] [10] [11] [12] [13] ...
      └───────────────── 单一大块 (block = 整个 gen_length) ─────────────────────┘

Step 0 (初始):
  Tokens:  [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Window:  [▓] [▓] [▓] [▓] [▓] [▓] [▓] [▓] .................................
           └─── W=8 个 mask ──┘
  Q = {0,1,2,3,4,5,6,7}  ← 最靠左的 8 个 mask（连续，因全是 mask）

Step t1 (一些 token 被解码):
  解码了位置 0, 3。它们不再是 mask：
  Tokens:  [A] [M] [M] [B] [M] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Mask位置:{   1,  2,      4,  5,  6,  7,  8,  9, 10, 11, 12, 13, ...}
  Window:       [▓] [▓]     [▓] [▓] [▓] [▓] [▓] [▓]
               └──── 前 8 个 mask（跳过已解码的 3）────┘
  Q = {1,2, 4,5,6,7,8,9}  ← 0 和 3 被驱逐，8,9 新接入

Step t2:
  位置 1,4 被解码：
  Tokens:  [A] [C] [M] [B] [D] [M] [M] [M] [M] [M] [M] [M] [M] [M] ...
  Mask位置:{        2,           5,  6,  7,  8,  9, 10, 11, 12, 13, ...}
  Window:            [▓]        [▓] [▓] [▓] [▓] [▓] [▓] [▓]
                    └────── 前 8 个 mask ──────┘
  Q = {2, 5,6,7,8,9,10,11}  ← 1,4 被驱逐，10,11 新接入

Step t3:
  位置 2,5,6,7,8 被解码：
  Tokens:  [A] [C] [E] [B] [D] [F] [G] [H] [I] [M] [M] [M] [M] [M] ...
  Mask位置:{                                 9, 10, 11, 12, 13, ...}
  Window:                                   [▓] [▓] [▓] [▓] [▓] ...
                                          └─ 前 8 个 mask ─┘
  Q = {9,10,11,12,13,14,15,16}

...继续滑动直到全部解码...
```

**核心特性**：被解码的 token 立即驱逐，窗口内只保留 mask token。mask 位置可能不连续。窗口始终追踪解码前沿。

***

## 2. 想法的深化与扩展

### 2.1 核心优势分析

#### 2.1.1 与现有方法的对比

| 维度                 | PrefixCache (dual, B=32) | d2Cache                         | **SlidingWindow** |
| ------------------ | ------------------------ | ------------------------------- | ----------------- |
| 分块方式               | $N\_b = G/B$ 个小块         | $N\_b = G/B$ 个小块                | **单一全局块**         |
| Q 集大小              | $B$（每块内固定）               | $\approx k + \text{extras}$（变长） | **$W$（固定）**       |
| 选择策略               | 无（全块计算）                  | 置信度+注意力滚出                       | **纯位置驱动**         |
| 需要 eager attention | 否                        | 是                               | **否**             |
| 解码模式               | 块内无序                     | 置信度引导                           | **全局准左到右**        |
| 实现复杂度              | 低                        | 高                               | **极低**            |

#### 2.1.2 SlidingWindow 的独特价值

1. **不需要 eager attention**：与 d2Cache 不同，完全基于位置信息决定 Q 集，兼容 SDPA / FlashAttention，大幅降低内存和 wall-clock 时间。

2. **全局准左到右解码**：d2Cache 论文证明左到右偏置可以提高生成质量（通过确定性先验）。SlidingWindow 通过窗口滑动机制天然实现，不需要额外 `sigma` 参数。

3. **计算量可精确控制**：Q 集恒定为 $W$，通过调整 $W$ 精确 trade-off 速度和质量。$W$ 越小越快，$W$ 越大越接近 PrefixCache。

4. **实现极简**：只需追踪第一个 mask 位置，不需要注意力滚出矩阵、置信度缓存等复杂状态。

### 2.2 算法设计

#### 2.2.1 数据结构

```python
class SlidingWindowCache(dCache):
    def __init__(self, model_config, window_size: int = 32):
        self.window_size = window_size       # W
        self.key_cache: list[torch.Tensor]   # 每层 K 缓存
        self.value_cache: list[torch.Tensor] # 每层 V 缓存
        self.active_q_mask = None            # 当前 Q 掩码 (batch, prompt+gen)
        self._new_block_start = False        # 新块标志
```

#### 2.2.2 核心算法：窗口位置计算 (on\_step\_end)

```
输入：new_frame (应用 delta 后的帧)
输出：self.active_q_mask

对于每个活跃序列 i:
  1. 找到所有仍为 mask 的位置:
     positions = {j | new_frame.generated_tokens[i,j] == MASK}
  
  2. 若 positions 为空（已全部解码）:
     skip
  
  3. 取前 W 个 mask 位置（按索引升序）:
     window_positions = positions[:window_size]
  
  4. 将 prompt 部分全部加入 Q 集:
     q_mask[i, :prompt_length] = True
     q_mask[i, prompt_length + window_positions] = True
```

#### 2.2.3 与解码流程的交互

关键在 `generate_step` 中，当 cache 的 `active_q_mask` 不为 None 时：

```python
# generate_step 中的逻辑（无需修改，现有机制自动生效）
if past_key_values is not None and past_key_values.active_q_mask is not None:
    valid_mask = past_key_values.active_q_mask[:, prompt_length:]
    transfer_index_mask[active_seq_idx].logical_and_(valid_mask)
```

这意味着 `unmasking_fn` 只能从窗口内的 mask 中选择 token 解码。窗口外的 mask 无法被选中。

#### 2.2.4 生命周期

```
on_block_start:
  → _new_block_start = True, active_q_mask = None
  → 在第一个 step 中，所有 token 都会重新计算（无窗口）

on_step_end (第一个 step 之后):
  → 根据解码后状态计算窗口位置
  → active_q_mask = 窗口 + prompt
  → _new_block_start = False

on_step_end (后续 steps):
  → 每次重新计算窗口位置（窗口随解码推进而滑动）
  → 只更新窗口内 token 的 K、V
```

### 2.3 窗口大小 W 的选择策略

* $W = 1$：退化为严格逐 token 自回归（每步只算一个 Q）

* $W = G$：等价于 PrefixCache（计算整个生成序列）

* $W = 32$（推荐）：与典型 block\_size 一致的好默认值

* $W = 64$：更接近 PrefixCache，更保守

***

## 3. 加速比的数学分析

### 3.1 符号约定

| 符号        | 含义                  | 典型值   |
| --------- | ------------------- | ----- |
| $P$       | 提示词长度               | 128   |
| $G$       | 生成长度（即 block\_size） | 256   |
| $W$       | 滑动窗口大小              | 32    |
| $L$       | 层数                  | 32    |
| $d$       | 隐藏维度                | 4096  |
| $d\_{ff}$ | FFN 中间维度            | 14336 |

### 3.2 各缓存方法的 Q 集大小

| 方法                       | 每步 $Q\_{\text{size}}$  | 说明                     |
| ------------------------ | ---------------------- | ---------------------- |
| Baseline (无缓存)           | $P + G = 384$          | 全部重新计算                 |
| PrefixCache (B=32, dual) | $P + B = 160$          | 提示词 + 当前块              |
| d2Cache                  | $\approx P + 50 = 178$ | 需要 eager attention     |
| **SlidingWindow**        | **$P + W = 160$**      | W=32 时与 PrefixCache 相同 |

### 3.3 理论加速比

$$\text{Speedup} = \frac{Q\_{\text{size}}^{\text{baseline}}}{Q\_{\text{size}}^{\text{method}}} = \frac{P + G}{P + W}$$

| W   | Q\_size | 理论 FLOPs 加速比 |
| --- | ------- | ------------ |
| 16  | 144     | **\~2.67×**  |
| 32  | 160     | **\~2.40×**  |
| 64  | 192     | **\~2.00×**  |
| 128 | 256     | **\~1.50×**  |
| 无缓存 | 384     | 1.0×         |

**重要说明**：FLOPs 加速比在 $2\times$-$2.7\times$ 范围内，相比之前错误理解（12×）大幅降低。这是因为 prompt 部分 $P$ 总是需要重新计算（它不在窗口"滑动"范围内）。SlidingWindow 的核心价值在于：

1. **兼容 SDPA/FlashAttention**：不需要 eager attention（d2Cache 需要），wall-clock 优势显著
2. **准左到右解码**：天然的质量提升偏置
3. **实现极简**：只有 \~130 行代码

### 3.4 Wall-Clock 优势

虽然 FLOPs 加速比没有 PrefixCache 那么高，但 SlidingWindow 有独特的 wall-clock 优势：

1. **与 PrefixCache 对比**：两者的 Q\_size 相近（都约 160），但 SlidingWindow 不需要分块，减少了块切换的开销
2. **与 d2Cache 对比**：SlidingWindow 不需要 eager attention，在 SDPA/FlashAttention 下实际速度快于 d2Cache（尽管理论 FLOPs 相近）
3. **质量优势**：准左到右解码模式可能减少总步数（高置信度 token 集中在窗口前沿）

### 3.5 为什么加速比看起来不大？

因为 prompt 部分的 token 总是需要参与注意力计算（它们提供了上下文）。这是所有 dLLM cache 方法都无法绕过的基础开销。真正的加速来源于：

* 生成部分只需要计算 $W$ 个 token 而不是 $G$ 个

* 当 $G$ 很大时（如 512、1024），节省效果更明显

* 配合 SDPA/FlashAttention 的实际 wall-clock 收益通常优于 FLOPs 理论值

***

## 4. 实现总结

### 4.1 已创建的文件

| 文件                      | 说明                              |
| ----------------------- | ------------------------------- |
| `src/cache/sw_cache.py` | SlidingWindowCache 类实现（\~130 行） |
| `configs/cache/sw.yaml` | 配置文件（`window_size: 32`）         |
| `src/cache/__init__.py` | 添加了 `SlidingWindowCache` 导出     |

### 4.2 使用方式

```bash
# 基本使用（block_length = gen_length = 256, window_size = 32）
python eval.py \
    model=llada-inst \
    cache=sw \
    cache.window_size=32 \
    generation=vanilla \
    generation.gen_length=256 \
    generation.block_length=256 \
    dataset.name=gsm8k

# 激进配置（window_size=16）
python eval.py \
    model=llada-inst \
    cache=sw \
    cache.window_size=16 \
    generation=vanilla \
    generation.gen_length=256 \
    generation.block_length=256 \
    dataset.name=gsm8k

# 配合并行解码
python eval.py \
    model=llada-inst \
    cache=sw \
    cache.window_size=32 \
    generation=vanilla \
    generation.gen_length=256 \
    generation.block_length=256 \
    generation.threshold=0.9 \
    dataset.name=gsm8k
```

### 4.3 实现要点

`on_step_end` 的核心逻辑：

```python
def on_step_end(self, block_mask, frame, delta):
    new_frame = frame.apply_delta(delta)
    P = frame.prompts.size(-1)
    G = new_frame.generated_tokens.size(-1)
    
    remaining_mask = new_frame.generated_tokens[self.active_seq_mask] == self.mask_token_id
    
    response_q = torch.zeros((B_active, G), dtype=torch.bool)
    for i in range(B_active):
        positions = torch.where(remaining_mask[i])[0]
        if len(positions) == 0:
            continue
        window_positions = positions[:self.window_size]
        response_q[i, window_positions] = True
    
    q_mask = F.pad(response_q, (P, 0), value=False)
    
    if is_adapted_from_ar(self.model_config):
        q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)
        q_mask[:, P - 1] = q_mask[:, P:].any(dim=-1)
    
    self.active_q_mask = q_mask
    self._new_block_start = False
```

### 4.4 Pipeline 变量解析后的用法

```bash
# bl=gen_len=256 表示单一全局块，win=32 表示窗口大小
python eval.py cache=sw generation.block_length=256 generation.gen_length=256 model=llada-inst dataset.name=gsm8k
```

