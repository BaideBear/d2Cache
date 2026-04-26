# AR 解码策略


## 算法逻辑精要

AR自回归解码强制扩散模型按从左到右的顺序依次生成token：通过get_ar_prefix_mask将每步的解码范围严格限制在紧邻已生成前缀的连续区间内，确保因果性不被破坏；autoregressive_unmasking在该受限前缀中按置信度选择解码位置。可选地，当启用threshold参数时，连续置信度不低于阈值的多个前缀位置可一步并行解码，在保持自回归因果约束的前提下实现加速。其复用vanilla的generate_step主流程，通过注入自定义的unmasking_fn实现顺序约束。

## 概述

AR（Autoregressive，自回归）解码策略将扩散模型的生成过程改造为严格的自回归模式。与 Vanilla 策略不同，AR 策略强制按照从左到右的顺序解码 token，确保每个 token 的预测只依赖于其左侧已解码的上下文，从而实现与传统自回归语言模型一致的生成行为。

## 算法原理

### 核心思想

AR 策略的核心洞察是：

> **通过限制解码顺序为从左到右，扩散模型可以模拟自回归生成行为，同时保留双向注意力机制的优势。**

### 自回归约束

在 AR 解码中：

1. **顺序约束**：只能解码当前最左边的可解码位置
2. **前缀限制**：解码范围限制在连续的可解码前缀
3. **置信度阈值**：可选地使用阈值进行并行前缀解码

### 与 Vanilla 的区别

```
Vanilla 解码:
位置:     0     1     2     3     4
置信度:  0.95  0.72  0.88  0.45  0.91
选择:     ✓     ✗     ✓     ✗     ✓
解码顺序: 位置 0, 2, 4（按置信度）

AR 解码:
位置:     0     1     2     3     4
置信度:  0.95  0.72  0.88  0.45  0.91
选择:     ✓     ✗     ✗     ✗     ✗
解码顺序: 位置 0（最左可解码）
```

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    AR 解码流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   单步解码过程                            │    │
│  │                                                         │    │
│  │  1. 确定可解码前缀                                        │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 找到连续的可解码前缀                │           │    │
│  │     │  ar_prefix_mask = get_ar_prefix_mask()│           │    │
│  │     │                                       │           │    │
│  │     │  示例:                                 │           │    │
│  │     │  位置:   0   1   2   3   4            │           │    │
│  │     │  状态:  [A] [M] [M] [A] [M]           │           │    │
│  │     │  前缀:   ✓   ✓   ✓   ✗   ✗            │           │    │
│  │     │                                       │           │    │
│  │     │  # 位置 3 已解码，但位置 1,2 未解码   │           │    │
│  │     │  # 所以前缀只到位置 2                 │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  2. 模型前向传播                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  logits = model([prompt + tokens])    │           │    │
│  │     │  confidence = logits.max(dim=-1)      │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  3. 自回归选择                                           │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  if threshold:                        │           │    │
│  │     │    # 并行前缀解码                      │           │    │
│  │     │    选择前缀中置信度 >= threshold 的    │           │    │
│  │     │    直到第一个低于阈值的位置            │           │    │
│  │     │  else:                                │           │    │
│  │     │    # 标准 AR                          │           │    │
│  │     │    只解码第一个位置                    │           │    │
│  │     │                                       │           │    │
│  │     │  示例 (threshold=0.9):                │           │    │
│  │     │  前缀位置: 0   1   2                  │           │    │
│  │     │  置信度:  0.95 0.88 0.72              │           │    │
│  │     │  选择:    ✓    ✓    ✗                 │           │    │
│  │     │  解码: 位置 0, 1 (直到位置 2 失败)     │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  4. 更新 Frame                                           │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 解码选中的位置                      │           │    │
│  │     │  frame = frame.apply_delta(delta)     │           │    │
│  │     └───────────────────────────────────────┘           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 前缀掩码生成

```
get_ar_prefix_mask 的工作原理:

输入: transfer_index_mask = [True, True, False, True, False]
      # True 表示该位置可解码

步骤:
1. 找到第一个可解码位置: start = 0
2. 从 start 开始，找到第一个不可解码位置: end = 2
3. 生成掩码: [True, True, False, False, False]

输出: 只有位置 0, 1 在前缀中
```

## 核心参数

### AR 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | None | 可选的置信度阈值，用于并行前缀解码 |
| `debias` | bool | False | 是否启用位置感知偏差校准 |

### 继承参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列总长度 |
| `block_length` | int | - | 块大小 |
| `num_transfer_tokens` | int | 1 | 每步最少解码数量 |
| `temperature` | float | 0.0 | 采样温度 |
| `alg` | str | "maskgit_plus" | 置信度计算算法 |

## 详细代码流程分析

AR 解码策略的核心实现在 `src/generation/ar.py` 中，包含两个独立辅助函数 `get_ar_prefix_mask` 和 `autoregressive_unmasking`，以及入口函数 `ar_generate`。AR 复用 vanilla 的 `generate_step`，通过注入自定义 `unmasking_fn` 实现自回归约束。以下按源码行号顺序逐模块讲解。

### get_ar_prefix_mask

```python
# 源文件: src/generation/ar.py L14-L37
def get_ar_prefix_mask(transfer_index_mask: torch.Tensor) -> torch.Tensor:
    """
    Restrict autoregressive decoding to the contiguous transferable span that
    immediately follows the generated prefix. This method masks out any tokens
    after the first non-transferable token.
    """
    batch_size, seq_len = transfer_index_mask.shape
    positions = torch.arange(seq_len, device=transfer_index_mask.device).expand(
        batch_size, -1
    )
    has_transferable = transfer_index_mask.any(dim=-1)
    start = torch.where(
        has_transferable,
        transfer_index_mask.int().argmax(dim=-1),
        torch.full((batch_size,), seq_len, device=transfer_index_mask.device),
    )
    after_start = positions >= start.unsqueeze(-1)
    first_invalid = (~transfer_index_mask) & after_start
    end = torch.where(
        first_invalid.any(dim=-1),
        first_invalid.int().argmax(dim=-1),
        torch.full((batch_size,), seq_len, device=transfer_index_mask.device),
    )
    return after_start & (positions < end.unsqueeze(-1))
```

**逐行解释：**

- L14：函数签名。输入 `transfer_index_mask` 是一个布尔张量，形状 `(B, seq_len)`，True 表示该位置可以解码（即当前为 [MASK]）。返回一个布尔掩码，仅保留紧邻已生成前缀的连续可解码区间。
- L19-L22：获取 `batch_size` 和 `seq_len`，构造 `positions` 张量 `[[0,1,2,...,seq_len-1]]` 扩展到 batch 维度。
- L24：`has_transferable` 标记每个序列是否至少有一个可解码位置。
- L25-L29：找到每个序列中第一个可解码位置 `start`。对有可解码位置的序列，使用 `argmax` 找到第一个 True 的索引；对无可解码位置的序列，填入 `seq_len`（后续条件将过滤掉整个序列）。
- L30：`after_start = positions >= start.unsqueeze(-1)` 标记每个序列中位于起始位置及其之后的所有位置。
- L31：`first_invalid = (~transfer_index_mask) & after_start` 在 `after_start` 区间中找到第一个不可解码（False）的位置。取反 `~transfer_index_mask` 将不可解码位置变为 True。
- L32-L36：`end` 为每个序列中第一个不可解码位置的索引。使用 `argmax` 找到 `first_invalid` 中第一个 True 的位置。若无不可解码位置，则 `end = seq_len`（整个区间都可解码）。
- L37：返回 `after_start & (positions < end)`。两个条件取交集得到 `[start, end)` 区间——即从第一个可解码位置开始到第一个不可解码位置之前（不含）的连续前缀。例如 `transfer_index_mask = [True, True, False, True]` → 返回 `[True, True, False, False]`，位置 3 虽可解码但因不连续而被排除。

### autoregressive_unmasking

```python
# 源文件: src/generation/ar.py L40-L98
def autoregressive_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: torch.Tensor | int,
    threshold: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    Select tokens to fix based on confidence while preserving left-to-right order.
    Unlike vanilla confidence-based unmasking, autoregressive decoding restricts
    selection to the contiguous transferable prefix immediately following the
    generated prefix, so any multi-token update still advances sequentially.

    Args:
        scores: A tensor of shape [B, gen_length] containing token confidence.
        transfer_index_mask: A boolean tensor of shape [B, gen_length] indicating
            which contiguous prefix tokens can be transferred.
        min_transfer_tokens: A tensor of shape [B,] indicating the minimum number
            of tokens to be transferred at each step.
        threshold: A threshold for parallel decoding. If provided, all prefix
            tokens whose confidence stays above this threshold will be kept until
            the first failure.
    """
    batch_size, seq_len = scores.shape
    device = scores.device
    if isinstance(min_transfer_tokens, int):
        min_transfer_tokens = torch.full(
            (batch_size,), min_transfer_tokens, device=device, dtype=torch.long
        )
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
    allowed_count = transfer_index_mask.sum(dim=-1)
    start = torch.where(
        allowed_count > 0,
        transfer_index_mask.int().argmax(dim=-1),
        torch.zeros(batch_size, dtype=torch.long, device=device),
    )
    gather_index = (start.unsqueeze(-1) + positions).clamp(max=seq_len - 1)
    prefix_scores = torch.gather(scores, dim=1, index=gather_index)
    valid_prefix_mask = positions < allowed_count.unsqueeze(-1)

    if threshold is not None:
        fail_mask = valid_prefix_mask & (prefix_scores < threshold)
        selected_count = torch.where(
            fail_mask.any(dim=-1),
            fail_mask.int().argmax(dim=-1),
            allowed_count,
        )
    else:
        selected_count = allowed_count

    selected_count = torch.minimum(
        torch.maximum(min_transfer_tokens.to(allowed_count.dtype), selected_count),
        allowed_count,
    )
    selected_mask = valid_prefix_mask & (positions < selected_count.unsqueeze(-1))
    flat_indices = gather_index[selected_mask]
    split_sizes = selected_mask.sum(dim=-1).tolist()
    return tuple(torch.split(flat_indices, split_sizes))
```

**逐行解释：**

- L40-L61：函数签名和 docstring。接收 `scores`（各位置置信度）、`transfer_index_mask`（AR 前缀掩码，已由 `get_ar_prefix_mask` 预处理为连续前缀）、`min_transfer_tokens`（每步最少解码数量，支持标量或逐 batch 张量）、`threshold`（可选的并行前缀解码阈值）。返回一个元组，每个元素为对应序列选中位置的索引张量。
- L62-L63：提取 `batch_size`、`seq_len` 和 `device`。
- L64-L67：若 `min_transfer_tokens` 是标量 int，则将其扩展为 batch 大小的张量，方便后续逐序列操作。
- L68：构造 `positions` 张量 `[[0,1,...,seq_len-1]]` 扩展到 batch 维度。
- L69：`allowed_count = transfer_index_mask.sum(dim=-1)` 统计每个序列的可解码位置总数。对于 AR 模式，这等于连续前缀长度。
- L70-L74：找到每个序列的前缀起始位置 `start`。对有可解码位置的序列用 `argmax` 找第一个 True；否则设为 0（该序列已被过滤）。
- L75：`gather_index = (start.unsqueeze(-1) + positions).clamp(max=seq_len - 1)`。构造一个"从 start 开始的递增索引"用于后续 gather 操作。例如 start=3 时 gather_index 为 [3,4,5,...,3+seq_len-1]，clamp 限制不超出边界。
- L76：`prefix_scores = torch.gather(scores, dim=1, index=gather_index)`。使用 gather_index 从 scores 中按起始位置对齐地收集前缀置信度，使得各序列的前缀分数在 dim=1 维度上从索引 0 开始对齐排列。
- L77：`valid_prefix_mask = positions < allowed_count.unsqueeze(-1)`。标记各序列中属于有效前缀的位置（从 0 到 allowed_count-1）。
- L79-L86：**并行前缀解码模式**（`threshold is not None`）。`fail_mask` 标记前缀中置信度低于 threshold 的位置。`selected_count` 使用 `argmax` 找到第一个失败位置——该位置及之前的 token 都被选中（恰好为失败位置的索引 = 选中的数量，因为从 0 开始）。例如前缀置信度 [0.95, 0.92, 0.88]，threshold=0.9 → fail_mask=[F,F,T]，argmax=2 → 选中前 2 个位置。
- L87-L88：**标准 AR 模式**（`threshold is None`）。`selected_count = allowed_count`，仅解码前缀的第一个位置。因为 `min_transfer_tokens` 通常为 1，且后续的 `valid_prefix_mask & (positions < selected_count)` 中 selected_count 会被 min_transfer_tokens 覆盖为 1。
- L91-L94：`selected_count = torch.minimum(torch.maximum(min_transfer_tokens, selected_count), allowed_count)`。将 `selected_count` 限制在 `[min_transfer_tokens, allowed_count]` 范围内。这确保：至少解码 `min_transfer_tokens` 个 token（保证推进），且不超过可用的前缀长度。
- L95：`selected_mask = valid_prefix_mask & (positions < selected_count.unsqueeze(-1))`。构造最终的选择掩码——在前缀范围内选取前 `selected_count` 个位置。
- L96：`flat_indices = gather_index[selected_mask]` 收集所有被选中位置的全局索引（而非对齐后的前缀偏移），展平为一维张量。
- L97：`split_sizes = selected_mask.sum(dim=-1).tolist()` 统计每个序列选中的数量，用于后续 split 操作。
- L98：`return tuple(torch.split(flat_indices, split_sizes))` 按各序列的选中数量将 flat_indices 拆分回批量的索引元组。

### ar_generate: unmasking_fn

```python
# 源文件: src/generation/ar.py L179-L198
    def unmasking_fn(
        *,
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        active_transfer_mask = transfer_index_mask & block_mask
        ar_transfer_mask = get_ar_prefix_mask(active_transfer_mask)
        return (
            autoregressive_unmasking(
                scores=scores,
                transfer_index_mask=ar_transfer_mask,
                min_transfer_tokens=num_transfer_tokens,
                threshold=threshold,
            ),
            {},
        )
```

**逐行解释：**

- L179-L188：`unmasking_fn` 是定义在 `ar_generate` 内部的闭包函数，使用仅关键字参数。它将作为 `unmasking_fn` 参数传入 `generate_step`，在每步解码时被调用以决定选择哪些位置。
- L189：`active_transfer_mask = transfer_index_mask & block_mask` 将全局可转移位置限制在当前 block 范围内。只有处于当前块内且尚未被解码的 [MASK] 位置才可被选中。
- L190：`ar_transfer_mask = get_ar_prefix_mask(active_transfer_mask)` 调用独立函数进一步限制为连续前缀。这一步是 AR 解码的核心——确保只解码紧邻已生成前缀的连续 [MASK] 区间。
- L191-L198：调用 `autoregressive_unmasking` 进行自回归选择，传入 AR 前缀掩码、最少解码数量、以及可选的并行前缀解码阈值 `threshold`。返回的第二个元素为空字典 `{}`，表示无需额外数据传递（与 KLASS 不同，AR 不需要跟踪历史状态）。

注意：`threshold` 变量被 `unmasking_fn` 闭包捕获自 `ar_generate` 的参数，因此无需显式传入。

### ar_generate: block 循环

```python
# 源文件: src/generation/ar.py L200-L257
    deltas = []

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)

        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=num_transfer_tokens,
                unmasking_fn=unmasking_fn,
                attention_mask=attention_mask,
                past_key_values=cache,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                debias=debias,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                break
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
```

**逐行解释：**

- L200：初始化全局 `deltas` 列表，用于累积所有块的解码步骤记录。
- L202：遍历所有块，`block_idx` 从 0 到 `num_blocks - 1`。
- L203-L207：为当前块创建 `block_mask`，初始化为全 False。
- L208-L211：将当前块范围 `[block_idx * block_length, (block_idx + 1) * block_length)` 置为 True，标记这些位置为当前块的解码目标。
- L213：保存当前块开始时的 Frame 快照 `start_frame`，供 Cache 记录块边界。
- L214-L215：若启用了 Cache（如 dCache 或 D2Cache），调用 `cache.on_block_start` 通知新块开始。AR 策略与 KV Cache 配合极佳，因为自回归解码顺序使得 Cache 可以精确复用。
- L217：初始化 `block_deltas`，累积当前块内的每步 delta。
- L218：进入 `while True` 循环，在当前块内反复解码直到无法继续。
- L219-L220：若启用了 Cache，调用 `cache.on_step_start` 准备当前步的 KV Cache 状态。
- L221-L239：调用复用的 `generate_step` 函数执行单步解码。关键区别是传入了自定义的 `unmasking_fn`，使得选择逻辑遵循 AR 的自回归约束。`past_key_values=cache` 传递 Cache 对象，实现 KV Cache 加速——由于 AR 严格从左到右解码，已计算过的 KV 状态可以精确复用。
- L240-L241：若 `delta is None`（无更多可解码 token），break 跳出 while 循环。
- L242-L243：若启用了 Cache，调用 `cache.on_step_end` 更新 KV Cache 状态，新解码位置的 KV 被记录供后续复用。
- L245：将 delta 转移到 CPU 并追加到 `block_deltas`，减少 GPU 显存占用。
- L246：`frame = frame.apply_delta(delta)` 将 delta 中选中的 token 应用到 Frame 的 `generated_tokens` 中。由于 AR 从左到右解码，新 token 总是在已生成前缀的末尾，拓展了连续前缀。
- L248-L249：若启用了 Cache，调用 `cache.on_block_end` 通知该块结束，进行块级别的 Cache 清理或优化。
- L251：将当前块的所有 delta 追加到全局 `deltas`。
- L253-L257：所有块处理完毕后，构建并返回 `DecodeRecord`。包含 `initial_frame`（初始 Frame，转移到 CPU）、`deltas`（所有步骤记录）和 `block_length`。

## Token 选择策略

### 标准 AR 模式

```
无 threshold 时:

位置:           0       1       2       3       4
状态:          [M]     [M]     [M]     [M]     [M]
前缀掩码:       ✓       ✓       ✓       ✓       ✓
置信度:       0.95    0.72    0.88    0.45    0.91

选择: 位置 0（前缀的第一个位置）
解码后:        [A]     [M]     [M]     [M]     [M]

下一步:
位置:           0       1       2       3       4
状态:          [A]     [M]     [M]     [M]     [M]
前缀掩码:       ✗       ✓       ✓       ✓       ✓
置信度:        -      0.88    0.75    0.50    0.92

选择: 位置 1
```

### 并行前缀 AR 模式

```
有 threshold=0.9 时:

位置:           0       1       2       3       4
状态:          [M]     [M]     [M]     [M]     [M]
前缀掩码:       ✓       ✓       ✓       ✓       ✓
置信度:       0.95    0.92    0.88    0.45    0.91

检查阈值:
位置 0: 0.95 >= 0.9 ✓
位置 1: 0.92 >= 0.9 ✓
位置 2: 0.88 < 0.9 ✗ (停止)

选择: 位置 0, 1
解码后:        [A]     [B]     [M]     [M]     [M]
```

## 使用示例

### 配置文件

```yaml
# configs/generation/ar.yaml
strategy: ar

threshold: null
alg: "maskgit_plus"
gen_length: null
block_length: null
num_transfer_tokens: 1
temperature: 0.0
top_p: null
top_k: null
stop_until_eos: false
debias: false
output_probs: false
```

### 命令行使用

```bash
# 标准 AR 解码
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=ar \
    generation.num_transfer_tokens=1 \
    generation.gen_length=256 \
    generation.block_length=32 \
    model=llada-inst

# 并行前缀 AR 解码
accelerate launch eval.py \
    generation=ar \
    generation.threshold=0.9 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.ar import ar_generate

# 标准 AR
result = ar_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    num_transfer_tokens=1,
    mask_token_id=tokenizer.mask_token_id,
)

# 并行前缀 AR
result = ar_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    threshold=0.9,  # 启用并行前缀解码
    mask_token_id=tokenizer.mask_token_id,
)
```

## 性能特点

### 优势

1. **一致性**：与传统自回归模型行为一致
2. **可预测性**：解码顺序固定，易于调试
3. **因果性**：严格保证因果依赖关系
4. **兼容性**：可与 KV Cache 完美配合

### 劣势

1. **速度较慢**：每步只能解码少量 token
2. **未利用双向性**：无法利用扩散模型的双向注意力优势
3. **灵活性低**：解码顺序固定

### 适用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 需要因果生成 | 无 threshold | 严格 AR |
| 平衡速度质量 | `threshold=0.9` | 并行前缀 AR |
| 配合 KV Cache | 无 threshold | 最佳实践 |
| 与 AR 模型对比 | 无 threshold | 公平比较 |

## 与其他策略的比较

| 策略 | 解码顺序 | 速度 | 因果性 | 双向性利用 |
|------|----------|------|--------|------------|
| Vanilla | 按置信度 | 慢 | 弱 | 高 |
| Parallel | 按置信度 | 快 | 弱 | 中 |
| AR | 从左到右 | 最慢 | 强 | 低 |

## 与 KV Cache 的协同

AR 策略与 KV Cache 配合效果最佳：

```python
from src.cache import D2Cache

result = ar_generate(
    model=model,
    input_ids=input_ids,
    cache_cls=D2Cache,  # 启用 KV Cache
)
```

原因：
- AR 的顺序解码特性使得 KV Cache 可以精确复用
- 不需要处理复杂的缓存失效问题
- 与传统 AR 模型的 KV Cache 行为一致

## 实现细节

### 前缀连续性保证

```python
# 确保解码范围是连续的前缀
after_start = positions >= start.unsqueeze(-1)
first_invalid = (~transfer_index_mask) & after_start
end = torch.where(
    first_invalid.any(dim=-1),
    first_invalid.int().argmax(dim=-1),
    torch.full((batch_size,), seq_len, device=device),
)
return after_start & (positions < end.unsqueeze(-1))
```

### 阈值检查的提前终止

```python
if threshold is not None:
    # 找到第一个低于阈值的位置
    fail_mask = valid_prefix_mask & (prefix_scores < threshold)
    selected_count = torch.where(
        fail_mask.any(dim=-1),
        fail_mask.int().argmax(dim=-1),  # 第一个失败位置
        allowed_count,  # 全部通过
    )
```

## 参考文献

- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) - LLaDA 论文
- [Fast-dLLM: Training-free Acceleration of Diffusion LLM](https://arxiv.org/abs/2505.22618) - 并行前缀解码
