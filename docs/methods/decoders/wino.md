# WINO 解码策略


## 算法逻辑精要

WINO采用"起草-验证"式的可撤销解码机制：先以较低的wide_in_thres阈值激进地并行解码多个候选token（Wide-In阶段），再以较高的narrow_out_thres阈值对已解码token进行回顾验证，将置信度不足的可疑token重新掩码回[MASK]（Narrow-Out阶段）。同时引入Shadow Block机制，在每个生成块后追加一个全掩码的shadow块用于leave-one-out交叉验证——每个shadow位置的预测可参考块中除自身外的所有其他已解码位置信息，从而更准确地判断原token的可靠性。

## 概述

WINO（Wide-In, Narrow-Out，宽进窄出）是一种可撤销的解码算法，专为扩散大语言模型设计。它采用"起草-验证"机制，激进地并行解码多个 token，同时利用模型的双向上下文能力验证并重新掩码可疑的 token，从而实现质量-速度权衡的突破。

## 算法原理

### 核心思想

WINO 的核心洞察是：

> **标准扩散模型解码是不可逆的——一旦 token 被解码就无法撤销，这容易导致错误上下文的累积。WINO 通过引入"撤销"机制，允许重新掩码可疑 token 进行修正。**

### Wide-In, Narrow-Out 机制

#### Wide-In（宽进）

- 使用较低的置信度阈值（如 0.6）
- 激进地解码更多 token
- 允许更多"候选" token 进入

#### Narrow-Out（窄出）

- 使用较高的置信度阈值（如 0.9）
- 验证已解码的 token
- 重新掩码低置信度的 token

### Shadow Block 机制

WINO 引入"影子块"（Shadow Block）来实现验证：

```
原始序列:  [Prompt] [Block 0] [Block 1] ...
                    ↓ Wide-In
解码后:    [Prompt] [已解码]  [Block 1] ...
                    ↓ 添加 Shadow Block
验证序列:  [Prompt] [已解码] [Block 1] [Shadow]
                                         ↓
                              Shadow 可以看到 Block 的上下文
                              用于验证 Block 中的 token
```

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    WINO 解码流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   单步解码过程                            │    │
│  │                                                         │    │
│  │  1. 准备输入（添加 Shadow Block）                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  input = [prompt + tokens + MASK*]    │           │    │
│  │     │  # MASK* 是 Shadow Block              │           │    │
│  │     │                                       │           │    │
│  │     │  示例:                                 │           │    │
│  │     │  [P] [A] [B]  [C]  [D]  [M] [M] [M]   │           │    │
│  │     │   ↑   ↑    ↑    ↑    ↑    ↑   ↑   ↑   │           │    │
│  │     │  prompt decoded  MASK  Shadow Block   │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  2. 构建注意力掩码                                       │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  特殊注意力模式:                        │           │    │
│  │     │  - Shadow 不被其他位置关注             │           │    │
│  │     │  - Shadow 关注 Block（leave-one-out） │           │    │
│  │     │                                       │           │    │
│  │     │  位置:    0   1   2   3   4   S1  S2  │           │    │
│  │     │  Block:   [A] [B] [C] [D]             │           │    │
│  │     │  Shadow:              [M] [M]         │           │    │
│  │     │                                       │           │    │
│  │     │  S1 可以看到 A,B,C,D (除了 D)         │           │    │
│  │     │  S2 可以看到 A,B,C,D (除了 D)         │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  3. 模型前向传播                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  outputs = model(input, attention_mask)│           │    │
│  │     │                                       │           │    │
│  │     │  # 获得两份 logits:                   │           │    │
│  │     │  # - Block logits: 原始位置           │           │    │
│  │     │  # - Shadow logits: 验证位置          │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  4. 合并 Logits                                          │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 对于 MASK 位置: 使用 Block logits  │           │    │
│  │     │  # 对于已解码位置: 使用 Shadow logits │           │    │
│  │     │  combined = where(is_mask, block, shadow)│        │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  5. Wide-In（宽进）                                      │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 使用较低阈值选择更多 token          │           │    │
│  │     │  wide_in_mask = confidence >= 0.6     │           │    │
│  │     │                                       │           │    │
│  │     │  示例: 解码 5 个 token                 │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  6. Narrow-Out（窄出）                                   │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 对已解码 token 进行验证             │           │    │
│  │     │  # 使用 Shadow logits 计算置信度       │           │    │
│  │     │  narrow_out_mask = shadow_conf < 0.9  │           │    │
│  │     │                                       │           │    │
│  │     │  # 重新掩码低置信度 token             │           │    │
│  │     │  remask(narrow_out_mask)              │           │    │
│  │     │                                       │           │    │
│  │     │  示例: 重新掩码 2 个 token             │           │    │
│  │     │  净解码: 5 - 2 = 3 个 token            │           │    │
│  │     └───────────────────────────────────────┘           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 验证机制详解

```
假设 Block 有 4 个位置，其中 2 个已解码，2 个是 MASK:

位置:        0       1       2       3
状态:      [A]     [B]     [M]     [M]
置信度:    0.85    0.70     -       -

Shadow Block 验证:
位置:       S0      S1
验证目标:   位置0   位置1

S0 看到 A,B,M,M（除了位置0的 A）
S0 对位置0的预测 = 0.95 → 验证通过

S1 看到 A,B,M,M（除了位置1的 B）
S1 对位置1的预测 = 0.60 → 验证失败，重新掩码

结果:
位置:        0       1       2       3
状态:      [A]     [M]     [C]     [D]
```

## 核心参数

### WINO 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wide_in_thres` | float | 0.6 | Wide-In 阈值，控制激进解码 |
| `narrow_out_thres` | float | 0.9 | Narrow-Out 阈值，控制验证严格程度 |

### 参数含义

| 参数 | 低值效果 | 高值效果 |
|------|----------|----------|
| `wide_in_thres` | 更激进，解码更多 token | 更保守，解码更少 token |
| `narrow_out_thres` | 宽松验证，保留更多 token | 严格验证，重新掩码更多 token |

### 继承参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列总长度 |
| `block_length` | int | - | 块大小 |
| `num_transfer_tokens` | int | 1 | 每步最少解码数量 |
| `sigma` | float | None | 确定性先验参数 |

## 详细代码流程分析

WINO 解码策略的核心实现在 `src/generation/wino.py` 中，包含两个主要函数：`wino_generate_step`（单步解码）和 `wino_generate`（入口函数）。以下按源码行号顺序，逐模块讲解。

### wino_generate_step: 输入准备

```python
# 源文件: src/generation/wino.py L41-L89
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = frame.prompts.device
    block_indices = torch.nonzero(block_mask[0], as_tuple=False).squeeze(-1)
    block_start = block_indices[0].item()
    block_end = block_indices[-1].item() + 1
    block_length = int(block_end - block_start)

    can_generate = check_can_generate(
        frame,
        eligible_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens,
        stop_until_eos=stop_until_eos,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )

    if not torch.any(can_generate):
        return None

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    # filtered inputs
    prompts_active = frame.prompts[can_generate]
    generated_active = frame.generated_tokens[can_generate]

    # append a block of mask_token_id to input ids
    x = F.pad(
        torch.cat([prompts_active, generated_active], dim=-1),
        (0, block_length),
        value=mask_token_id,
    )

    active_batch_size, total_len = x.shape
    active_attn_mask = attention_mask[can_generate]

    # prepare attention mask & position ids
    # see figure 2(b) of the original paper for details
    active_attn_mask_ext = F.pad(active_attn_mask, (0, block_length), value=1)

    prefix_pos_ids = (torch.cumsum(active_attn_mask, dim=1) - 1) * active_attn_mask
    position_ids = torch.zeros(
        (active_batch_size, total_len), device=device, dtype=torch.long
    )
    position_ids[:, :-block_length] = prefix_pos_ids
    position_ids[:, -block_length:] = prefix_pos_ids[
        :, prompt_length + block_start : prompt_length + block_end
    ]
```

**逐行解释：**

- L41：`frame = frame.as_batch()` 确保 Frame 是 batch 模式（即使 batch_size=1），统一后续处理逻辑。
- L42：获取 `batch_size` 和 `prompt_length`（prompt 区域的 token 数）。
- L43：获取当前设备（CPU 或 GPU）。
- L44-L47：通过 `torch.nonzero` 找到 `block_mask` 中所有为 True 的位置索引，取第一个 `block_start` 和最后一个 `block_end`，计算 `block_length`。`block_mask[0]` 是因为所有 batch 序列共享同一 block 布局。
- L49-L56：`check_can_generate` 检查当前 Frame 中是否还有足够的 [MASK] token 可解码。若还有至少 `num_transfer_tokens` 个未解码位置（或启用 `stop_until_eos` 时尚未遇到 EOS），则 `can_generate` 对应位置为 True。
- L58-L59：若没有任何序列可以继续生成，返回 `None`，通知调用方该块解码已完成。
- L61：`remaining_mask` 标记所有尚未解码的 [MASK] 位置（generated_tokens == mask_token_id）。
- L62：`transfer_index_mask = remaining_mask.clone()` 克隆得到可转移位置掩码。后续 Narrow-Out 会重新掩码一些位置，所以 transfer_index_mask 用于跟踪哪些位置可被选中解码。
- L65-L66：过滤出活跃序列的 prompts 和 generated_tokens。`can_generate` 张量筛选掉已完成的序列，减少后续计算量。
- L69-L73：构建扩展输入 `x`。将 prompts 和 generated_tokens 沿序列维度拼接，然后在右侧填充 `block_length` 个 `[MASK]` token 作为 Shadow Block。Shadow Block 是完全掩码的区域，用于 leave-one-out 交叉验证。
- L75：提取 `active_batch_size` 和 `total_len`（prompt + generated + shadow 总长度）。
- L76：从全局 `attention_mask` 中筛选出活跃序列对应的子集。
- L80：`active_attn_mask_ext = F.pad(active_attn_mask, (0, block_length), value=1)` 右侧填充 `block_length` 个 1，使 attention mask 覆盖 Shadow Block 位置。值为 1 表示该位置参与注意力计算。
- L82：计算前缀位置编码。`torch.cumsum(active_attn_mask, dim=1)` 沿序列维度累加，非填充位置递增序号，填充位置不变。`-1` 使序号从 0 开始，`* active_attn_mask` 将填充位置归零。
- L83-L85：初始化 `position_ids` 为全零张量，形状为 `(active_batch_size, total_len)`。
- L86：将 prompt + generated 区域的位置 ID 设为 `prefix_pos_ids`。
- L87-L89：将 Shadow Block 区域的位置 ID 设为与当前 Block 对应的前缀位置 ID 相同（`prompt_length + block_start : prompt_length + block_end` 范围）。这意味着 Shadow Block 中的每个位置复用 Block 中对应位置的位置编码，从而使 Shadow 预测与 Block 位置语义对齐。

### wino_generate_step: WINO 注意力掩码

```python
# 源文件: src/generation/wino.py L91-L108
    final_mask = (
        active_attn_mask_ext.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, 1, total_len, total_len)
        .bool()
        .clone()
    )

    # ----- apply wino mask constraints -----
    # nothing attends to shadow block (except shadow block itself)
    final_mask[:, :, :-block_length, -block_length:] = False

    # shadow block attends to current block with ~eye (leave-one-out)
    r_start = total_len - block_length
    c_start = prompt_length + block_start
    final_mask[:, :, r_start:, c_start : c_start + block_length] &= ~torch.eye(
        block_length, device=device, dtype=torch.bool
    )
```

**逐行解释：**

- L91-L97：构建初始注意力掩码 `final_mask`。从 `active_attn_mask_ext`（形状 `(B, total_len)`）开始，通过两次 `unsqueeze(1)` 扩展为 `(B, 1, 1, total_len)`，再 `expand` 为 `(B, 1, total_len, total_len)`。`.bool()` 转换为布尔类型，`.clone()` 创建独立副本以便后续修改。
- L100-L101：**约束 1**：将 `final_mask` 中所有非 Shadow Block 行（`:-block_length`）对 Shadow Block 列（`-block_length:`）的注意力设为 False。即 prompt 和 generated token 不能 attend 到 Shadow Block。Shadow Block 自身行仍可见自身列（因为只改了前 `total_len-block_length` 行）。
- L103-L108：**约束 2（Leave-One-Out）**：Shadow Block 行（`r_start:`）可以看到 Block 列（`c_start : c_start + block_length`），但要排除对角线——使用 `~torch.eye(block_length)` 构造对角线为 False 的单位矩阵取反，`&=` 将其与对应区域做逻辑与。这意味着 Shadow 位置 i 可以看到 Block 中除位置 i 之外的所有位置，实现了 leave-one-out 交叉验证。

### wino_generate_step: Shadow Block 前向

```python
# 源文件: src/generation/wino.py L110-L117
    # ----- forward with the shadow block -----
    outputs = model(
        x,
        attention_mask=final_mask,
        position_ids=position_ids,
        output_hidden_states=output_hidden_states,
        use_cache=False,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
```

**逐行解释：**

- L111-L117：使用包含 Shadow Block 的输入 `x` 和特殊注意力掩码 `final_mask` 执行一次完整前向传播。`use_cache=False` 是因为 WINO 每次都要处理完整的输入序列（含 Shadow Block 结构始终不同），无法复用 KV Cache。模型同时输出 Block 位置和 Shadow 位置的预测。
- L119：`prepare_logits_for_generation` 对原始 logits 进行必要的后处理（如根据模型类型调整维度、提取生成区域的 logits 等），得到可用于后续采样和置信度计算的 logits 张量。

### wino_generate_step: logits 组合

```python
# 源文件: src/generation/wino.py L121-L127
    # combine logits (B, block_length, vocab_size): main block + shadow block
    block_mask_curr = transfer_index_mask[can_generate][:, block_start:block_end]
    combined_logits = torch.where(
        block_mask_curr.unsqueeze(-1),
        logits[:, prompt_length + block_start : prompt_length + block_end],
        logits[:, -block_length:],
    ).to(torch.float64)
```

**逐行解释：**

- L122：`block_mask_curr` 从 `transfer_index_mask` 中提取当前块的活跃序列部分，形状为 `(active_batch_size, block_length)`。True 表示该位置仍是 [MASK]，False 表示已解码。
- L123-L127：合并 Block logits 和 Shadow logits。对于仍是 [MASK] 的位置（`block_mask_curr=True`），使用 Block 的原始 logits（`prompt_length + block_start : prompt_length + block_end`）；对于已解码的位置，使用 Shadow Block 的 logits（`-block_length:`）。Shadow logits 是在 leave-one-out 条件下预测的，更准确地反映 token 的可靠性。转换为 `float64` 精度以保证后续置信度计算的数值稳定性。

### wino_generate_step: sample_tokens

```python
# 源文件: src/generation/wino.py L135-L150
    combined_conf, x0, p = sample_tokens(
        combined_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        clip_alpha=clip_alpha,
        alg=alg,
    )

    # for masked positions, sample_tokens' confidence already matches x0; for unmasked, gather current tokens.
    current_tokens = generated_active[:, block_start:block_end]
    current_conf = torch.gather(p, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(
        -1
    )
    confidence = torch.where(block_mask_curr, combined_conf, current_conf)
```

**逐行解释：**

- L135-L143：`sample_tokens` 基于合并后的 logits 进行采样，返回三个值：`combined_conf`（每个位置的置信度分数）、`x0`（预测的具体 token）、`p`（完整的概率分布）。采样过程受温度 `temperature`、top-k/top-p 参数控制，`alg` 决定置信度计算算法（如 maskgit_plus）。
- L145-L149：对于 [MASK] 位置，`combined_conf` 已经正确反映了 x0 预测的置信度。对于已解码位置，需要通过 `torch.gather` 从概率分布 `p` 中收集当前已解码 token 对应的概率值，作为其置信度。
- L150：`confidence = torch.where(block_mask_curr, combined_conf, current_conf)`。对 [MASK] 位置使用采样置信度，对已解码位置使用当前 token 的概率值。这个置信度矩阵将在 Wide-In 和 Narrow-Out 阶段中使用。

### wino_generate_step: Wide-In 解码（宽进）

```python
# 源文件: src/generation/wino.py L152-L188
    # ----- unmasking (wide in) -----
    scores = torch.where(block_mask_curr, confidence, -torch.inf)
    if sigma is not None and sigma > 0:
        scores = (
            confidence
            * certainty_density(~remaining_mask[can_generate], sigma=sigma)[
                :, block_start:block_end
            ]
        )
    selected_indices = confidence_unmasking(
        scores=scores,
        transfer_index_mask=block_mask_curr,
        min_transfer_tokens=num_transfer_tokens,
        max_transfer_tokens=torch.maximum(
            torch.clamp(
                (block_mask_curr.sum(dim=1) * 0.7).int(),
                min=5,
                max=20,  # adopted from the official implementation
            ),
            torch.full(
                (active_batch_size,),
                num_transfer_tokens,
                device=scores.device,
                dtype=torch.long,
            ),
        ),
        threshold=wide_in_thres,
    )

    unmask_mask_block = torch.zeros_like(block_mask_curr, dtype=torch.bool)
    for i, idx in enumerate(selected_indices):
        if idx.numel() > 0:
            unmask_mask_block[i, idx.long()] = True

    unmask_mask = torch.zeros_like(transfer_index_mask[can_generate], dtype=torch.bool)
    unmask_mask[:, block_start:block_end] = unmask_mask_block
```

**逐行解释：**

- L153：`scores = torch.where(block_mask_curr, confidence, -torch.inf)`。对于 [MASK] 位置使用置信度作为 score，对于已解码位置设为 `-inf`，确保它们不会被再次选中解码。
- L154-L160：若提供了 `sigma` 参数（`sigma > 0`），则将置信度乘以 `certainty_density` 密度函数进行重加权。这允许根据解码确定性先验调整 score，使模型偏好解码确定性较高的位置。
- L161-L179：调用 `confidence_unmasking` 进行 Wide-In 选择。关键参数：
  - `scores` 和 `transfer_index_mask=block_mask_curr`：在块内 [MASK] 位置中选择。
  - `min_transfer_tokens=num_transfer_tokens`：至少解码指定数量。
  - `max_transfer_tokens`：使用 `torch.maximum` 取两个值中的较大者——`torch.clamp((remaining_mask_count * 0.7).int(), min=5, max=20)` 限制最大解码数量为剩余 [MASK] 数量的 70%（上下限为 5-20），以及 `num_transfer_tokens` 的保底值。这防止单步解码过于激进。
  - `threshold=wide_in_thres`：Wide-In 的宽松阈值（默认 0.6），使得置信度较低但仍有潜力的 token 也能被选入。
- L181-L184：将 `selected_indices`（每序列选中位置的索引列表）转换为布尔掩码 `unmask_mask_block`。对于每个活跃序列，使用 `scatter` 风格的操作将选中位置置为 True。
- L186-L187：将块级掩码 `unmask_mask_block` 填充到全局大小 `unmask_mask` 中（形状与 `transfer_index_mask[can_generate]` 相同）。

### wino_generate_step: Narrow-Out 重掩码（窄出）

```python
# 源文件: src/generation/wino.py L189-L222
    # ----- remasking (narrow out) -----
    remask_mask = torch.zeros_like(unmask_mask)
    current_wide_in = unmask_mask_block.sum(dim=1)
    num_last_wide_in[can_generate] = current_wide_in

    # only consider samples that have unmasked at least one token
    can_remask = current_wide_in > 0

    # use shadow block probs for already-unmasked tokens
    shadow_conf = torch.where(~block_mask_curr, confidence, torch.inf)

    for i in range(active_batch_size):
        if not can_remask[i]:
            continue

        row_conf = shadow_conf[i]
        row_remask = row_conf < narrow_out_thres

        target_k = max(int(current_wide_in[i].item()) - num_transfer_tokens, 0)

        if row_remask.sum() > target_k:
            if target_k <= 0:
                row_remask[:] = False
            else:
                # select bottom-k to remask, only when k is valid
                _, idx = torch.topk(row_conf.view(-1), k=target_k, largest=False)
                row_remask = (
                    torch.zeros_like(row_remask)
                    .view(-1)
                    .scatter_(0, idx, True)
                    .view_as(row_remask)
                )

        remask_mask[i, block_start:block_end] = row_remask
```

**逐行解释：**

- L190：初始化 `remask_mask`，全零张量，形状与 `unmask_mask` 相同。
- L191：`current_wide_in = unmask_mask_block.sum(dim=1)` 统计每个序列在 Wide-In 阶段解码了多少 token。
- L192：`num_last_wide_in[can_generate] = current_wide_in` 将当前块的 Wide-In 数量记录到跟踪张量中。`num_last_wide_in` 在 `wino_generate` 中被用于跨块传递 Wide-In 计数信息（影响后续块的解码策略）。
- L195：`can_remask = current_wide_in > 0` 只有在 Wide-In 阶段至少解码了一个 token 的序列才需要进行 Narrow-Out 验证。
- L198：`shadow_conf` 基于 Shadow Block 预测的置信度。对于已解码位置（`~block_mask_curr=True`），使用 Shadow 置信度；对于 [MASK] 位置，填入 `torch.inf` 确保它们不会被误认为需要重新掩码。
- L200-L202：遍历每个活跃序列，跳过 `can_remask[i]` 为 False 的序列。
- L204：`row_conf = shadow_conf[i]` 取第 i 个序列的 Shadow 置信度行。
- L205：`row_remask = row_conf < narrow_out_thres` 标记置信度低于窄出阈值的已解码位置。`narrow_out_thres` 默认 0.9，高于 `wide_in_thres`（0.6），体现"宽进窄出"原则——进时宽松，出时严格。
- L207：计算允许重新掩码的最大数量 `target_k`。公式为 `max(wide_in_count - num_transfer_tokens, 0)`，确保净解码数量至少为 `num_transfer_tokens`。例如 Wide-In 解码了 5 个、`num_transfer_tokens=1`，则最多允许重新掩码 4 个，保证至少保留 1 个。
- L209-L220：若待重新掩码的数量（`row_remask.sum()`）超过 `target_k`，需要进行裁剪：
  - L210-L211：若 `target_k <= 0`，说明不允许任何重新掩码，将 `row_remask` 全部置为 False。
  - L213-L220：否则，使用 `torch.topk(..., largest=False)` 选择置信度最低的 `target_k` 个已解码位置进行重新掩码（bottom-k 策略）。通过 `scatter_` 构建精确的重新掩码张量。
- L222：将逐序列的 `row_remask` 结果写入全局 `remask_mask` 的对应 block 范围。

### wino_generate_step: FrameDelta 构建

```python
# 源文件: src/generation/wino.py L224-L281
    # construct delta
    decoded_tokens = torch.full_like(generated_active, INVALID_TOKEN_ID)
    decoded_tokens[:, block_start:block_end].masked_scatter_(
        block_mask_curr, x0[block_mask_curr]
    )
    decoded_tokens[remask_mask] = mask_token_id

    total_mask = (
        unmask_mask | remask_mask
    )  # we need to transfer remasking tokens as well
    active_transfer_index = tuple(
        torch.nonzero(total_mask[i], as_tuple=False).squeeze(-1)
        for i in range(active_batch_size)
    )

    transfer_index_iter = iter(active_transfer_index)
    transfer_index = tuple(
        (
            next(transfer_index_iter)
            if is_active
            else torch.tensor([], dtype=torch.long, device=device)
        )
        for is_active in can_generate
    )

    confidence_ext = torch.full_like(
        generated_active, -torch.inf, dtype=confidence.dtype
    )
    confidence_ext[:, block_start:block_end].masked_scatter_(
        block_mask_curr, confidence[block_mask_curr]
    )
    confidence_ext[remask_mask] = 1.0

    probs_ext = None
    if output_probs:
        probs_ext = torch.full(
            (*generated_active.shape, p.size(-1)),
            -torch.inf,
            device=device,
            dtype=p.dtype,
        )
        probs_ext[:, block_start:block_end] = torch.where(
            block_mask_curr.unsqueeze(-1), p, -torch.inf
        )
        dummy_probs = torch.zeros((p.size(-1),), device=device, dtype=p.dtype)
        dummy_probs[mask_token_id] = 1.0
        probs_ext[remask_mask] = dummy_probs.unsqueeze(0)

    return FrameDelta(
        transfer_index=transfer_index,
        decoded_tokens=decoded_tokens,
        confidence=confidence_ext,
        probs=probs_ext,
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
        extra=dict(num_last_wide_in=num_last_wide_in),
    ).to(model.dtype)
```

**逐行解释：**

- L225：`decoded_tokens` 初始化为全 `INVALID_TOKEN_ID` 的张量，形状与 `generated_active` 相同。`INVALID_TOKEN_ID` 是一个占位标记，表示该位置无需解码。
- L226-L228：对 block 范围内的 [MASK] 位置（`block_mask_curr=True`），使用 `masked_scatter_` 将 x0 中对应位置的预测 token 填入 `decoded_tokens`。
- L229：`decoded_tokens[remask_mask] = mask_token_id` 将 Narrow-Out 标记为需要重新掩码的位置设回 `[MASK]` token，实现"撤销"效果。这些位置将在后续步中重新预测。
- L231-L232：`total_mask = unmask_mask | remask_mask` 将 Wide-In 选中的位置和 Narrow-Out 重新掩码的位置合并。重新掩码的位置也需要出现在 `transfer_index` 中，因为它们的状态发生了变化（token → MASK）。
- L233-L236：`active_transfer_index` 对每个活跃序列收集需转移的位置索引。
- L238-L246：构建完整的 `transfer_index` 元组。对于 `can_generate=True` 的序列，填入 `active_transfer_index` 中对应的索引；对于已完成的序列，填入空张量。这样 `FrameDelta` 可以整体应用于所有 batch 序列。
- L249-L251：初始化 `confidence_ext` 为全 `-inf` 的张量，形状与 `generated_active` 相同。
- L252-L254：对 block 范围内的 [MASK] 位置填入置信度值。
- L255：`confidence_ext[remask_mask] = 1.0` 将重新掩码位置的置信度设为 1.0。这确保 `Frame.apply_delta` 将这些位置正确地处理为需要重新解码的 [MASK]。
- L257-L270：仅在 `output_probs=True` 时构建 `probs_ext`。逻辑与 confidence 类似：block 范围内的 [MASK] 位置填入概率分布，重新掩码位置填入一个虚拟的 one-hot 分布（`[MASK]` token 的概率为 1.0）。
- L272-L281：返回 `FrameDelta` 对象，包含 `transfer_index`、`decoded_tokens`、`confidence_ext`、`probs_ext` 以及 `intermediate`（可选隐藏状态）。`.to(model.dtype)` 将 delta 转换为模型精度。`extra` 字典中传递 `num_last_wide_in`，供 `wino_generate` 在下一步使用。

### wino_generate: 入口参数处理和 block 循环

```python
# 源文件: src/generation/wino.py L284-L395
@register("wino")
def wino_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "maskgit_plus",
    block_length: int = 32,
    gen_length: int = 128,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    sigma: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    # wino
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> DecodeRecord:
    """
    Wino decoding strategy.
    """
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID", -1))
    pad_token_id = pad_token_id or int(os.environ.get("PAD_TOKEN_ID", -1))

    if -1 in [mask_token_id, pad_token_id]:
        raise ValueError(
            "mask_token_id and pad_token_id must be provided either as arguments or environment variables."
        )
    if stop_until_eos:
        if eos_token_id is None and os.environ.get("EOS_TOKEN_ID", None) is None:
            raise ValueError(
                "eos_token_id must be provided either as an argument or an environment variable if stop_until_eos is set to True."
            )
        eos_token_id = eos_token_id or int(os.environ.get("EOS_TOKEN_ID"))  # type: ignore

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if num_transfer_tokens <= 0:
        raise ValueError(f"{num_transfer_tokens=} must be > 0")

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    frame = initial_frame
    deltas = []

    for block_idx in range(num_blocks):
        num_last_wide_in = torch.full(
            (input_ids.size(0),), 30, device=model.device, dtype=torch.long
        )
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        while True:
            delta = wino_generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                attention_mask=attention_mask,
                num_transfer_tokens=num_transfer_tokens,
                alg=alg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                sigma=sigma,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                mask_token_id=mask_token_id,
                wide_in_thres=wide_in_thres,
                narrow_out_thres=narrow_out_thres,
                num_last_wide_in=num_last_wide_in,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )

            if delta is None:
                break

            # update num_last_wide_in based on Wide In count
            num_last_wide_in = delta.extra.pop("num_last_wide_in")

            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
```

**逐行解释：**

- L284：`@register("wino")` 将函数注册为名为 `"wino"` 的解码策略。
- L285-L306：函数签名和 docstring。WINO 特有参数 `wide_in_thres`（默认 0.6）和 `narrow_out_thres`（默认 0.9）。注意 WINO 不支持 `cache_cls` 参数，因为 Shadow Block 机制导致每次前向的输入结构变化，KV Cache 无法复用。
- L310-L322：参数校验。`mask_token_id` 和 `pad_token_id` 可从参数或环境变量获取；若 `stop_until_eos=True` 则需提供 `eos_token_id`。
- L324-L327：断言 `gen_length` 能被 `block_length` 整除，计算总块数；验证 `num_transfer_tokens > 0`。
- L329-L333：创建初始 Frame，所有生成位置填充为 `[MASK]` token。
- L335-L341：构建 attention_mask。若未传入则基于 `pad_token_id` 自动构建；若仅覆盖 prompt 区域则右侧填充 `gen_length` 个 1。
- L343-L344：`frame = initial_frame`，初始化 `deltas = []`。
- L346：开始 block 循环，`block_idx` 从 0 到 `num_blocks - 1`。
- L347-L349：`num_last_wide_in` 初始化为全 30。这是一个跨块传递的状态张量，`wino_generate_step` 会在每步解码后将本块的 Wide-In 数量记录到此张量中。
- L350-L358：创建 `block_mask`，标记当前块的生成位置范围为 True。
- L360：进入内层 while 循环，反复调用 `wino_generate_step`。
- L361-L380：调用 `wino_generate_step`，传入所有参数。注意 `num_last_wide_in` 每次调用都会被原地更新。
- L382-L383：若 `delta is None`，表示当前块内无可解码 token，break 跳出 while 循环进入下一块。
- L386：从 `delta.extra` 中弹出 `num_last_wide_in`，更新跨块状态。
- L388：将 delta 转移到 CPU 并追加到 `deltas`。
- L389：`frame = frame.apply_delta(delta)` 应用 delta，更新生成的 token。
- L391-L395：所有块处理完毕后，构建并返回 `DecodeRecord`，包含初始 Frame、所有 delta 序列和 block_length。

## Token 选择策略

### 双阶段选择

```
阶段 1: Wide-In（起草）
┌─────────────────────────────────────────┐
│  条件: confidence >= wide_in_thres      │
│  目标: 尽可能多地解码 token             │
│  结果: 候选 token 集合                  │
└─────────────────────────────────────────┘
           │
           ▼
阶段 2: Narrow-Out（验证）
┌─────────────────────────────────────────┐
│  条件: shadow_conf < narrow_out_thres   │
│  目标: 移除可疑 token                   │
│  结果: 最终解码 token 集合              │
└─────────────────────────────────────────┘
```

### 选择示例

```
配置: wide_in_thres=0.6, narrow_out_thres=0.9

位置:           0       1       2       3       4
原始状态:      [M]     [M]     [M]     [M]     [M]
Block 置信度:  0.95    0.75    0.65    0.45    0.30

Wide-In (thres=0.6):
  选择: 位置 0, 1, 2 (置信度 >= 0.6)
  解码后: [A]     [B]     [C]     [M]     [M]

Shadow 验证:
  位置 0: shadow_conf = 0.92 >= 0.9 ✓ 保留
  位置 1: shadow_conf = 0.85 < 0.9 ✗ 重新掩码
  位置 2: shadow_conf = 0.88 < 0.9 ✗ 重新掩码

Narrow-Out (thres=0.9):
  重新掩码: 位置 1, 2
  最终:    [A]     [M]     [M]     [M]     [M]

净解码: 1 个 token
```

## 使用示例

### 配置文件

```yaml
# configs/generation/wino.yaml
defaults:
  - vanilla
  - _self_

strategy: "wino"
wide_in_thres: 0.7
narrow_out_thres: 0.9
```

### 命令行使用

```bash
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=wino \
    generation.block_length=128 \
    generation.wide_in_thres=0.6 \
    generation.narrow_out_thres=0.9 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.wino import wino_generate

result = wino_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=128,
    wide_in_thres=0.6,      # Wide-In 阈值
    narrow_out_thres=0.9,   # Narrow-Out 阈值
    mask_token_id=tokenizer.mask_token_id,
)
```

### 参数调优示例

```python
# 激进模式（追求速度）
result = wino_generate(
    model, input_ids,
    wide_in_thres=0.5,      # 更低阈值
    narrow_out_thres=0.85,  # 更宽松验证
)

# 保守模式（追求质量）
result = wino_generate(
    model, input_ids,
    wide_in_thres=0.8,      # 更高阈值
    narrow_out_thres=0.95,  # 更严格验证
)
```

## 性能特点

### 优势

1. **可撤销解码**：允许修正错误决策
2. **质量-速度突破**：同时提升速度和质量
3. **利用双向上下文**：充分发挥扩散模型优势
4. **自适应**：根据验证结果动态调整

### 劣势

1. **计算开销**：Shadow Block 增加计算量
2. **内存占用**：需要额外的注意力掩码
3. **参数敏感**：两个阈值需要协调调整

### 性能数据

```
基准测试 (LLaDA-7B-Instruct):

任务          Vanilla    WINO      加速比    质量变化
GSM8K         42.3%      43.4%     6.0x      +1.1%
Flickr30K     28.5%      29.2%     10.0x     +0.7%
```

### 适用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 数学推理 | `wide_in=0.6`, `narrow_out=0.9` | 默认配置 |
| 代码生成 | `wide_in=0.7`, `narrow_out=0.95` | 更严格验证 |
| 文本摘要 | `wide_in=0.5`, `narrow_out=0.85` | 更激进 |
| 图像描述 | `wide_in=0.6`, `narrow_out=0.9` | 平衡配置 |

## 实现细节

### Leave-One-Out 注意力

```python
# Shadow 位置 i 只能看到 Block 中除了位置 i 的所有 token
final_mask[:, :, r_start:, c_start : c_start + block_length] &= ~torch.eye(
    block_length, device=device, dtype=torch.bool
)
```

这确保 Shadow 预测不受对应位置已解码 token 的影响。

### 最大解码数量限制

```python
max_transfer_tokens = torch.maximum(
    torch.clamp(
        (block_mask_curr.sum(dim=1) * 0.7).int(),
        min=5,
        max=20,
    ),
    ...
)
```

限制每步最多解码的 token 数量，防止过度激进。

### 重新掩码数量控制

```python
target_k = max(int(current_wide_in[i].item()) - num_transfer_tokens, 0)
```

确保净解码数量至少为 `num_transfer_tokens`。

## 与其他策略的比较

| 策略 | 可撤销 | 验证机制 | 速度 | 质量 |
|------|--------|----------|------|------|
| Vanilla | ✗ | ✗ | 慢 | 高 |
| Parallel | ✗ | ✗ | 快 | 中 |
| WINO | ✓ | ✓ | 快 | 高 |

## 参考文献

- [Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](https://arxiv.org/abs/2505.22618) - WINO 论文
