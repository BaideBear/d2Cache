# KLASS 解码策略


## 算法逻辑精要

KLASS通过追踪token预测分布的历史稳定性实现安全的多token并行解码：维护最近kl_history_length步的KL散度历史记录，每步计算当前概率分布与上一步分布的KL散度；若某位置连续kl_history_length步的KL散度均低于kl_threshold，则判定该位置的预测已趋于稳定，允许将其直接解码为确定token。对于尚未稳定的位置，则回退到标准的top-k置信度选择策略，确保解码可靠性不受影响。

## 概述

KLASS（KL-Adaptive Stability Sampling，KL 自适应稳定性采样）是一种基于 KL 散度的快速采样方法。它通过追踪 token 预测分布的稳定性，识别出稳定的高置信度预测，从而在每次迭代中安全地解码多个 token，实现显著的加速效果。

## 算法原理

### 核心思想

KLASS 的核心洞察是：

> **如果一个 token 的预测分布在多次迭代中保持稳定（KL 散度小），说明该预测是可靠的，可以安全地解码。**

### KL 散度稳定性检测

使用 KL 散度衡量当前预测分布与历史预测分布的差异：

```
KL(P_current || P_previous) = Σ P_current(x) * log(P_current(x) / P_previous(x))
```

- KL 散度小 → 分布稳定 → 预测可靠
- KL 散度大 → 分布变化大 → 预测不稳定

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    KLASS 解码流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  初始化:                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  kl_history = zeros(batch, gen_length, history_length)  │    │
│  │  prev_probs = zeros(batch, gen_length, vocab_size)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  每步解码:                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                         │    │
│  │  1. 模型前向传播                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  logits = model(input)                │           │    │
│  │     │  probs = softmax(logits)              │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  2. 计算 KL 散度                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  kl = probs * log(probs / prev_probs) │           │    │
│  │     │      .sum(dim=-1)  # 对词表求和        │           │    │
│  │     │                                       │           │    │
│  │     │  # 示例: kl = [0.001, 0.5, 0.002, ...] │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  3. 更新 KL 历史                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  kl_history = roll(kl_history, -1)    │           │    │
│  │     │  kl_history[..., -1] = kl             │           │    │
│  │     │                                       │           │    │
│  │     │  # 维护最近 kl_history_length 个 KL 值 │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  4. 判断稳定性                                           │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  stable_mask = all(kl_history <       │           │    │
│  │     │                    kl_threshold)      │           │    │
│  │     │                                       │           │    │
│  │     │  # 示例: [True, False, True, ...]     │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  5. 选择解码位置                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  if 稳定位置存在:                      │           │    │
│  │     │    选择稳定且高置信度的位置            │           │    │
│  │     │  else:                                │           │    │
│  │     │    回退到 top-k 置信度选择            │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  6. 更新历史概率                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  prev_probs = probs                   │           │    │
│  │     └───────────────────────────────────────┘           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 稳定性判断示例

```
假设 kl_threshold = 0.01, kl_history_length = 2

位置:           0       1       2       3
当前 KL:      0.005   0.500   0.003   0.100
历史 KL:      0.008   0.300   0.004   0.050

KL 历史:
位置 0: [0.008, 0.005] → max = 0.008 < 0.01 ✓ 稳定
位置 1: [0.300, 0.500] → max = 0.500 > 0.01 ✗ 不稳定
位置 2: [0.004, 0.003] → max = 0.004 < 0.01 ✓ 稳定
位置 3: [0.050, 0.100] → max = 0.100 > 0.01 ✗ 不稳定

稳定位置: [0, 2]
```

## 核心参数

### KLASS 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kl_threshold` | float | 0.01 | KL 散度阈值，用于判断预测稳定性 |
| `kl_history_length` | int | 2 | 考虑的历史 KL 值数量 |

### 继承参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列总长度 |
| `block_length` | int | - | 块大小 |
| `num_transfer_tokens` | int | 1 | 每步最少解码数量 |
| `threshold` | float | None | 可选的置信度阈值 |
| `factor` | float | None | 可选的速度因子 |

## 详细代码流程分析

KLASS 解码策略的核心实现在 `src/generation/klass.py` 中，入口函数为 `klass_generate`。以下按源码行号顺序，逐模块讲解每个函数和代码块的实现细节。

### 函数签名和参数处理

```python
# 源文件: src/generation/klass.py L16-L62
@register("klass")
def klass_generate(
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
    # klass
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
    # parallel decoding
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    KLASS generation strategy: KL-Adaptive Stability Sampling.
    """

    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))  # type: ignore
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
```

**逐行解释：**

- L16：`@register("klass")` 将函数注册为名为 `"klass"` 的解码策略，使得 Hydra 配置系统可通过 `strategy: klass` 自动发现并调用该函数。
- L17-L42：函数签名接受模型实例 `model`、输入 token ids `input_ids` 以及一系列控制生成行为的参数。KLASS 特有的参数包括 `kl_threshold`（KL 散度稳定性阈值，默认 0.01）和 `kl_history_length`（历史长度，默认 2）。`threshold` 和 `factor` 用于并行解码的可选置信度控制。
- L43-L45：docstring 描述该函数为「KLASS: KL-Adaptive Stability Sampling」生成策略。
- L47-L51：检查 `mask_token_id` 是否已通过参数或环境变量 `MASK_TOKEN_ID` 提供，若均未提供则抛出 `ValueError`；否则优先使用参数值，回退到环境变量。
- L52-L57：若 `stop_until_eos` 为 True，则同样检查 `eos_token_id` 是否可用，否则抛出 `ValueError`。
- L59：断言 `gen_length` 能被 `block_length` 整除，确保生成序列可以均匀划分为完整的块。
- L60：计算总块数 `num_blocks = gen_length // block_length`。
- L61-L62：验证 `num_transfer_tokens` 必须大于 0，否则抛出 `ValueError`。`num_transfer_tokens` 保证每步至少解码的 token 数量。

### Frame 初始化与状态准备

```python
# 源文件: src/generation/klass.py L64-L92
    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame
    batch_size, gen_length = frame.generated_tokens.shape
    deltas = []
    kl_history = torch.zeros(
        (batch_size, gen_length, kl_history_length),
        dtype=torch.float64,
        device=model.device,
    )
    prev_probs = torch.zeros(
        (batch_size, gen_length, model.config.vocab_size),
        dtype=torch.float64,
        device=model.device,
    )
```

**逐行解释：**

- L64-L68：调用 `Frame.create_initial_frame` 创建初始 Frame。该 Frame 包含 prompt tokens 和 `gen_length` 个 `[MASK]` token 组成的生成区域。随后将 Frame 转移到模型的设备和数据类型上。
- L70-L71：若未显式传入 `attention_mask` 但提供了 `pad_token_id`，则根据 `input_ids != pad_token_id` 自动构建 attention mask（非填充位置为 1，填充位置为 0）。
- L73-L76：若 `attention_mask` 存在且形状与 `input_ids` 相同（即仅覆盖 prompt 区域），则在其右侧填充 `gen_length` 个 1，使其覆盖整个 prompt + 生成区域，并转移到模型设备。
- L78：若传入了 `cache_cls`，则用模型配置实例化一个 Cache 对象（如 dCache 或 D2Cache），用于 KV Cache 加速；否则设为 `None`。
- L80：将 `frame` 指向 `initial_frame`，作为后续迭代的当前 Frame 基础。
- L81：从 `frame.generated_tokens` 中提取 `batch_size` 和 `gen_length`，用于初始化跟踪数据结构。
- L82：初始化 `deltas` 列表，用于累积每一步的 `FrameDelta`，最终构建 `DecodeRecord`。
- L83-L87：初始化 `kl_history` 张量，形状为 `(batch_size, gen_length, kl_history_length)`。每个位置维护最近 `kl_history_length` 步的 KL 散度值，使用 `float64` 精度以保证数值稳定性。
- L88-L92：初始化 `prev_probs` 张量，形状为 `(batch_size, gen_length, vocab_size)`。存储上一步的完整概率分布，用于计算当前步与历史步之间的 KL 散度。同样使用 `float64` 精度。全零初始化意味着第一步计算 KL 时 `prev_probs` 各项为 0（加 eps 后为极小值），KL 值会较大，这与直觉一致——初始状态下预测尚不稳定。

### unmasking_fn：KL 历史追踪

```python
# 源文件: src/generation/klass.py L94-L145
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

        eps = 1e-12
        kl_current_prev = (
            probs
            * (torch.log(probs + eps) - torch.log(prev_probs[active_seq_idx] + eps))
        ).sum(dim=-1)

        kl_history[active_seq_idx] = kl_history[active_seq_idx].roll(shifts=-1, dims=-1)
        kl_history[active_seq_idx, ..., -1] = kl_current_prev

        stable_mask = torch.all(kl_history[active_seq_idx] < kl_threshold, dim=-1)
        stable_transfer_mask = active_transfer_mask & stable_mask

        stable_transfer_index = confidence_unmasking(
            scores=scores,
            transfer_index_mask=stable_transfer_mask,
            min_transfer_tokens=0,
            threshold=threshold,
            factor=factor,
        )

        fallback_transfer_index = confidence_unmasking(
            scores=scores,
            transfer_index_mask=active_transfer_mask,
            min_transfer_tokens=num_transfer_tokens,
            threshold=None,
            factor=None,
        )
        transfer_index = tuple(
            stable_idx if stable_idx.numel() > 0 else fallback_idx
            for stable_idx, fallback_idx in zip(
                stable_transfer_index, fallback_transfer_index
            )
        )

        return (
            transfer_index,
            {"curr_probs": probs, "active_index": active_seq_idx},
        )
```

**逐行解释：**

- L94-L102：定义内部函数 `unmasking_fn`，使用仅关键字参数（`*` 后的参数必须按名称传递）。该函数将作为 `unmasking_fn` 参数传递给 `generate_step`，在每步解码时被调用来决定哪些位置应该被解码。`active_seq_idx` 是当前活跃（可生成）序列的索引；`scores` 是当前步的置信度分数；`probs` 是当前步的完整概率分布；`transfer_index_mask` 标记可转移的位置；`block_mask` 标记当前块的范围；`num_transfer_tokens` 是最少解码数量。
- L103：`active_transfer_mask = transfer_index_mask & block_mask` 将可转移位置限制在当前块范围内，得到当前步实际可解码的位置掩码。只有处于当前块内且尚未被解码的 [MASK] 位置才可被选中。
- L105：定义极小值 `eps = 1e-12`，防止后续 `log` 计算中出现 `log(0)` 导致的数值错误。
- L106-L109：计算当前概率分布与上一步概率分布之间的 KL 散度。公式为 `KL(P_current || P_prev) = Σ P_current * log(P_current / P_prev)`。通过 `probs * (log(probs+eps) - log(prev_probs+eps))` 实现逐元素计算，最后在词表维度（dim=-1）求和得到每个位置的 KL 值。`prev_probs[active_seq_idx]` 索引了对应活跃序列的历史概率，确保只对当前活跃的位置计算。
- L112：`kl_history[active_seq_idx] = kl_history[active_seq_idx].roll(shifts=-1, dims=-1)` 将活跃序列的 KL 历史沿最后一维向左滚动一位（`shifts=-1`），实现 FIFO 队列效果——最旧的 KL 值被移出队列，为最新值腾出空间。
- L113：将当前步计算得到的 `kl_current_prev` 写入 KL 历史队列的最后一个位置（`[..., -1]`），完成历史更新。
- L115：`torch.all(kl_history[active_seq_idx] < kl_threshold, dim=-1)` 检查每个位置的所有历史 KL 值是否都严格小于阈值 `kl_threshold`。若全部满足，则该位置被判定为"稳定"（`stable_mask` 对应位置为 True）。注意这里使用的是严格小于（`<`），即 KL 值恰好等于阈值时也被认为不稳定。
- L116：`stable_transfer_mask = active_transfer_mask & stable_mask` 将稳定掩码与当前块的可解码位置掩码取交集，得到既处于当前块内又处于稳定状态的位置掩码。
- L118-L125：**阶段 1（稳定性优先）**。调用 `confidence_unmasking`，使用 `stable_transfer_mask` 作为候选位置掩码，`min_transfer_tokens=0` 表示此阶段不强制解码任何 token——如果没有足够置信度的稳定位置，可以不选。可选地传入 `threshold` 和 `factor` 进一步过滤置信度不足的位置。
- L127-L134：**阶段 2（保底 fallback）**。再次调用 `confidence_unmasking`，但使用原始的 `active_transfer_mask` 作为候选位置，`min_transfer_tokens=num_transfer_tokens` 确保至少解码指定数量的 token。此阶段显式传入 `threshold=None, factor=None`，退化为纯 top-k 置信度选择策略，保证即使没有稳定位置也能推进解码。
- L135-L140：合并两个阶段的结果。对于 `zip(stable_transfer_index, fallback_transfer_index)` 中的每个序列，如果阶段 1 找到了稳定位置（`stable_idx.numel() > 0`），则优先使用稳定位置索引；否则回退到阶段 2 的保底结果。这保证了 KLASS 的"自适应"特性——有稳定位置时加速，没有时安全回退。
- L142-L145：返回一个元组。第一个元素 `transfer_index` 是各序列待解码的位置索引元组，将传递给 `generate_step` 用于更新 Frame。第二个元素是一个字典，保存了 `curr_probs`（当前概率分布）和 `active_index`（活跃序列索引），供外部在 `generate_step` 返回后用于更新 `prev_probs`。

### block 循环主流程

```python
# 源文件: src/generation/klass.py L147-L199
    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (batch_size, gen_length),
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
                sigma=sigma,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
                output_hidden_states=output_hidden_states,
                output_probs=output_probs,
            )
            if delta is None:
                break

            prev_probs[delta.extra.pop("active_index")] = delta.extra.pop("curr_probs")
            delta = delta.to(dtype=model.dtype)
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)
```

**逐行解释：**

- L147：遍历所有块，`block_idx` 从 0 到 `num_blocks - 1`。每个块内的 [MASK] 位置按 KLASS 稳定性策略逐个解码。
- L148-L152：为当前块创建 `block_mask`，初始化为全 False 的 `(batch_size, gen_length)` 布尔张量，放置在模型设备上。
- L153-L156：将 `block_mask` 中对应当前块的列范围 `[block_idx * block_length, (block_idx + 1) * block_length)` 设置为 True。这个掩码将在 `unmasking_fn` 中与 `transfer_index_mask` 做逻辑与，以限制解码范围。
- L158：保存当前块开始时的 Frame 快照 `start_frame`。这个快照供 Cache 记录块边界使用，以便在需要时回溯或重新计算。
- L159-L160：若启用了 Cache（如 dCache 或 D2Cache），调用 `cache.on_block_start(block_mask, frame)` 通知缓存系统新块开始，可进行必要的预处理（如为 KV Cache 预留存储空间）。
- L161：初始化 `block_deltas` 列表，用于累积当前块内每一步的 `FrameDelta`。与全局 `deltas` 分离是因为 Cache 需要块级别的 delta 信息。
- L162：进入无限循环 `while True`，在当前块内反复执行解码步骤。每步解码部分 [MASK] token，直到没有更多可解码 token 时 `break` 跳出。
- L163-L164：若启用了 Cache，调用 `cache.on_step_start(block_mask, frame)` 通知缓存系统新步骤开始，Cache 可以准备对应的 KV Cache 状态。
- L165-L183：调用 `generate_step` 执行单步解码。这是从 vanilla 策略复用的核心函数，内部完成：模型前向传播 → 获取 logits → 采样 → 通过 `unmasking_fn` 选择待解码位置 → 构建 FrameDelta。传入的自定义 `unmasking_fn` 使得每步的选择逻辑遵循 KLASS 的稳定性检测机制，而非 vanilla 的纯置信度策略。
- L184-L186：若 `delta` 为 `None`，表示当前块内没有更多可解码的 token（所有 [MASK] 已被解码或满足 `stop_until_eos` 停止条件），`break` 跳出内层 while 循环，进入下一个 block。
- L188：从 `delta.extra` 字典中弹出 `active_index` 和 `curr_probs`，使用当前步的概率分布更新 `prev_probs` 中对应活跃序列位置的记录。`pop` 操作同时将数据从 extra 中移除以节省内存。这样下一步 unmasking_fn 中的 KL 散度计算就能基于最新的历史概率。
- L189：将 `delta` 转换为模型的数据类型（如 bfloat16），与模型权重精度保持一致，避免后续 apply_delta 时的精度不匹配问题。
- L190-L191：若启用了 Cache，调用 `cache.on_step_end(block_mask, frame, delta)` 通知缓存系统当前步骤结束。Cache 可以将新解码位置的 KV 状态记录下来，后续步可直接复用而非重新计算。
- L193：将 delta 转移到 CPU 并追加到 `block_deltas` 列表中。转移到 CPU 可以减少 GPU 显存占用，因为 delta 的信息量相对较小（仅包含转移索引、解码 token、置信度等）。
- L194：`frame = frame.apply_delta(delta)` 将 delta 中选中的 token 应用到 Frame 的生成区域中，更新 `generated_tokens`。循环回到 while 开头，继续下一次解码。
- L196-L197：当前块处理完毕后（while 循环退出），若启用了 Cache，调用 `cache.on_block_end(block_mask, start_frame, block_deltas)` 通知缓存系统该块结束，可进行块级别的清理或优化。
- L199：将当前块的所有 delta 追加到全局 `deltas` 列表中。

### 返回 DecodeRecord

```python
# 源文件: src/generation/klass.py L201-L205
    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
```

**逐行解释：**

- L201-L205：所有块处理完毕后，构建并返回 `DecodeRecord` 对象。包含三个字段：
  - `initial_frame`：初始 Frame（转移到 CPU），记录 prompt 和初始的 `[MASK]` 序列。
  - `deltas`：所有解码步骤的 FrameDelta 序列，按时间顺序记录了每一步解码了哪些位置的 token、对应的置信度等信息。
  - `block_length`：块大小。调用方可通过 `DecodeRecord` 完整重现整个生成过程，包括每步解码的 token、置信度以及中间隐藏状态（若 `output_hidden_states=True`）。

## Token 选择策略

### 双重选择机制

KLASS 采用两阶段选择策略：

```
阶段 1: 稳定性优先
┌─────────────────────────────────────────┐
│  筛选条件: 所有历史 KL < kl_threshold   │
│  选择方式: 置信度排序 + threshold/factor│
└─────────────────────────────────────────┘
           │
           ▼ 如果没有稳定位置
阶段 2: 保底选择
┌─────────────────────────────────────────┐
│  筛选条件: 无                           │
│  选择方式: top-k 置信度                 │
└─────────────────────────────────────────┘
```

### 选择示例

```
配置: kl_threshold=0.01, kl_history_length=2, threshold=0.9

位置:           0       1       2       3       4
置信度:       0.95    0.88    0.92    0.75    0.85
KL 历史:      稳定   不稳定   稳定    不稳定   稳定
KL 值:       [0.005, 0.008]  [0.5, 0.3]  [0.003, 0.004]  [0.2, 0.1]  [0.007, 0.006]

阶段 1 (稳定位置):
  候选: 位置 0, 2, 4
  置信度 >= 0.9: 位置 0, 2
  选择: [0, 2]

阶段 2 (保底):
  不需要，因为阶段 1 已有选择

最终选择: [0, 2]
```

## 使用示例

### 配置文件

```yaml
# configs/generation/klass.yaml
defaults:
  - vanilla
  - _self_

strategy: klass

kl_threshold: 0.01
kl_history_length: 2
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
    generation=klass \
    generation.block_length=64 \
    generation.kl_threshold=0.01 \
    generation.kl_history_length=2 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.klass import klass_generate

result = klass_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=64,
    kl_threshold=0.01,       # KL 散度阈值
    kl_history_length=2,     # 历史长度
    threshold=0.9,           # 可选：置信度阈值
    mask_token_id=tokenizer.mask_token_id,
)
```

### 与 Parallel 结合

```python
result = klass_generate(
    model=model,
    input_ids=input_ids,
    kl_threshold=0.01,
    threshold=0.9,  # 结合置信度阈值
)
```

## 性能特点

### 优势

1. **显著加速**：在推理任务上实现最高 2.78x 加速
2. **质量保证**：稳定性检测确保解码质量
3. **自适应**：根据预测稳定性动态调整解码数量
4. **广泛适用**：可应用于文本、图像、分子生成等多种领域

### 劣势

1. **内存开销**：需要存储历史概率和 KL 值
2. **初始阶段**：前几步可能没有稳定位置
3. **参数敏感**：`kl_threshold` 需要根据任务调整

### 性能数据

```
推理基准测试 (LLaDA-7B-Instruct):

策略          准确率    速度提升    相对提升
Vanilla       42.3%    1.0x       baseline
KLASS         44.1%    2.78x      +1.8%
```

### 适用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 数学推理 | `kl_threshold=0.01`, `kl_history_length=2` | 默认配置 |
| 代码生成 | `kl_threshold=0.005`, `kl_history_length=3` | 更严格的稳定性 |
| 通用生成 | `kl_threshold=0.02`, `kl_history_length=2` | 更宽松的稳定性 |
| 快速生成 | `kl_threshold=0.05`, `kl_history_length=1` | 追求速度 |

## 参数调优指南

### kl_threshold 调优

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| 0.001-0.005 | 非常严格 | 高质量要求 |
| 0.005-0.02 | 平衡 | 通用场景 |
| 0.02-0.1 | 宽松 | 速度优先 |

### kl_history_length 调优

| 长度 | 效果 | 适用场景 |
|------|------|----------|
| 1 | 单步检测 | 快速生成 |
| 2-3 | 短期稳定 | 通用场景 |
| 4+ | 长期稳定 | 高质量要求 |

## 实现细节

### 数值稳定性

使用 `float64` 精度存储 KL 历史和概率：

```python
kl_history = torch.zeros(
    (batch_size, gen_length, kl_history_length),
    dtype=torch.float64,  # 高精度
    device=model.device,
)
```

### KL 散度计算

```python
eps = 1e-12
kl_current_prev = (
    probs
    * (torch.log(probs + eps) - torch.log(prev_probs[active_seq_idx] + eps))
).sum(dim=-1)
```

添加 `eps` 防止 log(0) 错误。

### 滚动更新

```python
kl_history[active_seq_idx] = kl_history[active_seq_idx].roll(shifts=-1, dims=-1)
kl_history[active_seq_idx, ..., -1] = kl_current_prev
```

使用 `roll` 实现高效的 FIFO 队列。

## 参考文献

- [KLASS: KL-Guided Fast Inference in Masked Diffusion Models](https://arxiv.org/abs/2505.13322) - KLASS 论文
