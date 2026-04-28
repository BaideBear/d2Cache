# Vanilla/Semi-AR 解码策略


## 算法逻辑精要

Vanilla解码采用迭代式掩码预测机制：初始化时将生成区域全部填充为[MASK]，每步由模型预测所有掩码位置的token分布并计算置信度，按置信度选择top-k个位置解码为确定token，重复迭代直至所有位置解码完成。当block_length < gen_length时，策略演变为Semi-AR模式，将生成序列划分为多个块，块内并行解码而块间顺序推进，兼顾解码速度与生成质量。核心流程由generate_step（单步前向推理、采样与选择）和confidence_unmasking（基于置信度的位置筛选）两个函数协作完成。

## 概述

Vanilla（原生）解码策略是扩散大语言模型（dLLM）中最基础的解码方法，源自 LLaDA 论文。它采用迭代式的掩码预测机制，逐步将掩码位置替换为确定的 token。当 `block_length < gen_length` 时，该策略演变为 Semi-Autoregressive（半自回归）模式，实现了块级别的顺序生成。

## 算法原理

### 核心思想

扩散语言模型的生成过程可以类比为图像扩散模型的去噪过程：

1. **初始化**：将生成区域全部填充为 `[MASK]` token
2. **迭代预测**：模型预测所有掩码位置的 token 分布
3. **置信度评估**：计算每个位置的预测置信度
4. **选择性解码**：根据置信度选择部分 token 进行解码
5. **重复迭代**：直到所有掩码位置都被解码

### Semi-Autoregressive 扩展

当设置 `block_length < gen_length` 时，生成过程被划分为多个块：

```
生成序列: [Block 0] -> [Block 1] -> [Block 2] -> ... -> [Block N-1]
```

每个块内部采用并行解码，块之间采用顺序解码，这种混合策略：
- 保留了并行解码的速度优势
- 通过块间依赖提升生成质量
- 类似于"粗粒度"的自回归生成

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Vanilla 解码流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                            │
│  │ 初始化 Frame │                                            │
│  │ (全 MASK)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐            │
│  │           Block 循环 (block_idx)             │            │
│  │  ┌───────────────────────────────────────┐  │            │
│  │  │         Step 循环 (迭代解码)           │  │            │
│  │  │  ┌─────────────────────────────────┐  │  │            │
│  │  │  │ 1. 模型前向传播                  │  │  │            │
│  │  │  │    input: [prompt + tokens]     │  │  │            │
│  │  │  │    output: logits               │  │  │            │
│  │  │  └──────────────┬──────────────────┘  │  │            │
│  │  │                 ▼                      │  │            │
│  │  │  ┌─────────────────────────────────┐  │  │            │
│  │  │  │ 2. Token 采样                    │  │  │            │
│  │  │  │    - temperature 控制           │  │  │            │
│  │  │  │    - top-k/top-p 过滤           │  │  │            │
│  │  │  │    - 计算置信度                  │  │  │            │
│  │  │  └──────────────┬──────────────────┘  │  │            │
│  │  │                 ▼                      │  │            │
│  │  │  ┌─────────────────────────────────┐  │  │            │
│  │  │  │ 3. 选择解码位置                  │  │  │            │
│  │  │  │    - 置信度排序                  │  │  │            │
│  │  │  │    - 选择 top-k 位置            │  │  │            │
│  │  │  └──────────────┬──────────────────┘  │  │            │
│  │  │                 ▼                      │  │            │
│  │  │  ┌─────────────────────────────────┐  │  │            │
│  │  │  │ 4. 更新 Frame                    │  │  │            │
│  │  │  │    - 解码选中的 token           │  │  │            │
│  │  │  │    - 更新置信度                  │  │  │            │
│  │  │  └──────────────┬──────────────────┘  │  │            │
│  │  │                 ▼                      │  │            │
│  │  │         还有 MASK? ──Yes──┐           │  │            │
│  │  │                 │         │           │  │            │
│  │  │                 No        │           │  │            │
│  │  └─────────────────│─────────┘           │  │            │
│  │                    ▼                      │  │            │
│  │              下一个 Block ◄───────────────┘  │            │
│  └─────────────────────────────────────────────┘            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │ 返回结果     │                                            │
│  │ DecodeRecord │                                            │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心参数

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列的总长度 |
| `block_length` | int | - | 块大小，必须能整除 `gen_length` |
| `num_transfer_tokens` | int | 1 | 每步最少解码的 token 数量 |
| `temperature` | float | 0.0 | 采样温度，0 表示贪婪解码 |
| `top_k` | int | None | top-k 过滤参数 |
| `top_p` | float | None | nucleus sampling 参数 |
| `alg` | str | "maskgit_plus" | 置信度计算算法 |

### 高级参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sigma` | float | None | 确定性先验的高斯核标准差 |
| `stop_until_eos` | bool | False | 是否在遇到 EOS 时停止 |
| `output_probs` | bool | False | 是否输出概率分布 |

## 详细代码流程分析

### generate_step — 单步生成主函数

```python
# 源文件: src/generation/vanilla.py L19-L139
@torch.no_grad()
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
    unmasking_fn: Callable,
    attention_mask: torch.Tensor | None = None,
    past_key_values: dCache | None = None,
    alg: str = "maskgit_plus",
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: float | None = None,
    mask_token_id: int = None,
    eos_token_id: int | None = None,
    sigma: float | None = None,
    stop_until_eos: bool = False,
    debias: bool = False,
    clip_alpha: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
) -> FrameDelta | None:
    frame = frame.as_batch()
    batch_size, prompt_length = frame.prompts.shape
    device = block_mask.device

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
    active_seq_idx = torch.nonzero(can_generate, as_tuple=True)[0]

    remaining_mask = frame.generated_tokens == mask_token_id
    transfer_index_mask = remaining_mask.clone()

    if past_key_values is not None:
        past_key_values.active_seq_mask = can_generate

    x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[active_seq_idx]
    attention_mask = (
        attention_mask[active_seq_idx] if attention_mask is not None else None
    )
    block_mask = block_mask[active_seq_idx]
    outputs = model(
        x,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        past_key_values=past_key_values,
        use_cache=past_key_values is not None,
    )

    logits = prepare_logits_for_generation(model, outputs.logits)
    if past_key_values is not None and past_key_values.active_q_mask is not None:
        if is_adapted_from_ar(model):
            valid_mask = past_key_values.active_q_mask[:, prompt_length - 1 : -1]
        else:
            valid_mask = past_key_values.active_q_mask[:, prompt_length:]
        transfer_index_mask[active_seq_idx].logical_and_(valid_mask)
    logits = logits[:, prompt_length:]
    transfer_index_mask = transfer_index_mask[active_seq_idx]
    remaining_mask = remaining_mask[active_seq_idx]

    hidden_states = (
        tuple((i, hs) for i, hs in enumerate(outputs.hidden_states))
        if output_hidden_states
        else None
    )

    confidence, x0, p = sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        debias=debias,
        clip_alpha=clip_alpha,
        alg=alg,
    )
    scores = confidence = torch.where(transfer_index_mask, confidence, -torch.inf)
    if sigma is not None and sigma > 0:
        scores = confidence * certainty_density(~remaining_mask, sigma=sigma)

    transfer_index, extra = unmasking_fn(
        active_seq_idx=active_seq_idx,
        scores=scores,
        probs=p,
        transfer_index_mask=transfer_index_mask,
        block_mask=block_mask,
        num_transfer_tokens=num_transfer_tokens,
    )
    if len(transfer_index) != int(active_seq_idx.numel()):
        raise ValueError(
            "Transfer selector must return one index tensor per active sequence."
        )

    full_transfer_index = [
        torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    for seq_idx, index in zip(active_seq_idx.tolist(), transfer_index):
        full_transfer_index[seq_idx] = index

    return FrameDelta(
        transfer_index=tuple(full_transfer_index),
        decoded_tokens=torch.where(transfer_index_mask, x0, INVALID_TOKEN_ID),
        confidence=confidence,
        probs=(
            torch.where(transfer_index_mask.unsqueeze(-1), p, -torch.inf)
            if output_probs
            else None
        ),
        intermediate=Intermediate(
            hidden_states=hidden_states if hidden_states is not None else tuple()
        ),
        extra=extra,
    )
```

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L19 | `@torch.no_grad()` — 禁用梯度计算，生成阶段不需要反向传播 |
| L20-L39 | 函数签名：接收模型、Frame、block_mask、unmasking_fn（策略注入点）等参数 |
| L40-L42 | `as_batch()` 将 Frame 转为批处理模式；获取 batch_size(B)、prompt_length(P)、device |
| L44-L51 | `check_can_generate` — 检查当前块内是否还有足够的掩码位置可解码，若无可生成位置则返回 None |
| L54 | `active_seq_idx` — 仍可生成的活动序列索引（非结束的序列） |
| L56-L57 | `remaining_mask` — 标记仍为 [MASK] 的位置；`transfer_index_mask` 为可转移位置的拷贝 |
| L59-L60 | 若启用缓存，将 `active_seq_mask` 设为 `can_generate`，通知缓存哪些序列活跃 |
| L62-L66 | 将 prompt 和 generated_tokens 沿最后一维拼接 (B, P+G) 再按 active_seq_idx 过滤；同步裁剪 attention_mask 和 block_mask |
| L67-L73 | 模型前向传播：输入拼接序列，输出 logits (B, P+G, vocab_size)；通过 past_key_values 注入缓存 |
| L75 | `prepare_logits_for_generation` — 标准化 logits 输出（处理不同模型格式差异） |
| L76-L81 | 当使用 KV Cache 且 `active_q_mask` 存在时：用 `active_q_mask` 过滤 `transfer_index_mask`，确保只考虑被缓存实际计算过的位置；对 Dream 模型特殊处理 prompt_length 偏移 |
| L82-L84 | 截取 logits 的生成部分 (B, G, vocab_size)；同步裁剪 transfer_index_mask 和 remaining_mask |
| L86-L90 | 可选地提取 hidden_states，附上层的索引便于后续使用 |
| L93-L101 | `sample_tokens` — 采样生成位置的所有 token：返回 confidence (B, G)、x0 (B, G) 采样结果、p (B, G, V) 完整概率分布 |
| L102 | 将不可转移位置的 scores/confidence 置为 `-inf`，防止被选中 |
| L103-L104 | 若 `sigma > 0`，用 `certainty_density`（确定性密度）对 scores 进行加权，引导选择周围已确定位置多的 token |
| L107-L114 | 调用策略注入的 `unmasking_fn`：返回 `transfer_index`（每个活动序列的解码位置索引）和 `extra`（额外数据如 KL 历史） |
| L115-L118 | 校验：unmasking_fn 必须为每个活动序列返回恰好一个索引张量 |
| L120-L124 | 将活动序列的 transfer_index 映射回完整的 batch 顺序 `full_transfer_index`（非活动序列为空张量） |
| L126-L139 | 构建 `FrameDelta`：包含 transfer_index、decoded_tokens（可转移位置填 x0，其余填 INVALID_TOKEN_ID）、confidence、可选的 probs 和 hidden_states、extra |

**数据流形状变化关键路径：**
- 输入 x: `(B_active, P+G)` → logits: `(B_active, P+G, V)` → 截取: `(B_active, G, V)`
- confidence/x0/p: `(B_active, G)` / `(B_active, G)` / `(B_active, G, V)`
- transfer_index: 每序列一个一维索引张量

---

### confidence_unmasking — 核心选位函数

```python
# 源文件: src/generation/vanilla.py L142-L269
def confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: torch.Tensor | int,
    max_transfer_tokens: torch.Tensor | None = None,
    threshold: float | torch.Tensor | None = None,
    factor: float | None = None,
    gamma: float | None = None,
    p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if (threshold is not None) + (factor is not None) + (gamma is not None) > 1:
        raise ValueError(
            "Only one of `threshold`, `factor`, or `gamma` should be provided."
        )

    batch_size, _ = scores.shape

    if isinstance(min_transfer_tokens, int):
        min_transfer_tokens = torch.full(
            (batch_size,),
            min_transfer_tokens,
            device=scores.device,
            dtype=torch.long,
        )

    if min_transfer_tokens.numel() != batch_size:
        raise ValueError(
            "`min_transfer_tokens` must have shape (batch_size,) to match scores."
        )

    if max_transfer_tokens is not None:
        if max_transfer_tokens.numel() != batch_size:
            raise ValueError(
                "`max_transfer_tokens` must have shape (batch_size,) to match scores."
            )
    else:
        max_transfer_tokens = torch.sum(transfer_index_mask, dim=-1)
    num_transfer_tokens = torch.minimum(min_transfer_tokens, max_transfer_tokens)

    confidence = torch.where(transfer_index_mask, scores, -torch.inf)
    transfer_index = [torch.tensor([]) for _ in range(batch_size)]
    if threshold is not None or factor is not None:
        if threshold is not None:
            col_indices = torch.nonzero(confidence >= threshold, as_tuple=False)[:, 1]
            counts = torch.sum(confidence >= threshold, dim=-1).cpu().tolist()
            transfer_index = list(torch.split(col_indices, counts))
            for i, t in enumerate(transfer_index):
                if t.numel() > max_transfer_tokens[i]:
                    transfer_index[i] = torch.tensor([])
                    num_transfer_tokens[i] = max_transfer_tokens[i]
        elif factor is not None:
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
    elif gamma is not None:
        if p is None:
            raise ValueError(
                "Probabilities of all tokens `p` must be provided for EB sampler."
            )
        _, ids = torch.sort(confidence, dim=-1, descending=True)
        entropy = torch.gather(
            dists.Categorical(probs=p.float()).entropy(), dim=-1, index=ids
        )
        acc_entropy = torch.cumsum(entropy, dim=1)
        cummax_entropy = torch.cummax(entropy, dim=0).values
        num_transfer_tokens = (acc_entropy - cummax_entropy <= gamma).sum(dim=1)

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

    return tuple(transfer_index)
```

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L168-L171 | 互斥校验：`threshold`、`factor`、`gamma` 最多只能提供一个，否则报错 |
| L173 | 获取 batch_size (B) |
| L175-L181 | 若 `min_transfer_tokens` 是整数，广播为形状 (B,) 的张量 |
| L183-L186 | `min_transfer_tokens` 必须匹配 batch_size |
| L188-L195 | 若未提供 `max_transfer_tokens`，则默认使用每个序列的可转移位置数作为上限；`num_transfer_tokens = min(min, max)` |
| L197 | 将 `scores` 通过 `transfer_index_mask` 过滤，不可转移位置置 `-inf` |
| L198 | 为每个 batch 初始化空 `transfer_index` 列表 |
| L199-L224 | **并行解码分支**（threshold 或 factor） |
| L200-L211 | **threshold 分支**：筛选所有 `confidence >= threshold` 的位置索引，通过 `split` 按序列分组。若某序列选中数量超过 `max_transfer_tokens`，清空该序列的 transfer_index 回退到 top-k |
| L212-L224 | **factor 分支**：对每个序列单独计算：将可转移位置的置信度降序排序；从 n=1 开始递增，直到 `(n+1)*(1-sorted_conf[n-1]) >= factor` 时停止；取前 n-1 个作为解码位置 |
| L225-L237 | **EB-Sampler 分支**（gamma）：按置信度降序排序，用 `Categorical.entropy()` 计算每个位置的熵；计算累积熵 `acc_entropy` 和累积最大熵 `cummax_entropy`；`num_transfer_tokens` = 满足 `acc_entropy - cummax_entropy <= gamma` 的位置数 |
| L239-L243 | `torch.clamp` 确保 `num_transfer_tokens` 在 [min, max] 范围内 |
| L245-L267 | **Top-k 回退**：对 transfer_index 数量不足 `num_transfer_tokens` 的序列，从置信度中取 top-k 个最高分位置补充，确保至少解码 min_transfer_tokens 个 token |
| L269 | 返回 `tuple[torch.Tensor, ...]`，每序列一个索引张量 |

**关键控制流：**
- Vanilla 策略：不传 threshold/factor/gamma → 上述分支全部跳过 → 直接进入 L245 的 top-k 回退 → 选择 `num_transfer_tokens` 个最高置信度位置
- Parallel 策略：传 threshold 或 factor → 走 L200-L224 分支
- EB-Sampler 策略：传 gamma → 走 L225-L237 分支

---

### vanilla_generate — 主入口函数

```python
# 源文件: src/generation/vanilla.py L272-L437
@register("vanilla")
def vanilla_generate(
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
    gamma: float | None = None,
    debias: bool = False,
    clip_alpha: float | None = None,
    threshold: float | None = None,
    factor: float | None = None,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(...)
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))
    if stop_until_eos:
        if eos_token_id is None and os.environ.get("EOS_TOKEN_ID", None) is None:
            raise ValueError(...)
        eos_token_id = eos_token_id or int(os.environ.get("EOS_TOKEN_ID"))

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if num_transfer_tokens <= 0:
        raise ValueError(f"{num_transfer_tokens=} must be > 0")

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    def unmasking_fn(...):
        return (
            confidence_unmasking(
                scores=scores,
                transfer_index_mask=transfer_index_mask & block_mask,
                min_transfer_tokens=num_transfer_tokens,
                threshold=threshold,
                factor=factor,
                gamma=gamma,
                p=probs,
            ),
            {},
        )

    deltas = []
    for block_idx in range(num_blocks):
        block_mask = torch.zeros(...)
        block_mask[:, block_idx*block_length:(block_idx+1)*block_length] = True
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            delta = generate_step(...)
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

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L272 | `@register("vanilla")` — 注册解码策略，使配置文件可通过 `"vanilla"` 名称引用 |
| L273-L300 | 函数签名：接收模型、输入 token id、所有解码参数（threshold/factor/gamma/debias 等）和可选的 `cache_cls` |
| L322-L326 | 参数校验：若未传 `mask_token_id` 且环境变量也未设置则报错；优先使用传入参数，否则从环境变量读取 |
| L327-L332 | EOS 参数校验：若 `stop_until_eos=True` 则需要提供 `eos_token_id` |
| L334-L337 | 断言 `gen_length` 可被 `block_length` 整除；计算 `num_blocks`；`num_transfer_tokens` 必须 > 0 |
| L339-L343 | 创建 `initial_frame`：prompts 来自 input_ids，generated_tokens 全填充为 [MASK]，迁移到模型设备 |
| L345-L351 | 构造 `attention_mask`：若未提供则从 `pad_token_id` 生成；若为 (B, P) 形状则右侧填充 gen_length 个 1 |
| L353-L354 | 若提供了 `cache_cls` 则实例化缓存对象；`frame` 指向 `initial_frame` |
| L356-L376 | 定义 `unmasking_fn`（闭包）：调用 `confidence_unmasking`，将 `transfer_index_mask & block_mask` 的交集作为可转移范围，传递 threshold/factor/gamma/p 参数 |
| L380-L431 | **Block 循环**：每个 block 创建布尔 `block_mask` 标记当前块范围；调用 `cache.on_block_start` 初始化块级缓存；while True 循环中：调用 `cache.on_step_start` → `generate_step` 执行单步解码 → 若 delta 为 None 则跳出循环 → `cache.on_step_end` → 将 delta 移至 CPU 并累积 → `frame.apply_delta(delta)` 更新 Frame |
| L433-L437 | 返回 `DecodeRecord`：包含初始 Frame、所有 deltas 和 block_length，用于后续分析和评估

## Token 选择策略

### 置信度计算算法

`sample_tokens` 函数支持多种置信度计算方式：

| 算法 | 描述 |
|------|------|
| `maskgit_plus` | 直接使用最大概率作为置信度（默认） |
| `topk_margin` | top-1 与 top-2 概率之差 |
| `entropy` | 负熵值 |
| `random` | 随机置信度 |

```python
if alg == "topk_margin":
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    top1_probs = sorted_probs[..., 0]
    top2_probs = sorted_probs[..., 1]
    confidence = top1_probs - top2_probs
elif alg == "entropy":
    log_probs = torch.log(probs + epsilon)
    confidence = torch.sum(probs * log_probs, dim=-1)
elif alg == "random":
    confidence = torch.rand_like(confidence)
```

### 选择逻辑

```python
confidence = torch.where(transfer_index_mask, confidence, -torch.inf)
topk_transfer_index = torch.topk(confidence, k, dim=-1).indices
```

## 使用示例

### 基础配置

```yaml
# configs/generation/vanilla.yaml
strategy: vanilla
alg: "maskgit_plus"
gen_length: null
block_length: null
num_transfer_tokens: 1
temperature: 0.0
top_p: null
top_k: null
sigma: null
stop_until_eos: false
output_probs: false
```

### 命令行使用

```bash
# 标准 Vanilla 解码
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
    model=llada-inst

# Semi-Autoregressive 模式
accelerate launch eval.py \
    generation=vanilla \
    generation.gen_length=256 \
    generation.block_length=64 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.vanilla import vanilla_generate
from src.generation.utils import decode_final_frame

result = vanilla_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    num_transfer_tokens=1,
    temperature=0.0,
    mask_token_id=tokenizer.mask_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

final_frame = result[-1]
generated_text = decode_final_frame(tokenizer, final_frame)
```

## 性能特点

### 优势

1. **简单可靠**：最基础的解码策略，易于理解和调试
2. **质量保证**：逐步解码确保生成质量
3. **灵活可控**：通过 `num_transfer_tokens` 控制速度-质量权衡
4. **Semi-AR 支持**：块级别顺序生成，兼顾质量与效率

### 劣势

1. **速度较慢**：每步只解码少量 token，需要多次迭代
2. **缺乏并行性**：相比其他策略，并行解码能力有限

### 适用场景

| 场景 | 推荐配置 |
|------|----------|
| 高质量生成 | `num_transfer_tokens=1`, `block_length=gen_length` |
| 平衡模式 | `num_transfer_tokens=2-4`, `block_length=gen_length/4` |
| 快速生成 | `num_transfer_tokens=4+`, 配合 KV Cache |

## 与其他策略的关系

Vanilla 是所有其他解码策略的基础：

```
Vanilla (基础)
    │
    ├── Parallel (添加 threshold/factor)
    │
    ├── PC-Sampler (添加 debias)
    │
    ├── EB-Sampler (添加 gamma)
    │
    ├── KLASS (添加 KL 稳定性检测)
    │
    └── AR (修改为自回归解码顺序)
```

## 参考文献

- [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) - LLaDA 论文
- [d2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094) - 确定性先验解码
