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

## 代码流程分析

### 主函数入口

```python
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
    ...
) -> DecodeRecord:
```

### 关键步骤详解

#### 1. 初始化 Frame

```python
initial_frame = Frame.create_initial_frame(
    input_ids,
    gen_length=gen_length,
    mask_token_id=mask_token_id,
).to(device=model.device, dtype=model.dtype)
```

Frame 是 d2Cache 框架中的核心数据结构，包含：
- `prompts`: 输入的 prompt tokens
- `generated_tokens`: 已生成/待生成的 tokens（初始为 MASK）
- `confidence`: 每个 token 的置信度
- `steps`: 每个 token 的解码步数

#### 2. Block 循环

```python
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
```

`block_mask` 标识当前正在处理的块范围。

#### 3. 生成步骤 (`generate_step`)

这是核心的单步生成函数：

```python
def generate_step(
    model,
    frame: Frame,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
    unmasking_fn: Callable,
    ...
) -> FrameDelta | None:
```

主要流程：

**a. 检查是否可以生成**

```python
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
```

**b. 模型前向传播**

```python
outputs = model(
    x,
    attention_mask=attention_mask,
    output_hidden_states=output_hidden_states,
    past_key_values=past_key_values,
    use_cache=past_key_values is not None,
)
logits = prepare_logits_for_generation(model, outputs.logits)
```

**c. Token 采样**

```python
confidence, x0, p = sample_tokens(
    logits,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    debias=debias,
    clip_alpha=clip_alpha,
    alg=alg,
)
```

**d. 选择解码位置**

```python
transfer_index, extra = unmasking_fn(
    active_seq_idx=active_seq_idx,
    scores=scores,
    probs=p,
    transfer_index_mask=transfer_index_mask,
    block_mask=block_mask,
    num_transfer_tokens=num_transfer_tokens,
)
```

#### 4. 置信度解码 (`confidence_unmasking`)

```python
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
```

对于 Vanilla 策略，使用基础的 top-k 选择：
- 选择置信度最高的 `num_transfer_tokens` 个位置
- 这些位置的 token 将被"固定"（解码）

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
