# PC-Sampler 解码策略


## 算法逻辑精要

PC-Sampler通过位置感知的置信度校准解决解码早期偏向平凡token的问题：当启用debias参数时，利用全局token频率统计对原始置信度进行校准（calibrated = −confidence × log(freq + ε)），从而有效抑制标点、停用词等高频但信息量低的token，使模型更倾向于选择有意义的内容词。校准后的置信度值进一步通过clip_alpha进行裁剪控制，避免极端值干扰。该机制作为通用采样增强，嵌入在标准解码流程的sample_tokens阶段。

## 概述

PC-Sampler（Position-Aware Confidence-Calibrated Sampling，位置感知置信度校准采样）是一种针对掩码扩散模型的解码策略。它通过位置感知的加权机制和校准置信度分数，解决了传统采样器在解码早期偏向平凡 token（如标点、停用词）的问题。

## 算法原理

### 问题背景

传统基于不确定性的采样器存在两个关键局限：

1. **缺乏全局轨迹控制**：解码过程没有考虑整体生成路径
2. **早期偏向平凡 token**：在解码早期阶段，模型倾向于选择高频但信息量低的 token

### 核心思想

PC-Sampler 提出两个关键机制：

#### 1. 位置感知加权（Position-Aware Weighting）

通过位置权重调节解码路径，使模型更加关注关键位置：

```
位置权重 = f(position, total_length)
```

#### 2. 校准置信度（Calibrated Confidence）

使用 token 频率对置信度进行校准，抑制高频平凡 token：

```
calibrated_confidence = -confidence * log(token_frequency + ε)
```

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    PC-Sampler 解码流程                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Token 采样阶段                          │    │
│  │                                                         │    │
│  │  1. 标准 Token 采样                                      │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  logits = model(input)                │           │    │
│  │     │  probs = softmax(logits / temp)       │           │    │
│  │     │  confidence, x0 = probs.max(dim=-1)   │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  2. Debias 校准 (关键步骤)                               │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  if debias:                           │           │    │
│  │     │    # 加载 token 频率统计              │           │    │
│  │     │    token_freq = get_token_freq()      │           │    │
│  │     │                                       │           │    │
│  │     │    # 校准置信度                        │           │    │
│  │     │    calibrated = -confidence *         │           │    │
│  │     │                   log(freq[x0] + ε)   │           │    │
│  │     │                                       │           │    │
│  │     │    # 裁剪到合理范围                    │           │    │
│  │     │    confidence = clamp(calibrated,     │           │    │
│  │     │                       max=clip_alpha) │           │    │
│  │     └───────────────────┬───────────────────┘           │    │
│  │                         ▼                               │    │
│  │  3. 选择解码位置                                         │    │
│  │     ┌───────────────────────────────────────┐           │    │
│  │     │  # 使用校准后的置信度进行选择          │           │    │
│  │     │  transfer_index = topk(confidence)    │           │    │
│  │     └───────────────────────────────────────┘           │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Debias 校准原理

```
原始置信度 vs 校准置信度:

Token:     "the"    "is"    "algorithm"   "function"
原始概率:   0.95    0.88      0.72         0.65
频率:       0.05    0.03      0.0001       0.0002

校准后:
           -0.95*log(0.05)  -0.88*log(0.03)  -0.72*log(0.0001)  -0.65*log(0.0002)
           = 2.85           = 3.11           = 6.62             = 5.51

排序变化:
原始: "the" > "is" > "algorithm" > "function"
校准: "algorithm" > "function" > "is" > "the"
```

## 核心参数

### PC-Sampler 特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debias` | bool | False | 是否启用位置感知偏差校准 |
| `clip_alpha` | float | 10.0 | 校准置信度的上界裁剪值 |

### 继承参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | int | - | 生成序列总长度 |
| `block_length` | int | - | 块大小 |
| `num_transfer_tokens` | int | 1 | 每步解码数量 |
| `temperature` | float | 0.0 | 采样温度 |
| `alg` | str | "maskgit_plus" | 置信度计算算法 |

## 详细代码流程分析

PC-Sampler 的实现嵌入在 `vanilla_generate` 的参数传递中，核心校准逻辑位于 `src/generation/utils.py` 的 `sample_tokens` 函数内部。当用户设置 `debias=True` 时，`sample_tokens` 在校准阶段使用词频统计修正置信度，再通过 `generate_step` → `confidence_unmasking` 进行位置选择。

### debias 参数在 generate_step 中的传递

```python
# 源文件: src/generation/vanilla.py L93-L101
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

**逐行讲解：**

| 行号 | 说明 |
|------|------|
| L93-L101 | `sample_tokens` 接收 `debias`（布尔，启用位置感知频率校准）和 `clip_alpha`（浮点，校准后置信度的上界裁剪值）；返回值 `confidence` (B, G) 已经是校准后的分数（若 debias=True），后续 `confidence_unmasking` 直接使用该校准后的分数进行 top-k 选择 |

### vanilla_generate 中的 debias 参数入口

```python
# 源文件: src/generation/vanilla.py L291-L293
# PC sampler
debias: bool = False,
clip_alpha: float | None = None,
```

**逐行讲解：**
- `vanilla_generate` 函数签名中声明 `debias`（默认 False）和 `clip_alpha`（默认 None）参数
- 这两个参数原封不动传递给 `generate_step`（L415-L418），再由 `generate_step` 传给 `sample_tokens`

### confidence_unmasking 如何使用校准后的分数

```python
# 源文件: src/generation/vanilla.py L356-L376
def unmasking_fn(
    active_seq_idx: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
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
```

**逐行讲解：**
- `scores` 参数已包含经过 `sample_tokens`（含 debias 校准）和可选 `certainty_density`（sigma）处理后的分数
- `confidence_unmasking` 在不传 threshold/factor/gamma 时进入 top-k 回退（L245-L267），直接对 `scores` 做 top-k 选择
- 因此 debias 的校准效果最终通过 `scores` 影响位置选择的排序

## Token 选择策略

### 校准公式详解

```python
calibrated_confidence = -confidence * log(token_frequency + ε)
```

**公式解释：**

1. `confidence`: 原始预测置信度（最大概率）
2. `token_frequency`: 该 token 在训练语料中的频率
3. `-log(token_frequency)`: 频率的负对数，高频 token 值小，低频 token 值大
4. 最终乘积：高频 token 的置信度被抑制，低频 token 的置信度被提升

### 裁剪机制

```python
confidence = torch.clamp_max(calibrated_confidence, max=clip_alpha)
```

裁剪的目的：
- 防止极低频 token 的置信度过大
- 保持数值稳定性
- `clip_alpha` 默认为 10.0，可根据需要调整

### 选择示例

```
假设 clip_alpha = 10.0

位置:        0       1       2       3
Token:     "the"  "code"  "the"  "solve"
原始置信度:  0.95   0.72    0.88    0.65
频率:       0.05   0.001   0.05    0.0005

校准计算:
位置 0: -0.95 * log(0.05) = 2.85
位置 1: -0.72 * log(0.001) = 4.97
位置 2: -0.88 * log(0.05) = 2.64
位置 3: -0.65 * log(0.0005) = 5.01

校准后置信度: [2.85, 4.97, 2.64, 5.01]
排序: 位置 3 > 位置 1 > 位置 0 > 位置 2

选择结果 (top-1): 位置 3 ("solve")
```

## 使用示例

### 配置文件

```yaml
# configs/generation/pc_sampler.yaml
defaults:
  - vanilla
  - _self_

strategy: pc_sampler

debias: true
clip_alpha: 10
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
    generation=pc_sampler \
    generation.num_transfer_tokens=1 \
    generation.gen_length=256 \
    generation.block_length=32 \
    generation.debias=true \
    generation.clip_alpha=10 \
    model=llada-inst
```

### 代码调用

```python
from src.generation.vanilla import vanilla_generate

result = vanilla_generate(
    model=model,
    input_ids=input_ids,
    gen_length=256,
    block_length=32,
    debias=True,           # 启用 PC-Sampler
    clip_alpha=10.0,       # 校准裁剪值
    num_transfer_tokens=1,
    mask_token_id=tokenizer.mask_token_id,
)
```

### 准备 Token 频率文件

要使用 PC-Sampler，需要准备对应模型的 token 频率统计：

```python
# 生成 token 频率统计的示例代码
import json
from collections import Counter

def generate_token_frequency(tokenizer, corpus_path, output_path):
    """从语料库生成 token 频率统计"""
    counter = Counter()
    total_tokens = 0
    
    with open(corpus_path, 'r') as f:
        for line in f:
            tokens = tokenizer.encode(line.strip())
            counter.update(tokens)
            total_tokens += len(tokens)
    
    # 计算频率
    freq = {token: count / total_tokens 
            for token, count in counter.items()}
    
    # 保存为 JSON
    with open(output_path, 'w') as f:
        json.dump(freq, f)
```

## 性能特点

### 优势

1. **抑制平凡 token**：有效减少高频低信息量 token 的过早选择
2. **提升关键内容**：优先选择信息量高的 token
3. **即插即用**：无需修改模型，仅调整解码策略
4. **广泛适用**：可与其他策略（如 Parallel）结合使用

### 劣势

1. **依赖频率数据**：需要预先准备 token 频率统计
2. **语言敏感性**：不同语言/领域可能需要不同的频率统计
3. **参数调节**：`clip_alpha` 需要根据具体任务调整

### 性能对比

```
GSM8K 基准测试 (LLaDA-7B-Instruct):

策略              准确率    相对提升
Vanilla           42.3%    baseline
PC-Sampler        46.8%    +4.5%
```

### 适用场景

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 数学推理 | `debias=true`, `clip_alpha=10` | 默认配置 |
| 代码生成 | `debias=true`, `clip_alpha=15` | 更强调关键词 |
| 通用对话 | `debias=true`, `clip_alpha=8` | 适度校准 |
| 文本摘要 | `debias=true`, `clip_alpha=10` | 平衡配置 |

## 与其他策略的结合

PC-Sampler 可以与 Parallel 解码结合使用：

```bash
# PC-Sampler + Parallel
accelerate launch eval.py \
    generation=pc_sampler \
    generation.debias=true \
    generation.threshold=0.9 \
    model=llada-inst
```

这种组合：
- Parallel 提供速度提升
- PC-Sampler 保证质量

## 实现细节

### 频率数据格式

```json
// llada_corpus.json 示例
{
    "0": 0.0001,    // token id -> 频率
    "1": 0.05,
    "2": 0.03,
    ...
}
```

### 内存优化

Token 频率张量在首次使用时加载并缓存：

```python
_token_freq: torch.Tensor | None = None

def get_token_freq(model_family: str, vocab_size: int) -> torch.Tensor:
    global _token_freq
    if _token_freq is not None:
        return _token_freq
    
    # 加载并转换为张量
    with open(f"src/third_party/{model_family}_corpus.json") as f:
        freq_dict = json.load(f)
    
    freq_tensor = torch.zeros(vocab_size)
    for token_id, freq in freq_dict.items():
        freq_tensor[int(token_id)] = freq
    
    _token_freq = freq_tensor
    return _token_freq
```

## 参考文献

- [PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models](https://arxiv.org/pdf/2508.13021) - PC-Sampler 论文
