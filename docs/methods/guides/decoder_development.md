# 解码策略扩展开发指南

本指南详细说明如何为 d2Cache 项目开发自定义解码策略。解码策略定义了扩散语言模型的生成算法，包括自回归、非自回归或迭代细化等方法。

## 目录

- [概述](#概述)
- [实现步骤](#实现步骤)
- [必须实现的方法](#必须实现的方法)
- [关键代码模板](#关键代码模板)
- [配置文件编写](#配置文件编写)
- [测试和验证](#测试和验证)
- [最佳实践](#最佳实践)

---

## 概述

d2Cache 框架支持模块化解码策略，允许研究人员实现自定义生成算法而无需修改核心模型代码。新策略通过 `@register` 装饰器注册，可通过配置文件或命令行参数选择。

### 核心概念

- **Frame**: 生成帧，封装生成状态（包括提示词和当前 token 序列）
- **FrameDelta**: 帧变化，包含一个解码步骤中的变化（新 token、置信度分数）
- **DecodeRecord**: 解码记录，包含初始帧和所有 delta 的历史
- **generate_step**: 生成步骤函数，执行一次前向传播并采样新 token

### 解码流程

```
1. 初始化 Frame（包含提示词和掩码 token）
2. 初始化 Cache（如果需要）
3. 迭代生成循环：
   a. 执行 generate_step
   b. 获取 FrameDelta
   c. 应用 delta 到 frame
4. 返回 DecodeRecord
```

---

## 实现步骤

### 步骤 1: 创建策略文件

在 `src/generation/` 目录下创建新的 Python 文件，例如 `my_strategy.py`：

```python
import os
import torch
import torch.nn.functional as F
from typing import Any, Type

from src.cache import dCache
from src.frame import Frame, FrameDelta, DecodeRecord
from src.generation.utils import register, generate_step
```

### 步骤 2: 定义 unmasking 函数

unmasking 函数决定每步选择哪些 token 进行解码：

```python
def my_unmasking(
    active_seq_idx: torch.Tensor,
    scores: torch.Tensor,
    probs: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    block_mask: torch.Tensor,
    num_transfer_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """
    自定义 unmasking 逻辑。
    
    Args:
        active_seq_idx: 活跃序列索引
        scores: 置信度分数，形状 [B, gen_length]
        probs: token 概率分布，形状 [B, gen_length, vocab_size]
        transfer_index_mask: 可传输位置的掩码
        block_mask: 当前块的掩码
        num_transfer_tokens: 每步最小传输 token 数
    
    Returns:
        tuple: (传输索引元组, 额外信息字典)
    """
    batch_size = scores.shape[0]
    device = scores.device
    
    # 选择 top-k 高置信度位置
    transfer_index = []
    for i in range(batch_size):
        valid_scores = torch.where(
            transfer_index_mask[i] & block_mask[i],
            scores[i],
            -torch.inf
        )
        k = min(num_transfer_tokens, valid_scores.sum().int().item())
        if k > 0:
            _, indices = torch.topk(valid_scores, k)
            transfer_index.append(indices)
        else:
            transfer_index.append(torch.tensor([], dtype=torch.long, device=device))
    
    return tuple(transfer_index), {}
```

### 步骤 3: 实现主生成函数

```python
@register("my_strategy")
def my_strategy_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
    # 自定义参数
    my_custom_param: float = 0.5,
) -> DecodeRecord:
    """
    自定义解码策略。
    
    Args:
        model: 掩码预测器
        input_ids: 输入 ID，形状 (B, prompt_len)
        gen_length: 生成长度
        block_length: 块长度
        num_transfer_tokens: 每步最小传输 token 数
        temperature: 采样温度
        top_k: top-k 过滤
        top_p: nucleus 采样阈值
        mask_token_id: 掩码 token ID
        pad_token_id: 填充 token ID
        eos_token_id: 结束 token ID
        stop_until_eos: 是否在 EOS 处停止
        output_hidden_states: 是否输出隐藏状态
        output_probs: 是否输出概率
        cache_cls: 缓存类
        my_custom_param: 自定义参数
    """
    # 参数验证
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID") is None:
        raise ValueError("mask_token_id 必须提供")
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))
    
    if stop_until_eos and eos_token_id is None:
        if os.environ.get("EOS_TOKEN_ID") is None:
            raise ValueError("stop_until_eos=True 时需要 eos_token_id")
        eos_token_id = int(os.environ.get("EOS_TOKEN_ID"))
    
    assert gen_length % block_length == 0, "gen_length 必须能被 block_length 整除"
    num_blocks = gen_length // block_length
    
    # 创建初始帧
    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    
    # 准备注意力掩码
    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()
    if attention_mask is not None and attention_mask.shape == input_ids.shape:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(model.device)
    
    # 初始化缓存
    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame
    deltas = []
    
    # 定义 unmasking 函数
    def unmasking_fn(
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        return my_unmasking(
            active_seq_idx=active_seq_idx,
            scores=scores,
            probs=probs,
            transfer_index_mask=transfer_index_mask,
            block_mask=block_mask,
            num_transfer_tokens=num_transfer_tokens,
        ), {}
    
    # 主生成循环
    for block_idx in range(num_blocks):
        # 创建块掩码
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[:, block_idx * block_length : (block_idx + 1) * block_length] = True
        
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
        
        block_deltas = []
        while True:
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            
            # 执行生成步骤
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=num_transfer_tokens,
                unmasking_fn=unmasking_fn,
                attention_mask=attention_mask,
                past_key_values=cache,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                stop_until_eos=stop_until_eos,
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

### 步骤 4: 导出策略

在 `src/generation/__init__.py` 中添加导入：

```python
from src.generation.my_strategy import my_strategy_generate
```

---

## 必须实现的方法

### 核心组件

| 组件 | 作用 | 必需性 |
|------|------|--------|
| `unmasking_fn` | 决定每步解码哪些 token | ⭐⭐⭐⭐⭐ |
| 主生成函数 | 实现完整的生成循环 | ⭐⭐⭐⭐⭐ |
| `@register` 装饰器 | 注册策略名称 | ⭐⭐⭐⭐ |

### unmasking_fn 详细说明

unmasking 函数是解码策略的核心，决定每步选择哪些位置进行解码：

```python
def unmasking_fn(
    active_seq_idx: torch.Tensor,    # 活跃序列索引
    scores: torch.Tensor,             # 置信度分数 [B, gen_length]
    probs: torch.Tensor,              # 概率分布 [B, gen_length, vocab_size]
    transfer_index_mask: torch.Tensor,# 可传输位置掩码
    block_mask: torch.Tensor,         # 当前块掩码
    num_transfer_tokens: int,         # 最小传输数
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """
    返回:
        - 传输索引元组：每个元素是一个 1D 张量，包含该序列要解码的位置
        - 额外信息字典：可存储任何额外信息
    """
    ...
```

### 主生成函数签名

```python
@register("strategy_name")
def strategy_generate(
    model,                    # 模型实例
    input_ids: torch.Tensor,  # 输入 ID [B, prompt_len]
    # 基本参数
    gen_length: int = 128,
    block_length: int = 32,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    # Token ID
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    # 控制参数
    stop_until_eos: bool = False,
    output_hidden_states: bool = False,
    output_probs: bool = False,
    # 缓存
    cache_cls: Type[dCache] | None = None,
    # 自定义参数
    **kwargs,
) -> DecodeRecord:
    ...
```

---

## 关键代码模板

### 模板 1: 自回归解码策略

```python
@register("autoregressive")
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    num_transfer_tokens: int = 1,
    **kwargs,
) -> DecodeRecord:
    """自回归解码：从左到右逐个生成"""
    
    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_token_id
    ).to(device=model.device, dtype=model.dtype)
    
    frame = initial_frame
    deltas = []
    
    def unmasking_fn(active_seq_idx, scores, probs, transfer_index_mask, 
                     block_mask, num_transfer_tokens):
        batch_size = scores.shape[0]
        device = scores.device
        transfer_index = []
        
        for i in range(batch_size):
            # 只选择第一个掩码位置
            mask_positions = torch.where(transfer_index_mask[i] & block_mask[i])[0]
            if len(mask_positions) > 0:
                transfer_index.append(mask_positions[:num_transfer_tokens])
            else:
                transfer_index.append(torch.tensor([], dtype=torch.long, device=device))
        
        return tuple(transfer_index), {}
    
    # 单块生成
    block_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    
    while True:
        delta = generate_step(
            model=model,
            frame=frame,
            block_mask=block_mask,
            num_transfer_tokens=num_transfer_tokens,
            unmasking_fn=unmasking_fn,
            **kwargs,
        )
        if delta is None:
            break
        deltas.append(delta.to("cpu"))
        frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=initial_frame.to("cpu"), deltas=deltas)
```

### 模板 2: 并行解码策略

```python
@register("parallel")
def parallel_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    threshold: float = 0.9,  # 并行解码阈值
    **kwargs,
) -> DecodeRecord:
    """并行解码：同时解码所有高置信度 token"""
    
    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_token_id
    ).to(device=model.device, dtype=model.dtype)
    
    frame = initial_frame
    deltas = []
    
    def unmasking_fn(active_seq_idx, scores, probs, transfer_index_mask,
                     block_mask, num_transfer_tokens):
        batch_size = scores.shape[0]
        device = scores.device
        transfer_index = []
        
        for i in range(batch_size):
            valid_mask = transfer_index_mask[i] & block_mask[i]
            valid_scores = torch.where(valid_mask, scores[i], -torch.inf)
            
            # 选择所有超过阈值的 token
            high_conf_mask = valid_scores >= threshold
            if high_conf_mask.any():
                indices = torch.where(high_conf_mask)[0]
                transfer_index.append(indices)
            else:
                # 如果没有超过阈值的，选择最高的一个
                best_idx = valid_scores.argmax().unsqueeze(0)
                transfer_index.append(best_idx)
        
        return tuple(transfer_index), {}
    
    block_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    
    while True:
        delta = generate_step(
            model=model,
            frame=frame,
            block_mask=block_mask,
            num_transfer_tokens=1,  # 最小传输数
            unmasking_fn=unmasking_fn,
            **kwargs,
        )
        if delta is None:
            break
        deltas.append(delta.to("cpu"))
        frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=initial_frame.to("cpu"), deltas=deltas)
```

### 模板 3: 迭代细化策略

```python
@register("iterative_refinement")
def iterative_refinement_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    num_iterations: int = 10,
    refinement_ratio: float = 0.1,  # 每次细化 10% 的 token
    **kwargs,
) -> DecodeRecord:
    """迭代细化：多次细化低置信度 token"""
    
    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_token_id
    ).to(device=model.device, dtype=model.dtype)
    
    frame = initial_frame
    deltas = []
    
    def unmasking_fn(active_seq_idx, scores, probs, transfer_index_mask,
                     block_mask, num_transfer_tokens):
        batch_size = scores.shape[0]
        device = scores.device
        transfer_index = []
        
        for i in range(batch_size):
            valid_mask = transfer_index_mask[i] & block_mask[i]
            valid_scores = torch.where(valid_mask, scores[i], -torch.inf)
            
            # 选择最低置信度的 token 进行细化
            num_to_refine = max(1, int(valid_mask.sum().item() * refinement_ratio))
            _, indices = torch.topk(valid_scores, num_to_refine, largest=False)
            transfer_index.append(indices)
        
        return tuple(transfer_index), {}
    
    block_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    
    # 首先完成初始生成
    while True:
        delta = generate_step(
            model=model,
            frame=frame,
            block_mask=block_mask,
            num_transfer_tokens=1,
            unmasking_fn=lambda **kwargs: (tuple(
                torch.where(kwargs['transfer_index_mask'][i] & kwargs['block_mask'][i])[0][:1]
                for i in range(kwargs['scores'].shape[0])
            ), {}),
            **kwargs,
        )
        if delta is None:
            break
        deltas.append(delta.to("cpu"))
        frame = frame.apply_delta(delta)
    
    # 迭代细化
    for iteration in range(num_iterations):
        # 重新评估所有 token
        delta = generate_step(
            model=model,
            frame=frame,
            block_mask=block_mask,
            num_transfer_tokens=1,
            unmasking_fn=unmasking_fn,
            **kwargs,
        )
        if delta is not None:
            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=initial_frame.to("cpu"), deltas=deltas)
```

### 模板 4: 块级解码策略

```python
@register("block_wise")
def block_wise_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 32,
    tokens_per_step: int = 4,  # 每步解码的 token 数
    **kwargs,
) -> DecodeRecord:
    """块级解码：按块进行解码"""
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_token_id
    ).to(device=model.device, dtype=model.dtype)
    
    frame = initial_frame
    deltas = []
    
    def unmasking_fn(active_seq_idx, scores, probs, transfer_index_mask,
                     block_mask, num_transfer_tokens):
        batch_size = scores.shape[0]
        device = scores.device
        transfer_index = []
        
        for i in range(batch_size):
            valid_mask = transfer_index_mask[i] & block_mask[i]
            valid_scores = torch.where(valid_mask, scores[i], -torch.inf)
            
            # 选择 top-k 高置信度 token
            k = min(tokens_per_step, valid_mask.sum().int().item())
            if k > 0:
                _, indices = torch.topk(valid_scores, k)
                transfer_index.append(indices)
            else:
                transfer_index.append(torch.tensor([], dtype=torch.long, device=device))
        
        return tuple(transfer_index), {}
    
    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[:, block_idx * block_length : (block_idx + 1) * block_length] = True
        
        while True:
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=tokens_per_step,
                unmasking_fn=unmasking_fn,
                **kwargs,
            )
            if delta is None:
                break
            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)
    
    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )
```

---

## 配置文件编写

### 创建 YAML 配置文件

在 `configs/generation/` 目录下创建配置文件，例如 `my_strategy.yaml`：

```yaml
# configs/generation/my_strategy.yaml

# 策略名称（必须与 @register 中的名称匹配）
strategy: my_strategy

# 基本参数
gen_length: 128
block_length: 32
num_transfer_tokens: 1

# 采样参数
temperature: 0.0
top_k: null
top_p: null

# 停止条件
stop_until_eos: false

# 自定义参数
my_custom_param: 0.5
threshold: 0.9

# 输出选项
output_hidden_states: false
output_probs: false
```

### 配置参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `strategy` | str | 策略名称（必须与注册名称匹配） | 必需 |
| `gen_length` | int | 生成长度 | 128 |
| `block_length` | int | 块长度 | 32 |
| `num_transfer_tokens` | int | 每步最小传输 token 数 | 1 |
| `temperature` | float | 采样温度 | 0.0 |
| `top_k` | int | top-k 过滤 | null |
| `top_p` | float | nucleus 采样阈值 | null |
| `stop_until_eos` | bool | 是否在 EOS 处停止 | false |

### 使用配置文件

```bash
# 命令行指定策略
python eval.py generation=my_strategy ...

# 覆盖参数
python eval.py generation=my_strategy gen_length=256 temperature=0.8 ...
```

### 高级配置

```yaml
# configs/generation/advanced_strategy.yaml

strategy: advanced

# 并行解码参数
threshold: 0.9        # 置信度阈值
factor: null          # factor-based 并行解码

# EB sampler 参数
gamma: null           # 熵边界阈值

# PC sampler 参数
debias: false         # 是否去偏
clip_alpha: null      # 裁剪 alpha

# 确定性先验
sigma: null           # 高斯核标准差

# 算法选择
alg: "maskgit_plus"   # 置信度算法
```

---

## 测试和验证

### 单元测试

创建测试文件 `tests/test_my_strategy.py`：

```python
import pytest
import torch

from src.generation.my_strategy import my_strategy_generate, my_unmasking


class TestMyStrategy:
    
    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')
                self.dtype = torch.float32
                self.config = type('Config', (), {'vocab_size': 1000})()
            
            def __call__(self, x, **kwargs):
                batch_size, seq_len = x.shape
                vocab_size = self.config.vocab_size
                return type('Output', (), {
                    'logits': torch.randn(batch_size, seq_len, vocab_size),
                    'hidden_states': None,
                })()
        
        return MockModel()
    
    def test_unmasking_function(self):
        """测试 unmasking 函数"""
        batch_size, gen_length = 2, 10
        
        scores = torch.rand(batch_size, gen_length)
        probs = torch.rand(batch_size, gen_length, 100)
        transfer_index_mask = torch.ones(batch_size, gen_length, dtype=torch.bool)
        block_mask = torch.ones(batch_size, gen_length, dtype=torch.bool)
        
        transfer_index, extra = my_unmasking(
            active_seq_idx=torch.arange(batch_size),
            scores=scores,
            probs=probs,
            transfer_index_mask=transfer_index_mask,
            block_mask=block_mask,
            num_transfer_tokens=2,
        )
        
        assert len(transfer_index) == batch_size
        for idx in transfer_index:
            assert idx.dim() == 1
            assert idx.numel() <= 2
    
    def test_generation(self, mock_model):
        """测试完整生成流程"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        result = my_strategy_generate(
            model=mock_model,
            input_ids=input_ids,
            gen_length=16,
            block_length=16,
            num_transfer_tokens=1,
            mask_token_id=0,
        )
        
        assert result.gen_length == 16
        assert len(result.deltas) > 0
        assert result[-1].generated_tokens.shape == (16,)
    
    def test_batch_generation(self, mock_model):
        """测试批量生成"""
        batch_size = 3
        input_ids = torch.randint(1, 100, (batch_size, 5))
        
        result = my_strategy_generate(
            model=mock_model,
            input_ids=input_ids,
            gen_length=16,
            block_length=16,
            mask_token_id=0,
        )
        
        assert result.gen_length == 16
        assert result[-1].generated_tokens.shape == (batch_size, 16)
```

### 集成测试

```python
def test_strategy_with_real_model():
    """使用真实模型测试策略"""
    from src.models import load_model
    from src.generation.my_strategy import my_strategy_generate
    
    model = load_model("path/to/model")
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    result = my_strategy_generate(
        model=model,
        input_ids=input_ids,
        gen_length=32,
        block_length=32,
        mask_token_id=model.config.mask_token_id,
    )
    
    # 验证生成结果
    final_frame = result[-1]
    assert final_frame.generated_tokens.shape == (32,)
    
    # 验证没有掩码 token 剩余
    assert not (final_frame.generated_tokens == model.config.mask_token_id).any()
```

### 性能基准测试

```python
import time

def benchmark_strategy():
    """性能基准测试"""
    from src.models import load_model
    from src.generation.vanilla import vanilla_generate
    from src.generation.my_strategy import my_strategy_generate
    
    model = load_model("path/to/model")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    # 测试 vanilla
    start = time.time()
    vanilla_result = vanilla_generate(
        model=model, input_ids=input_ids, gen_length=128
    )
    vanilla_time = time.time() - start
    
    # 测试自定义策略
    start = time.time()
    custom_result = my_strategy_generate(
        model=model, input_ids=input_ids, gen_length=128
    )
    custom_time = time.time() - start
    
    print(f"Vanilla: {vanilla_time:.2f}s, {len(vanilla_result.deltas)} steps")
    print(f"Custom: {custom_time:.2f}s, {len(custom_result.deltas)} steps")
    print(f"Speedup: {vanilla_time / custom_time:.2f}x")
```

### 验证检查清单

- [ ] unmasking 函数返回正确格式
- [ ] 生成循环正确终止
- [ ] DecodeRecord 包含完整的生成历史
- [ ] 批量生成正确处理
- [ ] EOS 停止条件正确工作
- [ ] 缓存集成无错误
- [ ] 内存使用合理
- [ ] 性能符合预期

---

## 最佳实践

### 1. 参数验证

```python
@register("robust_strategy")
def robust_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 32,
    **kwargs,
):
    # 验证输入
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids 必须是 2D，但得到 {input_ids.dim()}D")
    
    if gen_length <= 0:
        raise ValueError(f"gen_length 必须为正数，但得到 {gen_length}")
    
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) 必须能被 block_length ({block_length}) 整除"
        )
    
    # ... 继续实现 ...
```

### 2. 错误处理

```python
def safe_generate_step(model, frame, **kwargs):
    """安全的生成步骤，带错误处理"""
    try:
        delta = generate_step(model=model, frame=frame, **kwargs)
        return delta
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # 清理缓存并重试
            torch.cuda.empty_cache()
            return generate_step(model=model, frame=frame, **kwargs)
        raise
```

### 3. 进度跟踪

```python
from tqdm import tqdm

@register("tracked_strategy")
def tracked_generate(model, input_ids, gen_length, **kwargs):
    """带进度跟踪的生成"""
    frame = Frame.create_initial_frame(input_ids, gen_length=gen_length, ...)
    deltas = []
    
    with tqdm(total=gen_length, desc="Generating") as pbar:
        while True:
            delta = generate_step(model=model, frame=frame, **kwargs)
            if delta is None:
                break
            
            # 更新进度条
            num_new_tokens = sum(t.numel() for t in delta.transfer_index)
            pbar.update(num_new_tokens)
            
            deltas.append(delta)
            frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=..., deltas=deltas)
```

### 4. 日志记录

```python
import logging

logger = logging.getLogger(__name__)

@register("logged_strategy")
def logged_generate(model, input_ids, **kwargs):
    """带日志记录的生成"""
    logger.info(f"开始生成: input_shape={input_ids.shape}, kwargs={kwargs}")
    
    frame = Frame.create_initial_frame(input_ids, ...)
    deltas = []
    step = 0
    
    while True:
        step += 1
        delta = generate_step(model=model, frame=frame, **kwargs)
        
        if delta is None:
            logger.info(f"生成完成: {step} 步")
            break
        
        num_tokens = sum(t.numel() for t in delta.transfer_index)
        logger.debug(f"步骤 {step}: 解码 {num_tokens} 个 token")
        
        deltas.append(delta)
        frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=..., deltas=deltas)
```

### 5. 内存优化

```python
@register("memory_efficient_strategy")
def memory_efficient_generate(model, input_ids, gen_length, **kwargs):
    """内存优化的生成"""
    frame = Frame.create_initial_frame(input_ids, gen_length=gen_length, ...)
    
    # 使用生成器避免存储所有 delta
    def generate_deltas():
        while True:
            delta = generate_step(model=model, frame=frame, **kwargs)
            if delta is None:
                break
            yield delta.to("cpu")  # 立即移到 CPU
            frame = frame.apply_delta(delta)
    
    # 只保留最终帧
    deltas = list(generate_deltas())
    
    return DecodeRecord(initial_frame=..., deltas=deltas)
```

### 6. 可配置性

```python
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    """策略配置"""
    gen_length: int = 128
    block_length: int = 32
    num_transfer_tokens: int = 1
    temperature: float = 0.0
    threshold: float | None = None
    
    def validate(self):
        if self.gen_length % self.block_length != 0:
            raise ValueError("gen_length 必须能被 block_length 整除")
        if self.temperature < 0:
            raise ValueError("temperature 不能为负数")

@register("configurable_strategy")
def configurable_generate(model, input_ids, **kwargs):
    """可配置的生成策略"""
    config = StrategyConfig(**kwargs)
    config.validate()
    
    # 使用配置进行生成
    ...
```

---

## 常见问题

### Q: 如何实现自定义采样策略？

A: 在 unmasking 函数中使用自定义采样逻辑：

```python
def custom_sampling_unmasking(active_seq_idx, scores, probs, ...):
    # 使用不同的采样策略
    if sampling_method == "nucleus":
        # Nucleus 采样
        ...
    elif sampling_method == "top_k":
        # Top-k 采样
        ...
    elif sampling_method == "temperature":
        # 温度采样
        ...
    
    return transfer_index, {}
```

### Q: 如何处理提前终止？

A: 使用 `stop_until_eos` 参数或在 unmasking 函数中检查：

```python
def unmasking_fn(active_seq_idx, scores, probs, transfer_index_mask, 
                 block_mask, num_transfer_tokens):
    # 检查是否遇到 EOS
    eos_positions = (decoded_tokens == eos_token_id)
    
    # 只选择 EOS 之前的 token
    transfer_index = []
    for i in range(batch_size):
        first_eos = eos_positions[i].int().argmax()
        valid_mask = transfer_index_mask[i] & block_mask[i]
        valid_mask[first_eos:] = False
        ...
    
    return tuple(transfer_index), {}
```

### Q: 如何实现多阶段解码？

A: 在主生成函数中实现多个阶段：

```python
@register("multi_stage")
def multi_stage_generate(model, input_ids, **kwargs):
    # 阶段 1: 快速生成
    frame = initial_frame
    while mask_ratio > 0.5:
        delta = generate_step(..., unmasking_fn=fast_unmasking)
        frame = frame.apply_delta(delta)
    
    # 阶段 2: 细化
    while mask_ratio > 0:
        delta = generate_step(..., unmasking_fn=refine_unmasking)
        frame = frame.apply_delta(delta)
    
    # 阶段 3: 最终润色
    for _ in range(num_refinement_steps):
        delta = generate_step(..., unmasking_fn=polish_unmasking)
        frame = frame.apply_delta(delta)
    
    return DecodeRecord(...)
```

### Q: 如何调试解码策略？

A: 使用详细的日志和可视化：

```python
def debug_unmasking(active_seq_idx, scores, probs, ...):
    print(f"Active sequences: {active_seq_idx}")
    print(f"Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Transferable positions: {transfer_index_mask.sum()}")
    
    transfer_index, extra = my_unmasking(...)
    
    print(f"Selected {sum(t.numel() for t in transfer_index)} positions")
    return transfer_index, extra
```

---

## 参考资源

- [基础解码策略实现](file:///Users/lier/codes/d2Cache/src/generation/vanilla.py)
- [自回归策略实现](file:///Users/lier/codes/d2Cache/src/generation/ar.py)
- [解码工具函数](file:///Users/lier/codes/d2Cache/src/generation/utils.py)
- [Frame 和 FrameDelta 定义](file:///Users/lier/codes/d2Cache/src/frame.py)
- [自定义指南](file:///Users/lier/codes/d2Cache/docs/customization.md)
