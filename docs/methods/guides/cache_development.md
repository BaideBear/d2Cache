# 缓存扩展开发指南

本指南详细说明如何为 d2Cache 项目开发自定义缓存机制。缓存机制在扩散语言模型解码过程中用于重用中间状态，可以显著提升生成效率。

## 目录

- [概述](#概述)
- [实现步骤](#实现步骤)
- [必须重写的方法](#必须重写的方法)
- [关键代码模板](#关键代码模板)
- [配置文件编写](#配置文件编写)
- [测试和验证](#测试和验证)
- [最佳实践](#最佳实践)

---

## 概述

d2Cache 框架通过抽象基类 `dCache` 定义了缓存接口。开发者需要继承此基类并实现特定的缓存行为。缓存机制通过 Python 上下文管理器拦截和修改模型层内的计算流程。

### 核心概念

- **AttentionContext**: 注意力计算上下文，存储 Query、Key、Value 状态
- **FFNContext**: 前馈网络上下文，存储 FFN 输入输出
- **ModelForwardContext**: 模型前向传播上下文，存储输入嵌入和输出 logits
- **Frame/FrameDelta**: 生成帧和帧变化，跟踪解码过程

---

## 实现步骤

### 步骤 1: 创建缓存类文件

在 `src/cache/` 目录下创建新的 Python 文件，例如 `my_cache.py`：

```python
import torch
import torch.nn as nn
from contextlib import contextmanager

from src.cache.base import dCache, AttentionContext, FFNContext
from src.frame import Frame, FrameDelta


class MyCache(dCache):
    """自定义缓存实现"""
    
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config)
        # 初始化缓存存储结构
        ...
```

### 步骤 2: 实现初始化方法

```python
def __init__(self, model_config, custom_param=0.5):
    super().__init__(model_config)
    
    # 存储模型配置
    self.model_config = model_config
    
    # 初始化 Key/Value 缓存列表
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    
    # 存储自定义参数
    self.custom_param = custom_param
    
    # 初始化其他内部状态
    self._internal_state: torch.Tensor | None = None
```

### 步骤 3: 实现 model_forward 上下文管理器

```python
@contextmanager
def model_forward(self, x: torch.Tensor):
    """
    模型前向传播的上下文管理器。
    可以修改输入嵌入或准备全局掩码。
    
    Args:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
    """
    input_shape = x.shape
    ctx = ModelForwardContext(x=x)
    
    # 可选：选择输入的子集
    if self._select_mask is not None:
        ctx.x = x[self._select_mask]
    
    yield ctx
    
    # 验证 logits 形状
    if ctx.logits is None:
        raise RuntimeError("logits 未在上下文中设置")
    
    if ctx.logits.shape[:2] != input_shape[:2]:
        raise RuntimeError(
            f"logits 形状 {ctx.logits.shape} 与输入形状 {input_shape} 不兼容"
        )
```

### 步骤 4: 实现 attention 上下文管理器

```python
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
    """
    注意力计算的上下文管理器。
    这是最关键的缓存实现方法。
    
    Args:
        layer_idx: 层索引
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        attn_norm: 层归一化模块
        q_proj, k_proj, v_proj: Query/Key/Value 投影层
        attention_mask: 可选的注意力掩码
        position_ids: 可选的位置 ID
    """
    residual = x
    x = attn_norm(x)
    
    # 计算 Q、K、V 投影
    if x.numel() > 0:
        q, k, v = q_proj(x), k_proj(x), v_proj(x)
    else:
        q, k, v = x[:, 0:0], x[:, 0:0], x[:, 0:0]
    
    # 缓存管理逻辑
    if len(self.key_cache) <= layer_idx:
        # 首次前向传播，存储状态
        self.key_cache.append(k)
        self.value_cache.append(v)
    else:
        # 后续传播，更新缓存
        self.key_cache[layer_idx] = self._update_cache(
            self.key_cache[layer_idx], k
        )
        self.value_cache[layer_idx] = self._update_cache(
            self.value_cache[layer_idx], v
        )
        # 使用缓存的 K、V
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
    
    # 创建并返回注意力上下文
    ctx = AttentionContext(
        q=q,
        k=k,
        v=v,
        residual=residual,
        attention_mask=AttentionContext.convert_attention_mask(
            attention_mask,
            dtype=q.dtype,
            query_length=q.shape[1],
            key_value_length=k.shape[1],
        ),
        q_position_ids=position_ids,
        kv_position_ids=position_ids,
    )
    
    yield ctx
    
    # 验证输出
    if ctx.o is None:
        raise RuntimeError("注意力输出未在上下文中设置")
```

### 步骤 5: 实现生命周期钩子

```python
def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
    """
    每个生成步骤开始时调用。
    用于准备基于当前生成状态的掩码。
    
    Args:
        block_mask: 布尔掩码，指示块中哪些位置是活跃的
        frame: 应用 delta 之前的帧
    """
    # 准备下一步的查询掩码
    ...

def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
    """
    每个生成步骤结束时调用。
    用于用新生成的信息更新缓存。
    
    Args:
        block_mask: 布尔掩码，指示块中哪些位置是活跃的
        frame: 应用 delta 之前的帧
        delta: 要应用到帧的 delta
    """
    # 更新置信度分数或密度度量
    ...

def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
    """
    每个块开始时调用。
    """
    ...

def on_block_end(self, block_mask: torch.Tensor, frame: Frame, deltas: list[FrameDelta]):
    """
    每个块结束时调用。
    """
    ...
```

---

## 必须重写的方法

### 核心方法（必须实现）

| 方法 | 作用 | 重要性 |
|------|------|--------|
| `__init__` | 初始化缓存实例，设置内部存储结构 | ⭐⭐⭐ |
| `attention` | 拦截注意力计算，管理 K/V 缓存 | ⭐⭐⭐⭐⭐ |
| `model_forward` | 修改模型前向传播的输入输出 | ⭐⭐⭐ |

### 生命周期钩子（按需实现）

| 方法 | 作用 | 调用时机 |
|------|------|----------|
| `on_step_start` | 准备步骤状态 | 每个生成步骤开始前 |
| `on_step_end` | 更新缓存状态 | 每个生成步骤结束后 |
| `on_block_start` | 准备块状态 | 每个块开始前 |
| `on_block_end` | 处理块结果 | 每个块结束后 |

### 方法详细说明

#### `attention` 方法

这是最关键的缓存实现方法。其职责包括：

1. **计算投影**: 执行 Query、Key、Value 投影
2. **缓存管理**: 存储或更新 K/V 缓存
3. **掩码处理**: 准备注意力掩码
4. **位置编码**: 处理位置 ID

```python
@contextmanager
def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj, 
              attention_mask=None, position_ids=None):
    # 1. 保存残差连接
    residual = x
    
    # 2. 层归一化
    x = attn_norm(x)
    
    # 3. 计算 Q、K、V
    q, k, v = q_proj(x), k_proj(x), v_proj(x)
    
    # 4. 缓存逻辑（核心）
    # ... 自定义缓存实现 ...
    
    # 5. 创建上下文
    ctx = AttentionContext(q=q, k=k, v=v, residual=residual, ...)
    
    yield ctx
    
    # 6. 验证输出
    assert ctx.o is not None
```

#### `model_forward` 方法

用于控制整个模型的前向传播：

```python
@contextmanager
def model_forward(self, x: torch.Tensor):
    input_shape = x.shape
    ctx = ModelForwardContext(x=x)
    
    # 可选：选择输入子集以提高效率
    if self.active_q_mask is not None:
        ctx.x = x[self.active_q_mask].view(batch_size, -1, hidden_dim)
    
    yield ctx
    
    # 恢复完整形状的 logits
    if self.active_q_mask is not None:
        ctx.logits = self._restore_logits(ctx.logits, input_shape)
```

---

## 关键代码模板

### 完整的最小缓存实现

```python
import torch
import torch.nn as nn
from contextlib import contextmanager

from src.cache.base import dCache, AttentionContext
from src.frame import Frame, FrameDelta


class MinimalKVCache(dCache):
    """最小化的 KV 缓存实现"""
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
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
        residual = x
        x = attn_norm(x)
        
        if x.numel() > 0:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)
        else:
            q, k, v = x[:, 0:0], x[:, 0:0], x[:, 0:0]
        
        # 首次传播：存储 K/V
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k)
            self.value_cache.append(v)
        else:
            # 后续传播：使用缓存的 K/V
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
        
        ctx = AttentionContext(
            q=q, k=k, v=v, residual=residual,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask, dtype=q.dtype,
                query_length=q.shape[1], key_value_length=k.shape[1]
            ),
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
        )
        yield ctx
        
        if ctx.o is None:
            raise RuntimeError("注意力输出未设置")
```

### 带选择性更新的缓存实现

```python
class SelectiveUpdateCache(dCache):
    """选择性更新缓存 - 只更新特定位置的缓存"""
    
    def __init__(self, model_config, update_threshold: float = 0.8):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.update_threshold = update_threshold
        self._confidence_cache: torch.Tensor | None = None
    
    @contextmanager
    def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
                  attention_mask=None, position_ids=None):
        residual = x
        x = attn_norm(x)
        
        q, k, v = q_proj(x), k_proj(x), v_proj(x)
        
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k)
            self.value_cache.append(v)
        else:
            # 只更新高置信度位置的缓存
            if self.active_q_mask is not None:
                active_indices = self.active_q_mask.nonzero(as_tuple=False)
                self.key_cache[layer_idx][active_indices] = k.flatten(0, 1)
                self.value_cache[layer_idx][active_indices] = v.flatten(0, 1)
            
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
        
        ctx = AttentionContext(
            q=q, k=k, v=v, residual=residual,
            attention_mask=AttentionContext.convert_attention_mask(
                attention_mask, dtype=q.dtype,
                query_length=q.shape[1], key_value_length=k.shape[1]
            ),
            q_position_ids=position_ids,
            kv_position_ids=position_ids,
        )
        yield ctx
        
        assert ctx.o is not None
    
    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """更新置信度缓存"""
        if delta.confidence is not None:
            if self._confidence_cache is None:
                self._confidence_cache = delta.confidence
            else:
                # 更新活跃位置的置信度
                active_mask = self.active_seq_mask
                self._confidence_cache[active_mask] = delta.confidence
```

### 带注意力滚动的缓存实现

```python
class AttentionRolloutCache(dCache):
    """带注意力滚动的缓存 - 累积注意力权重用于重要性计算"""
    
    def __init__(self, model_config, rollout_p: float = 0.1):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.rollout_p = rollout_p
        self._attn_rollout: torch.Tensor | None = None
    
    @contextmanager
    def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
                  attention_mask=None, position_ids=None):
        with super().attention(
            layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
            attention_mask, position_ids
        ) as ctx:
            # 初始化注意力滚动矩阵
            if layer_idx == 0:
                seq_len = ctx.k.shape[1]
                self._attn_rollout = torch.eye(
                    seq_len, device=x.device, dtype=x.dtype
                ).expand(x.size(0), -1, -1)
            
            # 累积注意力权重
            if ctx.attn_weight is not None:
                self._accumulate_rollout(ctx.attn_weight)
            
            yield ctx
    
    def _accumulate_rollout(self, attn_scores: torch.Tensor):
        """累积注意力滚动"""
        B, n_heads, q_len, seq_len = attn_scores.shape
        device, dtype = attn_scores.device, attn_scores.dtype
        
        # 计算有效注意力
        effective_attn = attn_scores.mean(dim=1)
        
        # 添加残差连接
        residual_attn = effective_attn + torch.eye(seq_len, device=device, dtype=dtype)
        residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)
        
        # 累积
        self._attn_rollout = residual_attn @ self._attn_rollout
    
    def get_importance_scores(self) -> torch.Tensor:
        """获取基于注意力滚动的重要性分数"""
        if self._attn_rollout is None:
            raise RuntimeError("注意力滚动尚未初始化")
        return self._attn_rollout.sum(dim=1)
```

---

## 配置文件编写

### 创建 YAML 配置文件

在 `configs/cache/` 目录下创建配置文件，例如 `my_cache.yaml`：

```yaml
# configs/cache/my_cache.yaml

# 目标类路径（必需）
_target_: src.cache.MyCache

# 自定义参数
custom_param: 0.5
update_threshold: 0.8
rollout_p: 0.1

# 其他参数
enable_logging: true
cache_size_limit: null
```

### 配置参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `_target_` | str | 缓存类的完整导入路径 | 必需 |
| `custom_param` | float | 自定义参数示例 | 0.5 |
| `update_threshold` | float | 更新阈值 | 0.8 |
| `rollout_p` | float | 滚动概率 | 0.1 |

### 使用配置文件

```bash
# 命令行指定缓存配置
python eval.py cache=my_cache generation=vanilla ...

# 或在代码中加载
from omegaconf import OmegaConf
import hydra

cfg = OmegaConf.load("configs/cache/my_cache.yaml")
cache = hydra.utils.instantiate(cfg, model_config=model.config)
```

---

## 测试和验证

### 单元测试

创建测试文件 `tests/test_my_cache.py`：

```python
import pytest
import torch
import torch.nn as nn

from src.cache import MyCache
from src.frame import Frame


class TestMyCache:
    
    @pytest.fixture
    def model_config(self):
        """模拟模型配置"""
        return type('Config', (), {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
        })()
    
    @pytest.fixture
    def cache(self, model_config):
        """创建缓存实例"""
        return MyCache(model_config, custom_param=0.5)
    
    def test_initialization(self, cache):
        """测试初始化"""
        assert cache.key_cache == []
        assert cache.value_cache == []
        assert cache.custom_param == 0.5
    
    def test_attention_context(self, cache, model_config):
        """测试注意力上下文"""
        batch_size, seq_len, hidden_size = 2, 10, model_config.hidden_size
        
        # 模拟输入
        x = torch.randn(batch_size, seq_len, hidden_size)
        attn_norm = nn.LayerNorm(hidden_size)
        q_proj = nn.Linear(hidden_size, hidden_size)
        k_proj = nn.Linear(hidden_size, hidden_size)
        v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 测试注意力上下文
        with cache.attention(0, x, attn_norm, q_proj, k_proj, v_proj) as ctx:
            assert ctx.q.shape == (batch_size, seq_len, hidden_size)
            assert ctx.k.shape == (batch_size, seq_len, hidden_size)
            assert ctx.v.shape == (batch_size, seq_len, hidden_size)
            ctx.o = torch.randn(batch_size, seq_len, hidden_size)
        
        # 验证缓存已存储
        assert len(cache.key_cache) == 1
        assert len(cache.value_cache) == 1
    
    def test_step_lifecycle(self, cache):
        """测试步骤生命周期"""
        batch_size, prompt_len, gen_len = 2, 5, 10
        
        # 创建模拟帧
        prompts = torch.randint(0, 100, (batch_size, prompt_len))
        frame = Frame.create_initial_frame(prompts, gen_length=gen_len, mask_token_id=0)
        
        block_mask = torch.ones(batch_size, gen_len, dtype=torch.bool)
        
        # 测试步骤开始
        cache.on_step_start(block_mask, frame)
        
        # 测试步骤结束
        from src.frame import FrameDelta
        delta = FrameDelta(
            transfer_index=(torch.tensor([0, 1]), torch.tensor([2])),
            decoded_tokens=torch.randint(0, 100, (batch_size, gen_len)),
        )
        cache.on_step_end(block_mask, frame, delta)
```

### 集成测试

```python
def test_cache_with_model():
    """测试缓存与模型的集成"""
    from src.models import load_model
    from src.cache import MyCache
    
    # 加载模型
    model = load_model("path/to/model")
    cache = MyCache(model.config)
    
    # 准备输入
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    # 运行生成
    from src.generation.vanilla import vanilla_generate
    result = vanilla_generate(
        model=model,
        input_ids=input_ids,
        cache_cls=MyCache,
        gen_length=32,
    )
    
    # 验证结果
    assert result.gen_length == 32
    assert len(result.deltas) > 0
```

### 验证检查清单

- [ ] 缓存初始化正确
- [ ] K/V 缓存形状正确
- [ ] 注意力上下文输出正确
- [ ] 步骤生命周期钩子被正确调用
- [ ] 与模型集成无错误
- [ ] 生成结果符合预期
- [ ] 内存使用合理
- [ ] 性能有提升

---

## 最佳实践

### 1. 内存管理

```python
class EfficientCache(dCache):
    def __init__(self, model_config, max_cache_length: int = 2048):
        super().__init__(model_config)
        self.max_cache_length = max_cache_length
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
    def _prune_cache(self):
        """修剪缓存以控制内存"""
        if len(self.key_cache) > 0:
            cache_len = self.key_cache[0].shape[1]
            if cache_len > self.max_cache_length:
                # 保留最近的缓存
                for i in range(len(self.key_cache)):
                    self.key_cache[i] = self.key_cache[i][:, -self.max_cache_length:]
                    self.value_cache[i] = self.value_cache[i][:, -self.max_cache_length:]
```

### 2. 错误处理

```python
@contextmanager
def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
              attention_mask=None, position_ids=None):
    try:
        # ... 实现逻辑 ...
        yield ctx
    except Exception as e:
        raise RuntimeError(f"注意力计算错误 (layer {layer_idx}): {e}") from e
    finally:
        # 清理资源
        ...
```

### 3. 日志记录

```python
import logging

class LoggedCache(dCache):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化缓存: {kwargs}")
    
    def on_step_end(self, block_mask, frame, delta):
        super().on_step_end(block_mask, frame, delta)
        self.logger.debug(
            f"步骤结束: 活跃位置={block_mask.sum()}, "
            f"传输位置={sum(t.numel() for t in delta.transfer_index)}"
        )
```

### 4. 性能优化

```python
class OptimizedCache(dCache):
    @torch.compile
    def _update_cache_fast(self, old_cache: torch.Tensor, new_values: torch.Tensor):
        """使用 torch.compile 加速缓存更新"""
        return old_cache.clone().scatter_(1, self._active_indices, new_values)
```

### 5. 配置验证

```python
from pydantic import BaseModel, validator

class CacheConfig(BaseModel):
    custom_param: float
    update_threshold: float = 0.8
    
    @validator('custom_param')
    def validate_custom_param(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('custom_param 必须在 [0, 1] 范围内')
        return v

class ValidatedCache(dCache):
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config)
        config = CacheConfig(**kwargs)
        self.custom_param = config.custom_param
        self.update_threshold = config.update_threshold
```

---

## 常见问题

### Q: 如何处理可变长度序列？

A: 使用注意力掩码和位置 ID 来处理：

```python
@contextmanager
def attention(self, layer_idx, x, ...):
    # ...
    ctx = AttentionContext(
        q=q, k=k, v=v, residual=residual,
        attention_mask=AttentionContext.convert_attention_mask(
            attention_mask, dtype=q.dtype,
            query_length=q.shape[1], key_value_length=k.shape[1]
        ),
        q_position_ids=position_ids,
        kv_position_ids=position_ids,
    )
    yield ctx
```

### Q: 如何实现缓存压缩？

A: 在 `on_step_end` 中实现压缩逻辑：

```python
def on_step_end(self, block_mask, frame, delta):
    # 压缩低重要性位置的缓存
    importance = self._compute_importance()
    keep_mask = importance > self.threshold
    self._compress_cache(keep_mask)
```

### Q: 如何调试缓存问题？

A: 使用详细的日志和断言：

```python
def attention(self, layer_idx, x, ...):
    with super().attention(...) as ctx:
        # 添加调试断言
        assert ctx.q.shape[0] == ctx.k.shape[0], "批次大小不匹配"
        assert not torch.isnan(ctx.q).any(), "Q 包含 NaN"
        yield ctx
```

---

## 参考资源

- [基础缓存类实现](file:///Users/lier/codes/d2Cache/src/cache/base.py)
- [d2Cache 实现示例](file:///Users/lier/codes/d2Cache/src/cache/d2cache.py)
- [前缀缓存实现](file:///Users/lier/codes/d2Cache/src/cache/prefix_cache.py)
- [自定义指南](file:///Users/lier/codes/d2Cache/docs/customization.md)
