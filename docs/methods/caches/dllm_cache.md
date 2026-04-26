# dLLM-Cache 缓存方法详解

## 算法逻辑精要

dLLM-Cache 在 PrefixCache 的基础之上引入了周期性刷新与自适应语义刷新两重机制，以更精细地权衡计算开销与生成质量。通过参数 `kp` 和 `kr` 分别控制 Prompt 部分和 Response 部分的周期性刷新频率；在无需周期性刷新的步骤中，算法计算当前 Value 与缓存的余弦相似度，根据 `rou` 比例选取相似度最低（即变化最大）的 Response 位置进行自适应重新计算。此外，dLLM-Cache 同时缓存了 Attention 输出和 FFN 输出，使得后续步骤可直接复用这些中间结果。其核心调度是：在 `attention` 上下文管理器中将 Prompt 与 Response 分离、选择性执行 Q/K/V 投影并基于余弦相似度决定刷新位置；在 `ffn` 上下文管理器中对齐刷新索引，仅处理需要更新的位置。

## 概述

dLLM-Cache 是一种专门为扩散语言模型（Diffusion Language Models）设计的高级缓存策略。它在 PrefixCache 的基础上增加了自适应刷新机制，能够根据预设的调度策略和语义相似度动态决定哪些位置需要重新计算，从而在保证生成质量的同时进一步提升计算效率。

## 算法原理和理论基础

### 核心创新

dLLM-Cache 的核心创新在于引入了两个关键机制：

1. **周期性刷新调度**：通过 `kp` 和 `kr` 参数控制 Prompt 和 Response 部分的刷新频率
2. **自适应语义刷新**：使用余弦相似度检测 Value 状态的变化，只刷新变化最大的位置

### 理论基础

在扩散语言模型的生成过程中，不同位置的语义重要性会随着生成步骤而变化。dLLM-Cache 基于以下观察：

1. **Prompt 稳定性**：Prompt 部分的表示相对稳定，不需要频繁刷新
2. **Response 动态性**：Response 部分的表示会随着生成进度变化，但变化程度不一
3. **语义漂移检测**：通过比较 Value 状态的余弦相似度，可以识别出需要刷新的位置

### 工作流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        dLLM-Cache 工作流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │ on_step_start   │                                                    │
│  │ 检查刷新调度     │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────┐                    │
│  │              刷新决策逻辑                        │                    │
│  ├─────────────────────────────────────────────────┤                    │
│  │  refresh_prompt = (step + 1) % kp == 0          │                    │
│  │  refresh_response = (step + 1) % kr == 0        │                    │
│  └────────┬────────────────────────────────────────┘                    │
│           │                                                             │
│     ┌─────┴─────┬──────────────────┐                                    │
│     ▼           ▼                  ▼                                    │
│ ┌───────┐  ┌──────────┐    ┌──────────────┐                             │
│ │刷新    │  │自适应刷新│    │完全复用缓存  │                             │
│ │Prompt │  │Response  │    │              │                             │
│ └───────┘  └──────────┘    └──────────────┘                             │
│     │           │                  │                                    │
│     │           ▼                  │                                    │
│     │    ┌──────────────┐          │                                    │
│     │    │计算余弦相似度│          │                                    │
│     │    │选择top-rou   │          │                                    │
│     │    │最低相似度位置│          │                                    │
│     │    └──────┬───────┘          │                                    │
│     │           │                  │                                    │
│     └───────────┴──────────────────┘                                    │
│                 │                                                       │
│                 ▼                                                       │
│        ┌────────────────┐                                               │
│        │ 更新 KV 缓存    │                                               │
│        │ 更新 Attn 缓存  │                                               │
│        │ 更新 FFN 缓存   │                                               │
│        └────────────────┘                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心数据结构和参数说明

### 类定义

```python
class dLLMCache(dCache):
    def __init__(self, model_config, kp: int = 50, kr: int = 2, rou: float = 0.25):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.attn_cache: list[torch.Tensor] = []
        self.ffn_cache: list[torch.Tensor] = []
        self.kp = kp
        self.kr = kr
        self.rou = rou
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_config` | dict | 必需 | 模型配置对象 |
| `kp` | int | 50 | Prompt 部分的刷新周期（每 kp 步刷新一次） |
| `kr` | int | 2 | Response 部分的刷新周期（每 kr 步刷新一次） |
| `rou` | float | 0.25 | 自适应刷新比例（0-1之间，表示刷新多少比例的 Response） |

### 核心数据结构

| 属性 | 类型 | 说明 |
|------|------|------|
| `key_cache` | list[torch.Tensor] | 每层 Key 状态缓存 |
| `value_cache` | list[torch.Tensor] | 每层 Value 状态缓存 |
| `attn_cache` | list[torch.Tensor] | 每层注意力输出缓存 |
| `ffn_cache` | list[torch.Tensor] | 每层 FFN 输出缓存 |
| `refresh_prompt` | bool | 当前步骤是否刷新 Prompt |
| `refresh_response` | bool | 当前步骤是否刷新 Response |
| `_refresh_index` | torch.Tensor | 需要刷新的位置索引 |
| `_prompt_length` | int | Prompt 的长度 |

### 参数调优建议

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         参数调优指南                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  kp (Prompt 刷新周期):                                                   │
│  ├── 较大值 (50-100): 适合长 Prompt，减少计算，但可能损失精度            │
│  ├── 中等值 (20-50): 平衡性能和精度，推荐默认值                          │
│  └── 较小值 (5-20): 高精度场景，但计算开销增加                           │
│                                                                         │
│  kr (Response 刷新周期):                                                 │
│  ├── 较大值 (5-10): 适合稳定生成，减少计算                               │
│  ├── 中等值 (2-5): 平衡性能和精度，推荐默认值                            │
│  └── 较小值 (1-2): 高精度场景，每步或隔步刷新                            │
│                                                                         │
│  rou (自适应刷新比例):                                                   │
│  ├── 较大值 (0.5-1.0): 刷新更多位置，精度更高但计算更多                  │
│  ├── 中等值 (0.2-0.5): 平衡性能和精度，推荐默认值                        │
│  └── 较小值 (0.0-0.2): 最小化计算，但可能影响生成质量                    │
│      └── rou=0 时完全禁用自适应刷新，只依赖周期性刷新                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 详细代码流程分析

以下按源码文件 [`src/cache/dllm_cache.py`](file:///Users/lier/codes/d2Cache/src/cache/dllm_cache.py) 的模块顺序，逐方法展开分析。

### `__init__` — 初始化

```python
# 源文件: src/cache/dllm_cache.py L11-L19
def __init__(self, model_config, kp: int = 50, kr: int = 2, rou: float = 0.25):
    super().__init__(model_config)
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    self.attn_cache: list[torch.Tensor] = []
    self.ffn_cache: list[torch.Tensor] = []
    self.kp = kp
    self.kr = kr
    self.rou = rou
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L11 | `def __init__(self, model_config, kp=50, kr=2, rou=0.25):` | 构造函数。`kp` 为 Prompt 刷新周期（每 kp 步刷新一次），`kr` 为 Response 刷新周期，`rou` 为自适应刷新比例（0~1）。 |
| L12 | `super().__init__(model_config)` | 调用父类 `dCache` 初始化基础属性。 |
| L13-L14 | `self.key_cache/value_cache = []` | KV 缓存列表，每个元素形状 `(B, T, head_dim)`，按层索引。 |
| L15 | `self.attn_cache: list[torch.Tensor] = []` | 注意力输出缓存列表，每个元素形状 `(B, T, head_dim)`，缓存各层注意力模块的完整输出。 |
| L16 | `self.ffn_cache: list[torch.Tensor] = []` | FFN 输出缓存列表，每个元素形状 `(B, T, C)`，缓存各层 FFN 模块的完整输出。 |
| L17-L19 | `self.kp/kr/rou = kp/kr/rou` | 保存刷新调度参数。`kp` 控制 Prompt 部分全量刷新的间隔；`kr` 控制 Response 部分全量刷新的间隔；`rou` 控制自适应刷新时选取 Response 中最低相似度位置的比例。 |

---

### `attention` — 注意力层上下文管理器

```python
# 源文件: src/cache/dllm_cache.py L21-L178
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
    refresh_prompt = self.refresh_prompt or layer_idx == 0
    refresh_response = self.refresh_response or layer_idx == 0
    residual = x
    x = attn_norm(x)
    x_prompt = x[:, : self._prompt_length]
    x_response = x[:, self._prompt_length :]
    x = x[:, 0:0]
    refresh_index = torch.tensor([], device=x.device, dtype=torch.long)
    if refresh_prompt:
        x = x_prompt
        refresh_index = torch.arange(self._prompt_length, device=x.device)

    if self.rou > 0 or refresh_response:
        x = torch.cat([x, x_response], dim=1)
        if refresh_response:
            refresh_index = torch.cat(
                [
                    refresh_index,
                    self._prompt_length
                    + torch.arange(x_response.size(1), device=x.device),
                ]
            )
    refresh_index = refresh_index.unsqueeze(0).expand(x.size(0), -1)

    B, T, C = x.shape
    q = torch.empty((B, 0, q_proj.out_features), dtype=x.dtype, device=x.device)
    k = torch.empty((B, 0, k_proj.out_features), dtype=x.dtype, device=x.device)
    v = torch.empty((B, 0, v_proj.out_features), dtype=x.dtype, device=x.device)
    if refresh_response or self.rou == 0 or len(self.key_cache) <= layer_idx:
        if x.numel() > 0:
            q, k, v = q_proj(x), k_proj(x), v_proj(x)
    else:
        if refresh_prompt:
            x_prompt = x[:, : self._prompt_length]
            x_response = x[:, self._prompt_length :]
            q, k, v = q_proj(x_prompt), k_proj(x_prompt), v_proj(x_prompt)
        else:
            x_response = x

        v_response = v_proj(x_response)
        num_replace = int(x_response.size(1) * self.rou)
        cos_sim = F.cosine_similarity(
            v_response,
            self.value_cache[layer_idx][
                self.active_seq_mask, self._prompt_length :
            ],
            dim=-1,
        )
        refresh_index_response = torch.topk(
            cos_sim, largest=False, k=num_replace
        ).indices

        selected_x_response = torch.gather(
            x_response, 1, refresh_index_response.unsqueeze(-1).expand(-1, -1, C)
        )
        q = torch.cat([q, q_proj(selected_x_response)], dim=1)
        k = torch.cat([k, k_proj(selected_x_response)], dim=1)
        v = torch.cat([v, v_response], dim=1)

    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(k)
        self.value_cache.append(v)
        q_position_ids = position_ids
    else:
        if refresh_prompt:
            self.key_cache[layer_idx][
                self.active_seq_mask, : self._prompt_length
            ] = k[:, : self._prompt_length]
            self.value_cache[layer_idx][
                self.active_seq_mask, : self._prompt_length
            ] = v[:, : self._prompt_length]
            prompt_offset = self._prompt_length
        else:
            prompt_offset = 0

        q_position_ids = (
            position_ids[:, :prompt_offset] if position_ids is not None else None
        )

        if self.rou > 0 or refresh_response:
            if refresh_response:
                refresh_index_response = (
                    torch.arange(x_response.size(1)).unsqueeze(0).expand(B, -1)
                )

            refresh_index_response: torch.Tensor = refresh_index_response + self._prompt_length
            self.key_cache[layer_idx][
                self.active_seq_mask.nonzero(), refresh_index_response
            ] = k[:, prompt_offset:]
            self.value_cache[layer_idx][
                self.active_seq_mask, self._prompt_length :
            ] = v[:, prompt_offset:]

            if not refresh_response:
                refresh_index = torch.cat([refresh_index, refresh_index_response], dim=-1)

            if q_position_ids is not None:
                assert position_ids is not None
                row_indices = (
                    torch.arange(B).unsqueeze(-1).expand_as(refresh_index_response)
                )
                q_position_ids = torch.cat(
                    [
                        q_position_ids,
                        position_ids[row_indices, refresh_index_response],
                    ],
                    dim=-1,
                )

    self._refresh_index = refresh_index
    ctx = AttentionContext(
        q=q,
        k=self.key_cache[layer_idx][self.active_seq_mask],
        v=self.value_cache[layer_idx][self.active_seq_mask],
        residual=residual,
        attention_mask=AttentionContext.convert_attention_mask(
            attention_mask,
            dtype=q.dtype,
            query_length=q.shape[1],
            key_value_length=self.key_cache[layer_idx].shape[1],
        ),
        q_position_ids=q_position_ids,
        kv_position_ids=position_ids,
    )

    yield ctx

    assert ctx.o is not None
    if len(self.attn_cache) <= layer_idx:
        self.attn_cache.append(ctx.o)
    else:
        if ctx.o.numel() > 0:
            self.attn_cache[layer_idx][
                self.active_seq_mask.nonzero(), refresh_index
            ] = ctx.o

    ctx.o = self.attn_cache[layer_idx][self.active_seq_mask]
```

**逐行解释——第一部分：Prompt/Response 分离与刷新决策 (L21-L56)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L21 | `@contextmanager` | 上下文管理器装饰器。 |
| L22-L32 | `def attention(self, layer_idx, x, ...):` | 参数与 PrefixCache 相同：层索引、隐藏状态 `x` (B, T, C)、norm 层、Q/K/V 投影层、attention_mask、position_ids。 |
| L33 | `refresh_prompt = self.refresh_prompt or layer_idx == 0` | 判断是否需要刷新 Prompt：`on_step_start` 设置的 `refresh_prompt` 为 `True`，或当前是首次前向传播（`layer_idx == 0`）。 |
| L34 | `refresh_response = self.refresh_response or layer_idx == 0` | 同上，判断是否需要刷新 Response。 |
| L35 | `residual = x` | 保存残差连接的输入，形状 `(B, T, C)`。 |
| L36 | `x = attn_norm(x)` | 对输入执行 LayerNorm，形状不变 `(B, T, C)`。 |
| L37 | `x_prompt = x[:, : self._prompt_length]` | 切出 Prompt 部分，形状 `(B, P, C)`，其中 `P = self._prompt_length`。 |
| L38 | `x_response = x[:, self._prompt_length :]` | 切出 Response（已生成 token）部分，形状 `(B, T-P, C)`。 |
| L39 | `x = x[:, 0:0]` | 将 `x` 清空为空张量 `(B, 0, C)`，后续按需追加。 |
| L40 | `refresh_index = torch.tensor([], ...)` | 初始化需要刷新的位置索引为空张量，形状 `(0,)`。 |
| L41-L43 | `if refresh_prompt: ...` | 若需刷新 Prompt，将 `x_prompt` 赋值给 `x`，`refresh_index` 设为 `[0, 1, ..., P-1]`。 |
| L45-L55 | `if self.rou > 0 or refresh_response:` | 若开启自适应刷新（`rou > 0`）或需要完全刷新 Response： |
| L46 | `x = torch.cat([x, x_response], dim=1)` | 在 dim=1 上拼接 `x`（可能已含 Prompt）与 `x_response`，`x` 形状变为 `(B, prompt_part+R, C)`。 |
| L47-L55 | `if refresh_response: ...` | 若需完全刷新 Response，将 Response 的全部位置索引追加到 `refresh_index`。 |
| L56 | `refresh_index = refresh_index.unsqueeze(0).expand(x.size(0), -1)` | 将 `refresh_index` 从 `(N,)` 扩展为 `(B, N)`，适配 batch 维度。 |

**逐行解释——第二部分：Q/K/V 投影与自适应选择 (L58-L94)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L58 | `B, T, C = x.shape` | `T` 是当前 batch 中选中的位置数（prompt + response 需刷新部分）。 |
| L59-L61 | `q, k, v = torch.empty((B, 0, ...))` | 初始化 Q、K、V 为空张量 `(B, 0, H)`，后续按需拼接。 |
| L64-L66 | `if refresh_response or self.rou == 0 or len(self.key_cache) <= layer_idx:` | **全量计算分支**：当 (a) 需全刷新 Response、(b) 自适应已关闭 (`rou=0`)、或 (c) 首次前向（缓存为空）时，直接对所有选中的 `x` 执行 Q/K/V 投影。 |
| L65-L66 | `if x.numel() > 0: q, k, v = q_proj(x), k_proj(x), v_proj(x)` | 全量投影，`q/k/v` 形状 `(B, T, H)`。若 `x` 为空则保持空张量。 |
| L67-L94 | `else:` | **自适应刷新分支**：仅刷新部分 Response 位置。 |
| L68-L71 | `if refresh_prompt: ...` | 若需刷新 Prompt，重新切出 Prompt 和 Response 部分。`q/k/v` 初始化为 Prompt 部分的投影结果。 |
| L72-L73 | `else: x_response = x` | 若不刷新 Prompt，则 `x` 整体为 Response 部分（Prompt 不在 `x` 中）。 |
| L76 | `v_response = v_proj(x_response)` | 对所有 Response 位置计算 Value 投影，形状 `(B, R, H)`，用于相似度比较。 |
| L77 | `num_replace = int(x_response.size(1) * self.rou)` | 计算需替换的位置数 = Response 长度 × `rou`。 |
| L78-L84 | `cos_sim = F.cosine_similarity(v_response, cached_v, dim=-1)` | 沿 head_dim 维度计算当前 `v_response` (B, R, H) 与缓存的 value (B, R, H) 的余弦相似度，结果形状 `(B, R)`。相似度越低表示该位置语义变化越大。 |
| L85-L87 | `refresh_index_response = torch.topk(cos_sim, largest=False, k=num_replace).indices` | 取相似度最低的 `num_replace` 个 Response 位置索引，形状 `(B, num_replace)`。 |
| L89-L91 | `selected_x_response = torch.gather(x_response, 1, refresh_index_response...)` | 从 `x_response` (B, R, C) 中 gather 出选中的位置，形状 `(B, num_replace, C)`。 |
| L92 | `q = torch.cat([q, q_proj(selected_x_response)], dim=1)` | 对选中的 Response 位置计算 Q 投影并拼接到 `q` 末尾。 |
| L93 | `k = torch.cat([k, k_proj(selected_x_response)], dim=1)` | 同上，拼接 K 投影。 |
| L94 | `v = torch.cat([v, v_response], dim=1)` | 注意：`v` 拼接的是完整的 `v_response`（所有 Response 的 V），因为 Value 总是全量更新。 |

**逐行解释——第三部分：KV 缓存更新 (L96-L148)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L97-L101 | `if len(self.key_cache) <= layer_idx:` | 首次前向传播，将 K/V 整体追加为缓存。 |
| L103-L113 | `else: if refresh_prompt:` | 非首次时，若需刷新 Prompt，将 `k[:, :P]` 和 `v[:, :P]` 写入缓存的 Prompt 区域（`[active_seq_mask, :P]`）。`prompt_offset = P` 用于后续偏移计算。 |
| L112-L113 | `else: prompt_offset = 0` | 不刷新 Prompt，偏移量为 0。 |
| L114-L116 | `q_position_ids = position_ids[:, :prompt_offset]` | 若仅刷新了 Prompt，Q 的 position_ids 仅取前 `prompt_offset` 个。 |
| L118-L149 | `if self.rou > 0 or refresh_response:` | 需要刷新 Response 时： |
| L119-L123 | `if refresh_response:` | 完全刷新时，`refresh_index_response` 设为 `[P, P+1, ..., P+R-1]`，覆盖全部 Response 位置。 |
| L125 | `refresh_index_response += self._prompt_length` | 将 Response 局部索引转为全局序列索引（加上 `_prompt_length` 偏移）。 |
| L126-L128 | `self.key_cache[layer_idx][active_seq_mask.nonzero(), refresh_index_response] = k[:, prompt_offset:]` | **Key 缓存精度更新**：仅将自适应/周期性选中的 Response 位置的 Key 写入缓存。 |
| L130-L132 | `self.value_cache[layer_idx][active_seq_mask, self._prompt_length:] = v[:, prompt_offset:]` | **Value 缓存全量更新**：将 `v` 的 Response 部分完全覆盖缓存的 Response 区域。Value 总是全量更新，因为 FFN 的计算依赖它。 |
| L134-L135 | `if not refresh_response: refresh_index = torch.cat(...)` | 若非完全刷新，将自适应选择的 `refresh_index_response` 合并到总刷新索引中。 |
| L137-L148 | `if q_position_ids is not None: ...` | 拼接 `q_position_ids`：非完全刷新时，需补全自适应选择位置的 position_ids。 |

**逐行解释——第四部分：构建 AttentionContext 与 yield (L151-L178)：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L151 | `self._refresh_index = refresh_index` | 保存刷新索引供 `ffn` 方法使用，形状 `(B, N_refresh)`。 |
| L152-L163 | `ctx = AttentionContext(q=q, k=cached_k, v=cached_v, ...)` | 构建 `AttentionContext`：`q` 为本次计算的 Query（仅刷新位置），`k`/`v` 取自完整缓存（含未刷新位置），`attention_mask` 经 `convert_attention_mask` 转换为加性掩码。 |
| L167 | `yield ctx` | 交出控制权，让模型完成注意力计算。 |
| L170-L176 | `if len(self.attn_cache) <= layer_idx: self.attn_cache.append(ctx.o)` | yield 返回后，缓存注意力输出 `ctx.o`：首次追加，非首次按 `(active_seq_mask.nonzero(), refresh_index)` 索引增量更新。 |
| L178 | `ctx.o = self.attn_cache[layer_idx][self.active_seq_mask]` | 将完整的注意力输出缓存赋值给 `ctx.o`（含未刷新位置），下游代码看到完整输出。 |

**刷新策略决策树：**

| 条件 | 行为 |
|------|------|
| `len(key_cache) <= layer_idx`（首次） | 全量计算 Q/K/V，全量缓存 |
| `refresh_response` 或 `rou == 0` | 全量计算所有选中位置的 Q/K/V |
| `rou > 0` 且非刷新周期 | 自适应：仅对 `rou` 比例的最低相似度位置计算 Q/K |

---

### `ffn` — FFN 层上下文管理器

```python
# 源文件: src/cache/dllm_cache.py L180-L197
@contextmanager
def ffn(self, layer_idx: int, x: torch.Tensor):
    B, _, C = x.shape
    row_indices = torch.arange(B).unsqueeze(-1).expand_as(self._refresh_index)
    residual = x
    x = x[row_indices, self._refresh_index]
    ctx = FFNContext(x=x, residual=residual)

    yield ctx

    assert ctx.ffn_out is not None
    if len(self.ffn_cache) <= layer_idx:
        self.ffn_cache.append(ctx.ffn_out)
    else:
        self.ffn_cache[layer_idx][
            self.active_seq_mask.nonzero(), self._refresh_index
        ] = ctx.ffn_out
    ctx.ffn_out = self.ffn_cache[layer_idx][self.active_seq_mask]
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L180 | `@contextmanager` | 上下文管理器装饰器。 |
| L181 | `def ffn(self, layer_idx, x):` | 输入 `x` 为注意力输出与残差相加后的结果，形状 `(B, T, C)`。 |
| L182 | `B, _, C = x.shape` | 提取 batch、隐藏维度。 |
| L183 | `row_indices = torch.arange(B).unsqueeze(-1).expand_as(self._refresh_index)` | 构造行索引 `(B, N_refresh)`，每一行为 `[0,1,...,B-1]` 的列重复。 |
| L184 | `residual = x` | 保存完整 `x` 作为残差引用，形状 `(B, T, C)`。 |
| L185 | `x = x[row_indices, self._refresh_index]` | 仅选取需要刷新的位置，`x` 从 `(B, T, C)` 裁剪为 `(B, N_refresh, C)`。关键的索引对齐：第二维使用 `attention` 中记录的 `_refresh_index`。 |
| L186 | `ctx = FFNContext(x=x, residual=residual)` | 构建 FFN 上下文，`residual` 仍为完整张量，供后续残差加法使用。 |
| L188 | `yield ctx` | 交出控制权，让模型执行 FFN 计算（仅对 `N_refresh` 个位置）。 |
| L191-L196 | `if len(self.ffn_cache) <= layer_idx: ...` | 缓存 FFN 输出：首次追加完整缓存 `(B, T, C)`，非首次按 `(active_seq_mask.nonzero(), _refresh_index)` 增量更新。 |
| L197 | `ctx.ffn_out = self.ffn_cache[layer_idx][self.active_seq_mask]` | 将完整 FFN 缓存赋值回 `ctx.ffn_out`，下游代码看到完整输出 `(B_active, T, C)`。 |

---

### `on_step_start` — 步骤开始回调

```python
# 源文件: src/cache/dllm_cache.py L199-L220
def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
    current_steps = frame.steps.max(-1, keepdim=True).values
    refresh_prompt = (current_steps + 1) % self.kp == 0
    refresh_response = (current_steps + 1) % self.kr == 0
    B, self._prompt_length = frame.prompts.shape

    try:
        active_seq_mask = self.active_seq_mask
    except RuntimeError:
        active_seq_mask = torch.ones(
            (B,), dtype=torch.bool, device=current_steps.device
        )

    assert (
        torch.unique(refresh_prompt[active_seq_mask]).numel() <= 1
        and torch.unique(refresh_response[active_seq_mask]).numel() <= 1
    ), "All unfinished sequences must have the same refresh schedule."

    if refresh_prompt[active_seq_mask].numel() > 0:
        self.refresh_prompt = refresh_prompt[active_seq_mask][0].item()
        self.refresh_response = refresh_response[active_seq_mask][0].item()
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|------|------|------|
| L199 | `def on_step_start(self, block_mask, frame):` | 每个生成步骤开始时调用，决定本步的刷新策略。`frame.steps` 记录每个序列的当前步数，形状 `(B, seq_len)`。 |
| L200 | `current_steps = frame.steps.max(-1, keepdim=True).values` | 取每个序列的最大步数，形状 `(B, 1)`。 |
| L201 | `refresh_prompt = (current_steps + 1) % self.kp == 0` | 判断是否到达 Prompt 刷新周期。`+1` 因为步数从 0 开始计数，布尔张量形状 `(B, 1)`。 |
| L202 | `refresh_response = (current_steps + 1) % self.kr == 0` | 判断是否到达 Response 刷新周期。 |
| L203 | `B, self._prompt_length = frame.prompts.shape` | 获取 batch 大小和 prompt 长度 P。 |
| L205-L210 | `try: ... except RuntimeError:` | 安全获取 `active_seq_mask`：首次调用时父类属性可能尚未初始化，fallback 为全 `True` 掩码。 |
| L213-L216 | `assert torch.unique(refresh_prompt[active_seq_mask]).numel() <= 1 ...` | 断言所有活跃序列的刷新调度一致（批处理要求同步），确保 `refresh_prompt` 和 `refresh_response` 在活跃序列间无歧义。 |
| L218-L220 | `self.refresh_prompt/refresh_response = ... item()` | 从布尔张量中提取标量值，赋值给实例属性，供 `attention` 方法使用。 |

## 关键函数和上下文管理器说明

### 余弦相似度计算

```python
cos_sim = F.cosine_similarity(
    v_response,                                    # (B, response_len, head_dim)
    self.value_cache[layer_idx][                   # (B, response_len, head_dim)
        self.active_seq_mask, self._prompt_length :
    ],
    dim=-1,                                        # 沿 head_dim 计算相似度
)
# 结果形状: (B, response_len)
```

### Top-K 选择

```python
refresh_index_response = torch.topk(
    cos_sim,           # 相似度分数
    largest=False,     # 选择最小值（相似度最低）
    k=num_replace      # 要选择的数量
).indices
```

### 注意力输出缓存更新

```python
assert ctx.o is not None
if len(self.attn_cache) <= layer_idx:
    self.attn_cache.append(ctx.o)
else:
    if ctx.o.numel() > 0:
        self.attn_cache[layer_idx][
            self.active_seq_mask.nonzero(), refresh_index
        ] = ctx.o

# 返回完整的注意力缓存
ctx.o = self.attn_cache[layer_idx][self.active_seq_mask]
```

## 使用示例和参数配置

### 基本使用

```python
from src.cache import dLLMCache
from transformers import AutoConfig

# 加载模型配置
model_config = AutoConfig.from_pretrained("path/to/model")

# 创建 dLLMCache 实例
cache = dLLMCache(
    model_config,
    kp=50,      # 每 50 步刷新一次 Prompt
    kr=2,       # 每 2 步刷新一次 Response
    rou=0.25    # 自适应刷新 25% 的 Response 位置
)

# 在生成过程中使用
for step in range(num_steps):
    with cache.model_forward(hidden_states) as ctx:
        # 模型处理...
        pass
```

### 配置文件示例

```yaml
# configs/cache/dllm.yaml
_target_: src.cache.dLLMCache
kp: 50
kr: 2
rou: 0.25
```

### 不同场景的参数配置

```yaml
# 高精度配置（适合需要高质量生成的场景）
kp: 20
kr: 1
rou: 0.5

# 高效率配置（适合需要快速生成的场景）
kp: 100
kr: 5
rou: 0.1

# 平衡配置（默认推荐）
kp: 50
kr: 2
rou: 0.25
```

### 与模型集成

```python
class MyDiffusionModel(nn.Module):
    def forward(self, input_ids, cache=None):
        hidden_states = self.embed(input_ids)
        
        if cache is not None:
            with cache.model_forward(hidden_states) as ctx:
                for layer_idx, layer in enumerate(self.layers):
                    # 注意力层
                    with cache.attention(
                        layer_idx, ctx.x,
                        layer.attn_norm, layer.q_proj,
                        layer.k_proj, layer.v_proj,
                        attention_mask, position_ids
                    ) as attn_ctx:
                        attn_ctx.o = layer.attn(attn_ctx)
                        ctx.x = attn_ctx.o + attn_ctx.residual
                    
                    # FFN 层
                    with cache.ffn(layer_idx, ctx.x) as ffn_ctx:
                        ffn_ctx.ffn_out = layer.ffn(ffn_ctx.x)
                        ctx.x = ffn_ctx.ffn_out + ffn_ctx.residual
                
                ctx.logits = self.lm_head(ctx.x)
        
        return ctx.logits if cache else self.standard_forward(hidden_states)
```

## 性能特点和适用场景

### 性能特点

| 特点 | 说明 |
|------|------|
| **自适应刷新** | 根据语义相似度动态选择刷新位置，智能平衡精度和效率 |
| **周期性调度** | Prompt 和 Response 分离调度，适应不同的稳定性需求 |
| **多层缓存** | 同时缓存 KV、Attention 输出和 FFN 输出，最大化复用 |
| **批量支持** | 完全支持批量处理，自动处理序列结束情况 |

### 计算效率分析

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    计算效率对比 (假设序列长度 512)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  无缓存方法:                                                             │
│  ████████████████████████████████████████████████████████ 100%          │
│                                                                         │
│  PrefixCache (假设 10% 位置需要更新):                                    │
│  ████████████████████████████████████████████████████████ 100%          │
│  (首次)                                                                  │
│  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10%          │
│  (后续)                                                                  │
│                                                                         │
│  dLLM-Cache (kp=50, kr=2, rou=0.25):                                    │
│  ████████████████████████████████████████████████████████ 100%          │
│  (首次)                                                                  │
│  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%           │
│  (常规步骤: 只刷新 25% Response)                                         │
│  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12%          │
│  (kr 周期: 完全刷新 Response)                                           │
│  ████████████████████████████████████████████████████████ 100%          │
│  (kp 周期: 完全刷新 Prompt + Response)                                  │
│                                                                         │
│  图例: █ 需要计算   ░ 复用缓存                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 适用场景

1. **长文本生成**
   - 长 Prompt 场景下，通过较大的 `kp` 减少 Prompt 刷新开销
   - 自适应刷新避免不必要的 Response 重新计算

2. **批量推理**
   - 支持多序列并行处理
   - 自动处理不同序列的刷新调度同步

3. **质量敏感场景**
   - 通过调整 `kr` 和 `rou` 参数，可以在保证生成质量的同时提升效率
   - 语义相似度检测确保关键位置得到及时刷新

### 与 PrefixCache 的对比

| 特性 | PrefixCache | dLLM-Cache |
|------|-------------|------------|
| KV 缓存 | ✓ | ✓ |
| 选择性更新 | ✓ | ✓ |
| 周期性刷新 | ✗ | ✓ (kp, kr) |
| 自适应刷新 | ✗ | ✓ (rou) |
| Attention 缓存 | ✗ | ✓ |
| FFN 缓存 | ✗ | ✓ |
| 语义相似度检测 | ✗ | ✓ |
| 实现复杂度 | 低 | 中 |
| 计算效率提升 | 中 | 高 |

### 局限性

1. **调度同步要求**：批量中的所有序列必须具有相同的刷新调度
2. **额外内存开销**：需要存储 Attention 和 FFN 缓存
3. **参数敏感性**：参数选择对性能和质量影响较大，需要根据场景调优

## 总结

dLLM-Cache 是一种高级的扩散语言模型缓存策略，它通过周期性刷新调度和自适应语义刷新机制，在保证生成质量的同时显著提升了计算效率。相比 PrefixCache，dLLM-Cache 引入了更多的智能决策机制，能够根据语义变化动态调整刷新策略，特别适合长文本生成和批量推理场景。通过合理配置 `kp`、`kr` 和 `rou` 参数，可以在计算效率和生成质量之间找到最佳平衡点。
