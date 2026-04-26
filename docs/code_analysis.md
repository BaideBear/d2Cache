# d2Cache 项目代码分析文档

**版本**: 1.0  
**最后更新**: 2026-04-25  
**作者**: AI Assistant

---

## 目录

1. [项目概览](#1-项目概览)
   - 1.1 [背景与研究目标](#11-背景与研究目标)
   - 1.2 [支持的模型](#12-支持的模型)
   - 1.3 [支持的 KV 缓存方法](#13-支持的-kv-缓存方法)
   - 1.4 [支持的解码策略](#14-支持的解码策略)
   - 1.5 [核心优势](#15-核心优势)

2. [项目分级结构](#2-项目分级结构)
   - 2.1 [顶层目录结构](#21-顶层目录结构)
   - 2.2 [核心模块详解](#22-核心模块详解)
   - 2.3 [配置系统](#23-配置系统)
   - 2.4 [评估系统](#24-评估系统)

3. [关键数据结构](#3-关键数据结构)
   - 3.1 [Frame - 生成状态容器](#31-frame---生成状态容器)
   - 3.2 [FrameDelta - 状态变化描述](#32-framedelta---状态变化描述)
   - 3.3 [DecodeRecord - 解码轨迹记录](#33-decoderecord---解码轨迹记录)
   - 3.4 [dCache - 缓存抽象基类](#34-dcache---缓存抽象基类)
   - 3.5 [d2Cache - 双自适应缓存实现](#35-d2cache---双自适应缓存实现)
   - 3.6 [数据流转机制](#36-数据流转机制)

4. [扩展开发指南](#4-扩展开发指南)
   - 4.1 [实现自定义缓存机制](#41-实现自定义缓存机制)
   - 4.2 [开发新的解码策略](#42-开发新的解码策略)
   - 4.3 [添加新模型支持](#43-添加新模型支持)
   - 4.4 [配置文件使用指南](#44-配置文件使用指南)

5. [代码示例与最佳实践](#5-代码示例与最佳实践)
   - 5.1 [缓存机制实现示例](#51-缓存机制实现示例)
   - 5.2 [解码策略实现示例](#52-解码策略实现示例)
   - 5.3 [配置文件示例](#53-配置文件示例)
   - 5.4 [常见问题与解决方案](#54-常见问题与解决方案)

6. [快速参考](#6-快速参考)
   - 6.1 [核心类索引](#61-核心类索引)
   - 6.2 [配置参数速查](#62-配置参数速查)
   - 6.3 [命令行示例](#63-命令行示例)

---

## 1. 项目概览

### 1.1 背景与研究目标

**d2Cache** 是一个专注于扩散语言模型（Diffusion Language Models, dLLMs）研究的代码库，由研究团队开发并维护。该项目的主要目标包括：

1. **提供统一评估框架**: 为扩散语言模型提供标准化的测试环境，允许用户无缝切换不同的基线方法
2. **实现高效缓存机制**: 解决 dLLMs 推理效率低下的问题，通过创新的 KV 缓存策略加速推理
3. **支持多种解码策略**: 实现并对比多种解码算法，为研究人员提供可复现的实验基准
4. **促进研究复现**: 提供清晰、文档完善的代码，便于研究人员理解和扩展

#### 核心创新点

d2Cache 的核心创新在于提出了 **Dual aDaptive Cache（双自适应缓存）**，这是一种无需训练的近似 KV 缓存框架，专门用于加速 dLLM 推理。其主要特点：

- **两阶段细粒度选择策略**: 识别并自适应更新关键 token 的 KV 状态
- **准左到右生成**: 提供更可靠的解码方案，缓解序列末尾 token 的过早过度自信问题
- **质量与速度双重提升**: 不仅实现显著推理加速，还能提高生成质量

### 1.2 支持的模型

项目目前支持以下扩散语言模型：

| 模型 | 变体 | 论文 | 代码仓库 |
|------|------|------|----------|
| **LLaDA-8B** | `llada-base`, `llada-inst` | [NIPS Oral - 2502.09992](https://arxiv.org/abs/2502.09992) | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) |
| **LLaDA-1.5** | `llada-1.5` | [arXiv - 2505.19223](https://arxiv.org/abs/2505.19223) | [ML-GSAI/LLaDA-1.5](https://github.com/ML-GSAI/LLaDA-1.5) |
| **Dream-v0-7B** | `dream-base`, `dream-inst` | [arXiv - 2508.15487](https://arxiv.org/abs/2508.15487) | [DreamLM/Dream](https://github.com/DreamLM/Dream) |

#### 模型特点

- **LLaDA**: 从头训练的扩散语言模型，采用前向数据掩码和反向生成过程
- **Dream**: 基于自回归模型改编的扩散模型，保留了部分 AR 特性

### 1.3 支持的 KV 缓存方法

| 方法 | 论文 | 特点 | 代码仓库 |
|------|------|------|----------|
| **PrefixCache / DualCache** | [ICLR - 2505.22618](https://arxiv.org/abs/2505.22618) | 块级近似 KV 缓存，支持双向注意力 | [NVLabs/Fast-dLLM](https://github.com/NVLabs/Fast-dLLM) |
| **dLLM-Cache** | [arXiv - 2506.06295](https://arxiv.org/abs/2506.06295) | 自适应缓存框架，结合长间隔提示缓存和部分响应更新 | [maomaocun/dLLM-Cache](https://github.com/maomaocun/dLLM-Cache) |
| **d2Cache** | [ICLR - 2509.23094](https://arxiv.org/abs/2509.23094) | 双自适应缓存，两阶段细粒度选择策略 | 本项目 |

#### 缓存方法对比

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│     方法        │   选择策略       │   更新机制       │   性能提升       │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ PrefixCache     │ 块级选择         │ 固定更新         │ ~27.6x 吞吐量    │
│ dLLM-Cache      │ 特征相似度       │ 自适应更新       │ ~9.1x 加速       │
│ d2Cache         │ 两阶段细粒度     │ 自适应更新       │ 质量与速度双提升 │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

### 1.4 支持的解码策略

| 策略 | 论文 | 特点 | 适用场景 |
|------|------|------|----------|
| **Auto-regressive** | - | 传统自回归解码 | 基线对比 |
| **Vanilla / Semi-AR** | [NIPS Oral - 2502.09992](https://arxiv.org/abs/2502.09992) | 标准扩散解码，支持半自回归 | 通用生成 |
| **Parallel** | [ICLR - 2505.22618](https://arxiv.org/abs/2505.22618) | 置信度感知并行解码 | 快速生成 |
| **PC-Sampler** | [arXiv - 2508.13021](https://arxiv.org/abs/2508.13021) | 位置感知校准采样 | 逻辑推理任务 |
| **Certainty Prior** | [ICLR - 2509.23094](https://arxiv.org/abs/2509.23094) | 确定性先验引导解码 | 高质量生成 |
| **KLASS** | [NIPS Spotlight - 2511.05664](https://arxiv.org/abs/2511.05664) | KL 引导快速推理 | 推理加速 |
| **EB-Sampler** | [NIPS - 2505.24857](https://arxiv.org/abs/2505.24857) | 熵有界解掩码 | 自适应步数 |
| **WINO** | [ICLR - 2507.18578](https://arxiv.org/abs/2507.18578) | 可撤销解码机制 | 质量敏感任务 |

### 1.5 核心优势

1. **统一评估框架**
   - 标准化测试环境
   - 无缝切换不同基线方法
   - 支持多种评估数据集（GSM8K、HumanEval、MATH-500、MBPP 等）

2. **清晰且文档完善的代码**
   - 强调代码可读性和清晰度
   - 完整的文档和代码阅读指南
   - 模块化设计，易于理解和扩展

3. **活跃且持续的维护**
   - 持续更新和改进
   - 未来将包含更多开箱即用的基线方法
   - 社区支持和反馈

---

## 2. 项目分级结构

### 2.1 顶层目录结构

```
d2Cache/
├── assets/              # 项目资源文件（图片、logo等）
│   ├── d2cache.png     # d2Cache 架构图
│   ├── framework.png   # 整体框架图
│   ├── hooks.png       # 钩子机制图
│   └── logo.png        # 项目 logo
│
├── configs/             # 配置文件目录
│   ├── cache/          # 缓存配置
│   ├── generation/     # 生成策略配置
│   ├── model/          # 模型配置
│   ├── default.yaml    # 默认配置
│   ├── eval.yaml       # 评估配置
│   └── gen_args.py     # 动态参数生成脚本
│
├── docs/                # 文档目录
│   ├── code_reading_guides.md    # 代码阅读指南
│   ├── customization.md          # 自定义开发指南
│   ├── decoding_strategies.md    # 解码策略文档
│   └── kv_caching.md             # KV 缓存文档
│
├── requirements/        # 依赖文件
│   └── common.txt      # 通用依赖
│
├── scripts/             # 脚本文件
│   ├── fix_code_eval.sh    # 修复代码评估脚本
│   └── run_eval.sh         # 运行评估脚本
│
├── src/                 # 源代码目录
│   ├── cache/          # KV 缓存实现
│   ├── generation/     # 解码策略实现
│   ├── models/         # 模型定义
│   ├── third_party/    # 第三方资源
│   ├── utils/          # 工具函数
│   ├── frame.py        # 核心数据结构
│   └── __init__.py
│
├── tasks/               # 评估任务配置
│   ├── humaneval/      # HumanEval 任务
│   └── math-500/       # MATH-500 任务
│
├── eval.py              # 评估入口脚本
├── pyproject.toml       # 项目配置
├── README.md            # 项目说明
└── LICENSE              # 许可证
```

### 2.2 核心模块详解

#### 2.2.1 `src/frame.py` - 核心数据结构

这是项目最核心的模块，定义了生成过程中状态管理的三个关键类：

**主要类**:
- `Frame`: 存储完整的生成状态信息
- `FrameDelta`: 表示两步之间的状态变化
- `DecodeRecord`: 聚合整个解码轨迹
- `Intermediate`: 存储中间状态（隐藏状态、KV 状态等）

**设计理念**:
- 支持单序列和批量序列两种模式
- 使用 Pydantic 进行数据验证
- 提供便捷的状态转换和查询方法

**文件位置**: [src/frame.py](file:///Users/lier/codes/d2Cache/src/frame.py)

#### 2.2.2 `src/cache/` - KV 缓存模块

缓存模块实现了多种 KV 缓存策略，采用统一的接口设计：

```
src/cache/
├── __init__.py          # 模块导出
├── base.py              # 缓存基类 dCache
├── d2cache.py           # d2Cache 实现
├── prefix_cache.py      # PrefixCache 实现
└── dllm_cache.py        # dLLM-Cache 实现
```

**核心设计**:
- 使用 Python 上下文管理器（`@contextmanager`）拦截模型计算流程
- 三个关键上下文管理器：`model_forward`、`attention`、`ffn`
- 生命周期钩子：`on_step_start`、`on_step_end`、`on_block_start`、`on_block_end`

**文件位置**: [src/cache/](file:///Users/lier/codes/d2Cache/src/cache/)

#### 2.2.3 `src/generation/` - 解码策略模块

解码策略模块实现了多种生成算法：

```
src/generation/
├── __init__.py          # 生成函数注册和调用
├── vanilla.py           # Vanilla/Semi-AR 解码
├── ar.py                # 自回归解码
├── klass.py             # KLASS 解码
├── wino.py              # WINO 解码
└── utils.py             # 工具函数
```

**核心机制**:
- 使用装饰器 `@register` 注册解码策略
- 统一的 `generate` 函数接口
- 支持动态参数验证和兼容性检查

**文件位置**: [src/generation/](file:///Users/lier/codes/d2Cache/src/generation/)

#### 2.2.4 `src/models/` - 模型定义模块

模型模块包含了对 LLaDA 和 Dream 模型的修改和适配：

```
src/models/
├── __init__.py
├── eval_mdlm.py         # 评估模型封装
├── llada/               # LLaDA 模型
│   ├── __init__.py
│   ├── configuration_llada.py
│   ├── modeling_llada.py
│   ├── eval_model.py
│   └── generation_utils.py
└── dream/               # Dream 模型
    ├── __init__.py
    ├── configuration_dream.py
    ├── modeling_dream.py
    ├── eval_model.py
    └── generation_utils.py
```

**关键修改**:
- 在模型层中应用 `@attention` 和 `@ffn` 上下文管理器
- 在模型前向传播中应用 `@model_forward` 上下文管理器
- 支持缓存机制的集成

**文件位置**: [src/models/](file:///Users/lier/codes/d2Cache/src/models/)

#### 2.2.5 `src/utils/` - 工具函数模块

工具模块提供了各种辅助功能：

```
src/utils/
├── __init__.py
├── common.py            # 通用工具函数
└── models.py            # 模型加载工具
```

**主要功能**:
- `Registry`: 注册机制，用于管理解码策略
- `Timer`: 计时器，用于性能测量
- `certainty_density`: 确定性密度计算（用于 d2Cache）
- `tensor_insert` / `tensor_delete`: 张量操作工具
- 模型加载和分词器加载函数

**文件位置**: [src/utils/](file:///Users/lier/codes/d2Cache/src/utils/)

### 2.3 配置系统

项目使用 Hydra 进行配置管理，采用 YAML 文件和 Python 脚本相结合的方式。

#### 2.3.1 配置文件层次结构

```yaml
# configs/default.yaml
defaults:
    - model: llada-base      # 模型配置
    - generation: vanilla    # 生成策略配置
    - cache: null            # 缓存配置（可选）
    - _self_

seed: 42
gen_args_script: configs/gen_args.py
batch_size: ${dataset.batch_size}
attn_implementation: "sdpa"  # sdpa 或 eager
dataset:
    name: null
    size: null
    n_shot: null
    system_prompt: null
    batch_size: 1
```

#### 2.3.2 配置文件类型

1. **模型配置** (`configs/model/`)
   - `llada-base.yaml`: LLaDA-8B-Base 配置
   - `llada-inst.yaml`: LLaDA-8B-Instruct 配置
   - `llada-1.5.yaml`: LLaDA-1.5 配置
   - `dream-base.yaml`: Dream-v0-Base-7B 配置
   - `dream-inst.yaml`: Dream-v0-Instruct-7B 配置

2. **缓存配置** (`configs/cache/`)
   - `d2cache.yaml`: d2Cache 配置
   - `dllm.yaml`: dLLM-Cache 配置
   - `prefix.yaml`: PrefixCache 配置

3. **生成策略配置** (`configs/generation/`)
   - `vanilla.yaml`: Vanilla 解码配置
   - `ar.yaml`: 自回归解码配置
   - `klass.yaml`: KLASS 解码配置
   - `eb_sampler.yaml`: EB-Sampler 配置
   - `pc_sampler.yaml`: PC-Sampler 配置
   - `wino.yaml`: WINO 解码配置

#### 2.3.3 动态参数生成

`configs/gen_args.py` 提供了动态默认值的功能：

```python
# 根据模型和数据集动态设置参数
def get_gen_args(cfg):
    # 返回动态生成的参数
    return extra_gen_kwargs
```

### 2.4 评估系统

#### 2.4.1 评估入口

`eval.py` 是评估的主入口，使用 Hydra 进行配置管理：

```python
@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. 预初始化
    extra_cfg = pre_initialize(cfg)
    
    # 2. 加载模型
    model = load_eval_model(cfg, extra_gen_kwargs=extra_cfg.get("extra_gen_kwargs"))
    
    # 3. 运行评估
    results = simple_evaluate(model=model, **overwrite_eval_task(cfg))
    
    # 4. 保存结果
    # ...
```

#### 2.4.2 支持的评估数据集

- **数学推理**: GSM8K, MATH-500
- **代码生成**: HumanEval, HumanEval-Plus, MBPP
- **通用任务**: lm-eval 支持的所有任务

#### 2.4.3 任务配置

每个评估任务在 `tasks/` 目录下有专门的配置：

```yaml
# tasks/humaneval/humaneval.yaml
dataset:
    name: humaneval
    batch_size: 1
    
generation:
    gen_length: 512
    block_length: 64
    num_transfer_tokens: 1
```

---

## 3. 关键数据结构

### 3.1 Frame - 生成状态容器

`Frame` 是存储扩散模型生成状态的核心数据结构，支持单序列和批量序列两种模式。

#### 3.1.1 数据结构定义

```python
class Frame(Base):
    prompts: torch.Tensor          # 提示词 token，形状 (prompt_length,) 或 (batch_size, prompt_length)
    generated_tokens: torch.Tensor # 生成的 token（包括未解码的 mask token）
    confidence: torch.Tensor       # 每个 token 的置信度
    steps: torch.Tensor            # 每个 token 的解码步骤
```

#### 3.1.2 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `prompts` | `torch.Tensor` | 提示词 token，不可变 |
| `generated_tokens` | `torch.Tensor` | 生成的 token 序列，初始为 mask token |
| `confidence` | `torch.Tensor` | 每个 token 的置信度分数 |
| `steps` | `torch.Tensor` | 每个 token 被解码的步骤号，mask token 为 -1 |
| `is_batched` | `bool` | 是否为批量模式 |
| `current_steps` | `int \| torch.Tensor` | 当前解码步骤 |

#### 3.1.3 核心方法

**创建初始帧**:
```python
@classmethod
def create_initial_frame(
    cls, 
    prompts: torch.Tensor, 
    gen_length: int, 
    mask_token_id: int | None = None
) -> "Frame":
    """
    创建初始帧，所有生成位置初始化为 mask token
    
    Args:
        prompts: 提示词 token
        gen_length: 生成长度
        mask_token_id: mask token 的 ID
    
    Returns:
        Frame: 初始帧对象
    """
```

**应用状态变化**:
```python
def apply_delta(self, delta: FrameDelta, mask_token_id: int | None = None) -> "Frame":
    """
    应用 FrameDelta 到当前帧，返回新帧
    
    Args:
        delta: 状态变化对象
        mask_token_id: mask token ID
    
    Returns:
        Frame: 更新后的新帧
    """
```

#### 3.1.4 使用示例

```python
# 创建初始帧
prompts = torch.tensor([[1, 2, 3]])  # batch_size=1, prompt_length=3
frame = Frame.create_initial_frame(
    prompts=prompts,
    gen_length=10,
    mask_token_id=126081
)

# 查看帧状态
print(frame.generated_tokens)  # tensor([[126081, 126081, ..., 126081]])
print(frame.steps)              # tensor([[-1, -1, ..., -1]])
print(frame.current_steps)      # -1（初始状态）

# 应用 delta
delta = FrameDelta(
    transfer_index=(torch.tensor([0, 2]),),
    decoded_tokens=torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]),
    confidence=torch.tensor([[0.9, 0.8, 0.95, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55, 0.88]])
)
new_frame = frame.apply_delta(delta)
```

### 3.2 FrameDelta - 状态变化描述

`FrameDelta` 描述了两步解码之间的状态变化，包括哪些 token 被解码、哪些被插入或删除。

#### 3.2.1 数据结构定义

```python
class FrameDelta(Base):
    transfer_index: torch.Tensor | tuple[torch.Tensor, ...]  # 被解码 token 的目标位置
    transfer_src_index: torch.Tensor | tuple[torch.Tensor, ...] | None  # 源位置索引
    insert_index: torch.Tensor | None      # 插入位置
    insert_src_index: torch.Tensor | None  # 插入源索引
    delete_index: torch.Tensor | None      # 删除位置
    decoded_tokens: torch.Tensor           # 解码的 token
    confidence: torch.Tensor | None        # 置信度分数
    probs: torch.Tensor | None             # 概率分布
    intermediate: Intermediate             # 中间状态
    extra: dict                            # 额外信息
```

#### 3.2.2 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `transfer_index` | `torch.Tensor \| tuple` | 被解码 token 在生成序列中的位置索引 |
| `decoded_tokens` | `torch.Tensor` | 本步解码产生的 token |
| `confidence` | `torch.Tensor \| None` | 每个 token 的置信度 |
| `probs` | `torch.Tensor \| None` | 完整的概率分布 |
| `intermediate` | `Intermediate` | 中间状态（隐藏状态、KV 状态等） |
| `is_batched` | `bool` | 是否为批量模式 |

#### 3.2.3 核心方法

**获取已解码 token**:
```python
@property
def transferred_tokens(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    根据 transfer_index 获取已解码的 token
    
    Returns:
        torch.Tensor 或 tuple: 已解码的 token
    """
```

**批量转换**:
```python
def as_batch(self) -> "FrameDelta":
    """
    将单序列 delta 转换为批量格式
    
    Returns:
        FrameDelta: 批量格式的 delta
    """
```

#### 3.2.4 使用示例

```python
# 单序列 delta
delta_single = FrameDelta(
    transfer_index=torch.tensor([0, 2, 5]),
    decoded_tokens=torch.tensor([10, 20, 30, 40, 50, 60, 70, 80]),
    confidence=torch.tensor([0.9, 0.8, 0.95, 0.7, 0.6, 0.85, 0.75, 0.65])
)

# 批量 delta
delta_batch = FrameDelta(
    transfer_index=(
        torch.tensor([0, 1]),      # 序列 0 的转移索引
        torch.tensor([2, 3, 4]),   # 序列 1 的转移索引
    ),
    decoded_tokens=torch.tensor([
        [10, 20, 30, 40],          # 序列 0 的解码 token
        [50, 60, 70, 80, 90],      # 序列 1 的解码 token
    ]),
    confidence=torch.tensor([
        [0.9, 0.8, 0.7, 0.6],
        [0.95, 0.85, 0.75, 0.65, 0.55],
    ])
)

# 获取已解码 token
transferred = delta_single.transferred_tokens  # tensor([10, 30, 60])
```

### 3.3 DecodeRecord - 解码轨迹记录

`DecodeRecord` 聚合整个解码过程，包含初始帧和所有状态变化。

#### 3.3.1 数据结构定义

```python
class DecodeRecord(Base, Sequence):
    initial_frame: Frame              # 初始帧
    deltas: list[FrameDelta]          # 状态变化列表
    block_length: int | None          # 块长度（用于半自回归）
```

#### 3.3.2 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `initial_frame` | `Frame` | 解码过程的初始帧 |
| `deltas` | `list[FrameDelta]` | 所有状态变化的列表 |
| `block_length` | `int \| None` | 块长度，用于半自回归解码 |
| `frames` | `list[Frame]` | 所有帧的列表（按需计算） |
| `num_steps` | `int` | 解码步数 |
| `gen_length` | `int` | 生成长度 |

#### 3.3.3 核心方法

**访问特定步骤的帧**:
```python
def __getitem__(self, index: int) -> Frame:
    """
    获取特定步骤的帧
    
    Args:
        index: 步骤索引（0 为初始帧）
    
    Returns:
        Frame: 该步骤的帧
    """
```

**添加状态变化**:
```python
def append(self, delta: FrameDelta) -> None:
    """
    添加新的状态变化
    
    Args:
        delta: 状态变化对象
    """
```

#### 3.3.4 使用示例

```python
# 创建解码记录
record = DecodeRecord(
    initial_frame=initial_frame,
    deltas=[delta1, delta2, delta3],
    block_length=32
)

# 访问特定步骤的帧
frame_step_0 = record[0]  # 初始帧
frame_step_1 = record[1]  # 第一步后的帧
frame_final = record[-1]  # 最终帧

# 获取所有帧
all_frames = record.frames

# 查询统计信息
print(f"解码步数: {record.num_steps}")
print(f"生成长度: {record.gen_length}")
```

### 3.4 dCache - 缓存抽象基类

`dCache` 是所有缓存机制的抽象基类，定义了统一的接口和上下文管理器。

#### 3.4.1 核心设计理念

dLLMs 的 KV 缓存策略涉及：
1. 选择一部分关键 token 进行重新计算
2. 其余 token 的状态从缓存中获取
3. 在注意力阶段，只有选中的 token 通过 K、V、Q 投影矩阵
4. 在 FFN 层，只对选中的 token 进行计算

#### 3.4.2 数据结构定义

```python
class dCache:
    def __init__(self, model_config):
        self.model_config = model_config
        self.active_q_mask: torch.Tensor | None = None
        self._active_seq_mask: torch.Tensor | None = None
```

#### 3.4.3 核心上下文管理器

**1. `model_forward` - 模型前向传播上下文**:
```python
@contextmanager
def model_forward(self, x: torch.Tensor):
    """
    包装整个模型的前向传播
    
    Args:
        x: 输入张量，形状 (batch_size, seq_len, d_model)
    
    Yields:
        ModelForwardContext: 包含修改后的输入和输出
    """
```

**作用**:
- 修改输入嵌入或准备全局掩码
- 恢复最终 logits 到完整序列形状

**2. `attention` - 注意力计算上下文**:
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
    包装注意力计算
    
    Args:
        layer_idx: 层索引
        x: 输入张量
        attn_norm: 层归一化模块
        q_proj, k_proj, v_proj: Q、K、V 投影层
        attention_mask: 注意力掩码
        position_ids: 位置 ID
    
    Yields:
        AttentionContext: 包含 Q、K、V 和残差
    """
```

**作用**:
- 计算或检索 Q、K、V 状态
- 存储计算的 K/V 对以供后续重用
- 修改注意力掩码以实现特定的稀疏模式

**3. `ffn` - 前馈网络上下文**:
```python
@contextmanager
def ffn(self, layer_idx: int, x: torch.Tensor):
    """
    包装前馈网络计算
    
    Args:
        layer_idx: 层索引
        x: 输入张量
    
    Yields:
        FFNContext: 包含输入和残差
    """
```

**作用**:
- 检查或修改 FFN 的输入和输出
- 只对选中的 token 进行 FFN 计算

#### 3.4.4 生命周期钩子

```python
def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
    """每步开始时调用，准备掩码"""
    ...

def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
    """每步结束时调用，更新缓存"""
    ...

def on_block_start(self, block_mask: torch.Tensor, frame: Frame):
    """每个块开始时调用"""
    ...

def on_block_end(self, block_mask: torch.Tensor, frame: Frame, deltas: list[FrameDelta]):
    """每个块结束时调用"""
    ...
```

#### 3.4.5 上下文数据结构

**AttentionContext**:
```python
@dataclass
class AttentionContext:
    q: torch.Tensor                    # Query 向量
    k: torch.Tensor                    # Key 向量
    v: torch.Tensor                    # Value 向量
    residual: torch.Tensor             # 残差连接
    o: torch.Tensor | None             # 注意力输出（由模型赋值）
    attn_weight: torch.Tensor | None   # 注意力权重
    q_position_ids: torch.Tensor | None
    kv_position_ids: torch.Tensor | None
    attention_mask: torch.Tensor | None
```

**FFNContext**:
```python
@dataclass
class FFNContext:
    x: torch.Tensor              # FFN 输入
    residual: torch.Tensor       # 残差连接
    ffn_out: torch.Tensor | None # FFN 输出（由模型赋值）
```

**ModelForwardContext**:
```python
@dataclass
class ModelForwardContext:
    x: torch.Tensor               # 模型输入
    logits: torch.Tensor | None   # 模型输出（由模型赋值）
```

### 3.5 d2Cache - 双自适应缓存实现

`d2Cache` 是本项目的核心创新，实现了双自适应缓存策略。

#### 3.5.1 核心思想

d2Cache 采用两阶段细粒度选择策略：

**阶段 1: 基于确定性密度的选择**
- 计算每个 mask token 的确定性密度（certainty density）
- 优先选择确定性高的 token 进行重新计算
- 使用高斯核通过 FFT 计算密度

**阶段 2: 基于注意力滚动的选择**
- 使用注意力滚出（attention rollout）计算全局重要性
- 选择全局重要性高的 token 进行缓存更新
- 结合核选择（nucleus selection）策略

#### 3.5.2 数据结构定义

```python
class d2Cache(dCache):
    def __init__(
        self,
        model_config,
        rollout_p: float = 0.1,      # 注意力滚出的 top-p 比例
        current_k: int = 32,          # 每步更新的 token 数量
        sigma: float = 10.0,          # 确定性密度的高斯核标准差
        inflate_w: int = 4,           # 掩码膨胀的窗口大小
    ):
        super().__init__(model_config)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._conf_cache: torch.Tensor | None = None
        self._full_q_mask: torch.Tensor | None = None
        self._density_score: torch.Tensor
        self._global_importance: torch.Tensor
```

#### 3.5.3 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rollout_p` | `float` | 0.1 | 注意力滚出选择的 top-p 比例 |
| `current_k` | `int` | 32 | 每步更新 mask token 的数量 |
| `sigma` | `float` | 10.0 | 确定性密度计算的高斯核标准差 |
| `inflate_w` | `int` | 4 | 掩码膨胀的窗口大小 |

#### 3.5.4 核心方法

**确定性密度计算**:
```python
def certainty_density(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    使用高斯核通过 FFT 计算确定性密度
    
    Args:
        mask: 布尔张量，True 表示已生成的 token
        sigma: 高斯核标准差
    
    Returns:
        密度值张量
    """
```

**注意力滚出累积**:
```python
def accumulate_attn_rollout(self, attn_scores: torch.Tensor):
    """
    累积注意力滚出
    
    Args:
        attn_scores: 注意力分数，形状 (B, num_heads, q_len, seq_len)
    """
```

**掩码补全**:
```python
def top_up_mask(self, q_mask: torch.Tensor) -> torch.Tensor:
    """
    补全查询掩码，确保每个序列有相同数量的选中 token
    
    Args:
        q_mask: 查询掩码
    
    Returns:
        补全后的掩码
    """
```

#### 3.5.5 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     d2Cache 工作流程                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  初始化缓存      │
                    │  第一次前向传播  │
                    └─────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │         对每个解码步骤循环            │
        └──────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ on_step_start    │        │ on_step_end      │
    │ 准备掩码          │        │ 更新缓存          │
    └──────────────────┘        └──────────────────┘
                │                           │
                │                           ▼
                │               ┌──────────────────────┐
                │               │ 1. 更新置信度缓存     │
                │               │ 2. 计算确定性密度     │
                │               │ 3. 选择 mask token    │
                │               │ 4. 累积注意力滚出     │
                │               │ 5. 全局重要性选择     │
                │               │ 6. 掩码膨胀           │
                │               └──────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌─────────────────┐
                    │  模型前向传播    │
                    │  应用缓存策略    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  返回结果        │
                    └─────────────────┘
```

### 3.6 数据流转机制

#### 3.6.1 完整的生成流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      扩散模型生成流程                              │
└─────────────────────────────────────────────────────────────────┘

1. 初始化阶段
   ┌──────────────┐
   │ 输入提示词    │
   └──────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ 创建初始 Frame        │
   │ - prompts: 提示词     │
   │ - generated_tokens:   │
   │   全部为 mask token   │
   │ - steps: 全部为 -1    │
   └──────────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ 创建 DecodeRecord     │
   │ - initial_frame       │
   │ - deltas: []          │
   └──────────────────────┘

2. 迭代解码阶段
   ┌──────────────────────────────────────┐
   │         对每个块循环（半自回归）       │
   └──────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │         对每个步骤循环                 │
   └──────────────────────────────────────┘
          │
          ├─► on_step_start (缓存准备)
          │
          ▼
   ┌──────────────────────┐
   │ generate_step        │
   │ - 模型前向传播        │
   │ - 采样 token         │
   │ - 选择解码位置        │
   └──────────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ 创建 FrameDelta      │
   │ - transfer_index     │
   │ - decoded_tokens     │
   │ - confidence         │
   └──────────────────────┘
          │
          ├─► on_step_end (缓存更新)
          │
          ▼
   ┌──────────────────────┐
   │ frame.apply_delta    │
   │ 生成新的 Frame        │
   └──────────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ record.append(delta) │
   └──────────────────────┘
          │
          └─► 继续循环，直到所有 mask token 被解码

3. 结果返回
   ┌──────────────────────┐
   │ 返回 DecodeRecord    │
   │ - 包含完整解码轨迹    │
   └──────────────────────┘
```

#### 3.6.2 缓存机制的数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                     缓存机制数据流                                │
└─────────────────────────────────────────────────────────────────┘

模型前向传播
          │
          ▼
   ┌──────────────────────┐
   │ model_forward 上下文  │
   │ - 选择子集 token      │
   │ - 准备输入嵌入        │
   └──────────────────────┘
          │
          ▼
   对每一层循环
          │
          ├──────────────────────────┐
          │                          │
          ▼                          ▼
   ┌─────────────────┐    ┌─────────────────┐
   │ attention 上下文 │    │ ffn 上下文       │
   │ - 计算 Q、K、V   │    │ - FFN 计算       │
   │ - 更新缓存       │    │ - 只对选中 token │
   │ - 恢复完整序列   │    └─────────────────┘
   └─────────────────┘
          │
          ▼
   ┌──────────────────────┐
   │ 恢复完整 logits      │
   │ - 形状 (B, T, V)     │
   └──────────────────────┘
          │
          ▼
   返回给生成步骤
```

#### 3.6.3 批量处理机制

```
单序列模式:
  Frame:
    prompts: (prompt_length,)
    generated_tokens: (gen_length,)
    
  FrameDelta:
    transfer_index: torch.Tensor (num_transferred,)
    decoded_tokens: (gen_length,)

批量模式:
  Frame:
    prompts: (batch_size, prompt_length)
    generated_tokens: (batch_size, gen_length)
    
  FrameDelta:
    transfer_index: tuple[torch.Tensor, ...]  # 每个 batch 一个张量
    decoded_tokens: (num_active_sequences, gen_length)
```

---

## 4. 扩展开发指南

### 4.1 实现自定义缓存机制

#### 4.1.1 基本步骤

1. **继承 `dCache` 基类**
2. **重写关键方法**
3. **创建配置文件**
4. **测试和验证**

#### 4.1.2 完整实现示例

```python
# src/cache/my_cache.py
import torch
import torch.nn as nn
from contextlib import contextmanager
from src.cache.base import dCache, AttentionContext
from src.frame import Frame, FrameDelta


class MyCache(dCache):
    """
    自定义缓存实现示例
    """
    
    def __init__(
        self, 
        model_config,
        my_param: float = 0.5,  # 自定义参数
    ):
        super().__init__(model_config)
        self.my_param = my_param
        
        # 初始化缓存存储
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
    @contextmanager
    def model_forward(self, x: torch.Tensor):
        """
        包装模型前向传播
        """
        with super().model_forward(x=x) as ctx:
            B, T, C = x.shape
            
            # 自定义逻辑：选择需要重新计算的 token
            if self.active_q_mask is not None:
                # 只处理选中的 token
                ctx.x = x[self.active_q_mask].view(B, -1, C)
            
            yield ctx
            
            # 恢复完整的 logits
            if self.active_q_mask is not None:
                assert ctx.logits is not None
                ctx.logits = torch.zeros(
                    (B, T, ctx.logits.size(-1)),
                    dtype=ctx.logits.dtype,
                    device=ctx.logits.device,
                ).masked_scatter_(self.active_q_mask.unsqueeze(-1), ctx.logits)
    
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
        包装注意力计算
        """
        with super().attention(
            layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
            attention_mask, position_ids
        ) as ctx:
            
            # 第一次前向传播：存储 KV
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(ctx.k)
                self.value_cache.append(ctx.v)
            else:
                # 后续前向传播：更新缓存
                # 自定义更新逻辑
                self.key_cache[layer_idx] = ctx.k
                self.value_cache[layer_idx] = ctx.v
                
                # 使用缓存
                ctx.k = self.key_cache[layer_idx]
                ctx.v = self.value_cache[layer_idx]
            
            # 缓存共享变量
            if layer_idx == 0:
                self._q_position_ids, self._kv_position_ids = \
                    AttentionContext.select_position_ids(
                        position_ids, self.active_q_mask
                    )
                self._attention_mask = AttentionContext.convert_attention_mask(
                    attention_mask,
                    dtype=ctx.k.dtype,
                    query_length=ctx.q.shape[1],
                    key_value_length=self.value_cache[layer_idx].shape[1],
                )
            
            ctx.q_position_ids = self._q_position_ids
            ctx.kv_position_ids = self._kv_position_ids
            ctx.attention_mask = self._attention_mask
            
            yield ctx
    
    def on_step_start(self, block_mask: torch.Tensor, frame: Frame):
        """
        每步开始时的准备工作
        """
        # 自定义逻辑：准备 active_q_mask
        pass
    
    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """
        每步结束时的缓存更新
        """
        # 自定义逻辑：更新缓存状态
        pass
```

#### 4.1.3 创建配置文件

```yaml
# configs/cache/my_cache.yaml
_target_: src.cache.MyCache
my_param: 0.5
```

#### 4.1.4 使用自定义缓存

```bash
# 命令行使用
python eval.py \
    cache=my_cache \
    cache.my_param=0.7 \
    model=llada-inst \
    dataset.name=gsm8k
```

### 4.2 开发新的解码策略

#### 4.2.1 基本步骤

1. **使用 `@register` 装饰器注册策略**
2. **实现生成函数**
3. **创建配置文件**
4. **测试和验证**

#### 4.2.2 完整实现示例

```python
# src/generation/my_strategy.py
import torch
from typing import Type
from src.frame import Frame, FrameDelta, DecodeRecord
from src.cache import dCache
from src.generation.utils import register
from src.generation.vanilla import generate_step


@register("my_strategy")
def my_strategy_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    num_transfer_tokens: int = 1,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
    stop_until_eos: bool = False,
    # 自定义参数
    my_param: float = 0.5,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    自定义解码策略实现
    
    Args:
        model: 模型
        input_ids: 输入 token ID
        gen_length: 生成长度
        num_transfer_tokens: 每步解码的 token 数量
        temperature: 采样温度
        mask_token_id: mask token ID
        eos_token_id: 结束 token ID
        stop_until_eos: 是否在遇到 EOS 时停止
        my_param: 自定义参数
        cache_cls: 缓存类
    
    Returns:
        DecodeRecord: 解码记录
    """
    
    # 1. 初始化
    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)
    
    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame
    deltas = []
    
    # 2. 自定义解码逻辑
    def my_unmasking_fn(
        active_seq_idx: torch.Tensor,
        scores: torch.Tensor,
        probs: torch.Tensor,
        transfer_index_mask: torch.Tensor,
        block_mask: torch.Tensor,
        num_transfer_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        """
        自定义的 token 选择逻辑
        """
        # 实现你的选择策略
        # 例如：基于置信度和自定义参数选择 token
        
        batch_size = scores.size(0)
        transfer_index = []
        
        for i in range(batch_size):
            # 自定义选择逻辑
            # 这里是一个示例：选择 top-k token
            _, indices = torch.topk(
                scores[i][transfer_index_mask[i]], 
                k=num_transfer_tokens
            )
            transfer_index.append(indices)
        
        return tuple(transfer_index), {"my_param": my_param}
    
    # 3. 迭代生成
    block_mask = torch.ones(
        (input_ids.size(0), gen_length),
        dtype=torch.bool,
        device=model.device,
    )
    
    while True:
        if cache is not None:
            cache.on_step_start(block_mask, frame)
        
        delta = generate_step(
            model=model,
            frame=frame,
            block_mask=block_mask,
            num_transfer_tokens=num_transfer_tokens,
            unmasking_fn=my_unmasking_fn,
            past_key_values=cache,
            temperature=temperature,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            stop_until_eos=stop_until_eos,
        )
        
        if delta is None:
            break
        
        if cache is not None:
            cache.on_step_end(block_mask, frame, delta)
        
        deltas.append(delta.to("cpu"))
        frame = frame.apply_delta(delta)
    
    # 4. 返回结果
    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
    )
```

#### 4.2.3 创建配置文件

```yaml
# configs/generation/my_strategy.yaml
name: my_strategy
gen_length: 256
num_transfer_tokens: 1
temperature: 0.0
my_param: 0.5
```

#### 4.2.4 使用自定义策略

```bash
# 命令行使用
python eval.py \
    generation=my_strategy \
    generation.my_param=0.7 \
    model=llada-inst \
    dataset.name=gsm8k
```

### 4.3 添加新模型支持

#### 4.3.1 基本步骤

1. **创建模型配置类**
2. **创建模型类**
3. **修改模型以支持缓存**
4. **创建评估模型封装**
5. **更新模型加载函数**

#### 4.3.2 模型配置类

```python
# src/models/my_model/configuration_my_model.py
from transformers import PretrainedConfig


class MyModelConfig(PretrainedConfig):
    """
    自定义模型配置
    """
    
    model_type = "my_model"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        # ... 其他参数
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # ...
```

#### 4.3.3 模型类（支持缓存）

```python
# src/models/my_model/modeling_my_model.py
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from src.cache import dCache
from .configuration_my_model import MyModelConfig


class MyModelBlock(nn.Module):
    """
    模型块，需要应用缓存上下文管理器
    """
    
    def __init__(self, config: MyModelConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_values: dCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        # 应用注意力上下文管理器
        with past_key_values.attention(
            self.layer_id, x, self.attn_norm,
            self.q_proj, self.k_proj, self.v_proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ) as ctx:
            # 手动实现注意力计算
            # ctx.q, ctx.k, ctx.v 已经准备好
            attn_output = self.attention(ctx.q, ctx.k, ctx.v, ctx.attention_mask)
            ctx.o = self.o_proj(attn_output)
        
        x = ctx.residual + ctx.o
        
        # 应用 FFN 上下文管理器
        with past_key_values.ffn(self.layer_id, x) as ctx:
            ctx.ffn_out = self.ffn(ctx.x)
        
        x = ctx.residual + ctx.ffn_out
        return x
    
    def attention(self, q, k, v, mask):
        # 实现注意力计算
        # ...
        pass


class MyModel(PreTrainedModel):
    """
    自定义模型
    """
    
    config_class = MyModelConfig
    
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MyModelBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: dCache | None = None,
        use_cache: bool = False,
    ):
        x = self.embed_tokens(input_ids)
        
        # 应用模型前向传播上下文管理器
        with past_key_values.model_forward(x) as ctx:
            x = ctx.x
            
            for layer in self.layers:
                x = layer(
                    x, 
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            
            x = self.norm(x)
            logits = self.lm_head(x)
            ctx.logits = logits
        
        return ctx.logits
```

#### 4.3.4 评估模型封装

```python
# src/models/my_model/eval_model.py
from src.models.eval_mdlm import EvalModel


class MyModelEval(EvalModel):
    """
    评估模型封装
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # 初始化模型和分词器
        # ...
    
    def generate(self, input_ids, **kwargs):
        # 实现生成逻辑
        # ...
        pass
```

#### 4.3.5 更新模型加载函数

```python
# src/utils/models.py
def load_pretrained_model(cfg: DictConfig, **model_kwargs) -> PreTrainedModel:
    from ..models import LLaDAModelLM, DreamModel, MyModel
    
    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        model = LLaDAModelLM.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "dream":
        model = DreamModel.from_pretrained(cfg.model.path, **model_kwargs)
    elif model_family == "my_model":
        model = MyModel.from_pretrained(cfg.model.path, **model_kwargs)
    else:
        raise ValueError(f"Unsupported pretrained model: {cfg.model.name}")
    
    return model


def load_eval_model(cfg: DictConfig, **model_kwargs):
    from ..models import LLaDAEval, DreamEval, MyModelEval
    
    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        eval_model = LLaDAEval(cfg, **model_kwargs)
    elif model_family == "dream":
        eval_model = DreamEval(cfg, **model_kwargs)
    elif model_family == "my_model":
        eval_model = MyModelEval(cfg, **model_kwargs)
    else:
        raise NotImplementedError(
            f"Model family {model_family} is not implemented for evaluation."
        )
    
    return eval_model
```

### 4.4 配置文件使用指南

#### 4.4.1 配置文件结构

```yaml
# configs/model/my_model-base.yaml
name: my_model-base
path: /path/to/my_model-base

# configs/cache/my_cache.yaml
_target_: src.cache.MyCache
my_param: 0.5

# configs/generation/my_strategy.yaml
name: my_strategy
gen_length: 256
num_transfer_tokens: 1
my_param: 0.5
```

#### 4.4.2 命令行覆盖

```bash
# 基本使用
python eval.py \
    model=my_model-base \
    cache=my_cache \
    generation=my_strategy

# 覆盖参数
python eval.py \
    model=my_model-base \
    cache=my_cache \
    cache.my_param=0.7 \
    generation=my_strategy \
    generation.gen_length=512

# 多参数组合
python eval.py \
    model=llada-inst \
    cache=d2cache \
    cache.rollout_p=0.15 \
    cache.current_k=64 \
    cache.sigma=15.0 \
    generation=vanilla \
    generation.num_transfer_tokens=2 \
    generation.sigma=12 \
    dataset.name=gsm8k \
    batch_size=4
```

#### 4.4.3 动态参数生成

```python
# configs/gen_args.py
def get_gen_args(cfg):
    """
    根据配置动态生成参数
    
    Args:
        cfg: Hydra 配置对象
    
    Returns:
        dict: 动态生成的参数
    """
    extra_gen_kwargs = {}
    
    # 根据模型类型设置参数
    model_family = cfg.model.name.split("-")[0]
    if model_family == "llada":
        extra_gen_kwargs["mask_token_id"] = 126081
        extra_gen_kwargs["eos_token_id"] = 126081
    elif model_family == "dream":
        extra_gen_kwargs["mask_token_id"] = 151643
        extra_gen_kwargs["eos_token_id"] = 151643
    
    # 根据数据集设置参数
    if cfg.dataset.name == "humaneval":
        extra_gen_kwargs["gen_length"] = 512
        extra_gen_kwargs["block_length"] = 64
    elif cfg.dataset.name == "gsm8k":
        extra_gen_kwargs["gen_length"] = 256
        extra_gen_kwargs["block_length"] = 32
    
    return extra_gen_kwargs
```

---

## 5. 代码示例与最佳实践

### 5.1 缓存机制实现示例

#### 5.1.1 PrefixCache 实现要点

```python
# PrefixCache 的核心逻辑
class PrefixCache(dCache):
    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """
        PrefixCache 的关键：在块开始时确定查询掩码
        """
        if self.active_q_mask is None:
            # 创建查询掩码
            q_mask = F.pad(block_mask, (frame.prompts.size(-1), 0), value=False)
            
            # 如果不使用 dual cache，则包含块之后的所有 token
            if not self.use_dual:
                block_start = int(block_mask[0].int().argmax() + 1)
                q_mask[:, frame.prompts.size(-1) + block_start :] = True
            
            # 对于从 AR 改编的模型，需要特殊处理
            if is_adapted_from_ar(self.model_config):
                q_mask = F.pad(q_mask[:, 1:], (0, 1), value=False)
            
            self.active_q_mask = q_mask
```

#### 5.1.2 d2Cache 实现要点

```python
# d2Cache 的核心逻辑
class d2Cache(dCache):
    def on_step_end(self, block_mask: torch.Tensor, frame: Frame, delta: FrameDelta):
        """
        d2Cache 的关键：两阶段选择策略
        """
        # 1. 更新置信度缓存
        if self._conf_cache is None:
            self._conf_cache = confidence
        
        # 2. 计算确定性密度
        scores = self._conf_cache[self.active_seq_mask] * certainty_density(
            ~remaining_mask, self.sigma
        )
        
        # 3. 选择 mask token（基于确定性密度）
        _, indices = torch.topk(
            torch.where(search_mask & remaining_mask, scores, -torch.inf),
            k=min(self.current_k, remaining_mask.size(-1)),
            dim=-1,
        )
        selected_mask = torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(
            1, indices, True
        ) & remaining_mask
        
        # 4. 累积注意力滚出
        global_importance = self._attn_rollout.sum(dim=1)
        
        # 5. 全局重要性选择
        q_mask = F.pad(response_mask, (P, 0), value=False)
        q_mask |= nucleus_select(global_importance, self.rollout_p, mask=~q_mask)
        
        # 6. 掩码膨胀
        if self.inflate_w > 0:
            # ... 膨胀逻辑
            pass
```

### 5.2 解码策略实现示例

#### 5.2.1 Vanilla 解码核心逻辑

```python
@register("vanilla")
def vanilla_generate(
    model,
    input_ids: torch.Tensor,
    block_length: int = 32,
    gen_length: int = 128,
    num_transfer_tokens: int = 1,
    # ...
):
    """
    Vanilla 解码：标准扩散解码，支持半自回归
    """
    # 初始化
    initial_frame = Frame.create_initial_frame(input_ids, gen_length, mask_token_id)
    frame = initial_frame
    deltas = []
    
    # 半自回归：分块处理
    num_blocks = gen_length // block_length
    
    for block_idx in range(num_blocks):
        # 创建块掩码
        block_mask = torch.zeros((input_ids.size(0), gen_length), dtype=torch.bool)
        block_mask[:, block_idx * block_length : (block_idx + 1) * block_length] = True
        
        # 块内迭代解码
        while True:
            delta = generate_step(
                model=model,
                frame=frame,
                block_mask=block_mask,
                num_transfer_tokens=num_transfer_tokens,
                # ...
            )
            
            if delta is None:
                break
            
            deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)
    
    return DecodeRecord(initial_frame=initial_frame.to("cpu"), deltas=deltas)
```

#### 5.2.2 并行解码实现

```python
def confidence_unmasking(
    scores: torch.Tensor,
    transfer_index_mask: torch.Tensor,
    min_transfer_tokens: int,
    threshold: float | None = None,
    factor: float | None = None,
):
    """
    并行解码：基于置信度选择多个 token
    """
    if threshold is not None:
        # 选择所有置信度高于阈值的 token
        col_indices = torch.nonzero(scores >= threshold, as_tuple=False)[:, 1]
        counts = torch.sum(scores >= threshold, dim=-1).cpu().tolist()
        transfer_index = list(torch.split(col_indices, counts))
    
    elif factor is not None:
        # 基于因子的自适应选择
        for i in range(batch_size):
            sorted_conf, _ = torch.sort(
                scores[i][transfer_index_mask[i]], dim=-1, descending=True
            )
            for n in range(1, num_unmasked_tokens[i] + 1):
                if (n + 1) * (1 - sorted_conf[n - 1]) >= factor:
                    break
            transfer_index[i] = torch.topk(scores[i], min(n - 1, max_transfer_tokens[i]), dim=-1).indices
    
    return tuple(transfer_index)
```

### 5.3 配置文件示例

#### 5.3.1 完整评估配置

```yaml
# configs/eval.yaml
defaults:
    - model: llada-inst
    - generation: vanilla
    - cache: d2cache
    - _self_

seed: 42
gen_args_script: configs/gen_args.py
batch_size: 1
attn_implementation: eager  # 必须为 d2Cache 设置为 eager

dataset:
    name: gsm8k
    batch_size: 1

generation:
    gen_length: 256
    block_length: 32
    num_transfer_tokens: 1
    temperature: 0.0
    sigma: 10  # 确定性先验引导

cache:
    rollout_p: 0.1
    current_k: 32
    sigma: 10
    inflate_w: 4
```

#### 5.3.2 模型配置

```yaml
# configs/model/llada-inst.yaml
name: llada-inst
path: /path/to/LLaDA-8B-Instruct

# 模型特定参数
vocab_size: 128256
hidden_size: 4096
num_hidden_layers: 32
num_attention_heads: 32
```

#### 5.3.3 缓存配置

```yaml
# configs/cache/d2cache.yaml
_target_: src.cache.d2Cache
rollout_p: 0.1
current_k: 32
sigma: 10.0
inflate_w: 4
```

### 5.4 常见问题与解决方案

#### 5.4.1 缓存相关问题

**问题 1: 缓存形状不匹配**

```python
# 错误信息
RuntimeError: The attention output shape (B, q_len, d_model) is not compatible 
with the residual shape (B, seq_len, d_model)

# 解决方案
# 确保在 attention 上下文管理器中正确设置 residual
with past_key_values.attention(...) as ctx:
    # ctx.residual 应该与 ctx.o 形状相同
    ctx.o = self.o_proj(attn_output)
    # 检查形状
    assert ctx.residual.shape == ctx.o.shape
```

**问题 2: 注意力权重未输出**

```python
# 错误信息
AssertionError: The attention weights must be outputed, 
make sure you've set attn_implementation="eager"

# 解决方案
# 在命令行中设置 attn_implementation
python eval.py attn_implementation=eager cache=d2cache

# 或在配置文件中设置
attn_implementation: eager
```

#### 5.4.2 解码相关问题

**问题 1: 生成步数过多**

```python
# 问题：生成步数远超预期

# 解决方案 1: 调整 num_transfer_tokens
generation.num_transfer_tokens=2  # 每步解码更多 token

# 解决方案 2: 使用并行解码
generation.threshold=0.9  # 置信度阈值

# 解决方案 3: 使用更高效的缓存
cache=d2cache  # d2Cache 可以减少重计算
```

**问题 2: 生成质量下降**

```python
# 问题：使用缓存后生成质量下降

# 解决方案 1: 调整缓存参数
cache.rollout_p=0.15  # 增加全局重要性选择的 token 数量
cache.current_k=64    # 增加每步更新的 token 数量

# 解决方案 2: 使用确定性先验引导
generation.sigma=10  # 启用确定性先验

# 解决方案 3: 禁用缓存进行对比
cache=null  # 禁用缓存
```

#### 5.4.3 模型相关问题

**问题 1: 模型加载失败**

```python
# 错误信息
ValueError: Unsupported pretrained model: my_model

# 解决方案
# 1. 检查模型名称格式
model=my_model-base  # 必须以模型家族名开头

# 2. 确保模型路径正确
# configs/model/my_model-base.yaml
path: /correct/path/to/my_model

# 3. 检查模型是否已注册
# src/utils/models.py 中必须包含加载逻辑
```

**问题 2: 自定义模型不支持缓存**

```python
# 问题：自定义模型无法使用缓存

# 解决方案：确保模型正确应用上下文管理器
class MyModelBlock(nn.Module):
    def forward(self, x, past_key_values=None, ...):
        # 必须应用 attention 和 ffn 上下文管理器
        with past_key_values.attention(...) as ctx:
            # 注意力计算
            ctx.o = ...
        
        with past_key_values.ffn(...) as ctx:
            # FFN 计算
            ctx.ffn_out = ...
```

#### 5.4.4 性能优化建议

1. **批量大小优化**
```bash
# 对于小模型，可以增加批量大小
batch_size=4

# 对于大模型或长序列，减小批量大小
batch_size=1
```

2. **缓存参数调优**
```bash
# 快速生成（牺牲一些质量）
cache.rollout_p=0.05 cache.current_k=16

# 高质量生成（牺牲一些速度）
cache.rollout_p=0.2 cache.current_k=64
```

3. **解码策略选择**
```bash
# 快速生成
generation.threshold=0.8  # 并行解码

# 高质量生成
generation.sigma=10  # 确定性先验引导
generation.num_transfer_tokens=1  # 保守解码
```

---

## 6. 快速参考

### 6.1 核心类索引

| 类名 | 模块 | 说明 |
|------|------|------|
| `Frame` | `src.frame` | 生成状态容器 |
| `FrameDelta` | `src.frame` | 状态变化描述 |
| `DecodeRecord` | `src.frame` | 解码轨迹记录 |
| `dCache` | `src.cache.base` | 缓存抽象基类 |
| `d2Cache` | `src.cache.d2cache` | 双自适应缓存 |
| `PrefixCache` | `src.cache.prefix_cache` | 前缀缓存 |
| `dLLMCache` | `src.cache.dllm_cache` | dLLM 缓存 |
| `AttentionContext` | `src.cache.base` | 注意力上下文 |
| `FFNContext` | `src.cache.base` | FFN 上下文 |
| `Registry` | `src.utils.common` | 注册机制 |

### 6.2 配置参数速查

#### 6.2.1 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gen_length` | `int` | 128 | 生成长度 |
| `block_length` | `int` | 32 | 块长度（半自回归） |
| `num_transfer_tokens` | `int` | 1 | 每步解码 token 数 |
| `temperature` | `float` | 0.0 | 采样温度 |
| `top_k` | `int` | None | Top-k 采样 |
| `top_p` | `float` | None | Nucleus 采样 |
| `sigma` | `float` | None | 确定性先验参数 |

#### 6.2.2 缓存参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rollout_p` | `float` | 0.1 | 注意力滚出 top-p |
| `current_k` | `int` | 32 | 每步更新 token 数 |
| `sigma` | `float` | 10.0 | 确定性密度标准差 |
| `inflate_w` | `int` | 4 | 掩码膨胀窗口 |
| `use_dual` | `bool` | False | 是否使用 dual cache |

### 6.3 命令行示例

#### 6.3.1 基本评估

```bash
# GSM8K 评估
python eval.py \
    model=llada-inst \
    dataset.name=gsm8k \
    batch_size=1

# HumanEval 评估
python eval.py \
    model=dream-inst \
    dataset.name=humaneval_instruct \
    batch_size=1
```

#### 6.3.2 使用缓存

```bash
# d2Cache
python eval.py \
    model=llada-inst \
    cache=d2cache \
    attn_implementation=eager \
    dataset.name=gsm8k

# PrefixCache
python eval.py \
    model=llada-base \
    cache=prefix \
    dataset.name=humaneval
```

#### 6.3.3 并行解码

```bash
# 置信度阈值
python eval.py \
    model=llada-inst \
    generation=vanilla \
    generation.threshold=0.9 \
    dataset.name=gsm8k

# 因子自适应
python eval.py \
    model=llada-inst \
    generation=vanilla \
    generation.factor=1.0 \
    dataset.name=gsm8k
```

#### 6.3.4 完整示例

```bash
# d2Cache + 确定性先验 + 并行解码
python eval.py \
    model=llada-inst \
    cache=d2cache \
    cache.rollout_p=0.1 \
    cache.current_k=32 \
    cache.sigma=10 \
    cache.inflate_w=4 \
    generation=vanilla \
    generation.gen_length=256 \
    generation.block_length=32 \
    generation.num_transfer_tokens=1 \
    generation.sigma=10 \
    generation.threshold=0.9 \
    attn_implementation=eager \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=42
```

---

## 附录

### A. 项目架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        d2Cache 项目架构                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   eval.py   │────►│   configs/  │────►│  tasks/     │
│  评估入口    │     │  配置系统    │     │  评估任务    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                        src/ 核心模块                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ frame.py │  │  cache/  │  │generation│  │ models/  │  │
│  │ 数据结构  │  │ 缓存机制  │  │ 解码策略  │  │ 模型定义  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────┐                            │
│                    │  utils/  │                            │
│                    │ 工具函数  │                            │
│                    └──────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### B. 数据流转图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流转示意图                              │
└─────────────────────────────────────────────────────────────────┘

输入提示词
     │
     ▼
┌─────────────┐
│Frame.create │
│_initial_frame│
└─────────────┘
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frame     │────►│generate_step│────►│ FrameDelta  │
│  当前状态    │     │  模型前向    │     │  状态变化    │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                    │
     │                    │                    │
     │                    ▼                    │
     │              ┌─────────────┐            │
     │              │   dCache    │            │
     │              │  缓存机制    │            │
     │              └─────────────┘            │
     │                    │                    │
     └────────────────────┴────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ DecodeRecord│
                   │  解码记录    │
                   └─────────────┘
                          │
                          ▼
                     最终结果
```

### C. 参考文献

1. **LLaDA**: Nie, S., et al. "Large Language Diffusion Models." arXiv:2502.09992 (2025).
2. **Dream**: Dream Team. "Dream: Diffusion-based Language Model." arXiv:2508.15487 (2025).
3. **Fast-dLLM**: Wu, C., et al. "Fast-dLLM: Training-free Acceleration of Diffusion LLM." arXiv:2505.22618 (2025).
4. **dLLM-Cache**: Liu, Z., et al. "dLLM-Cache: Accelerating Diffusion Large Language Models." arXiv:2506.06295 (2025).
5. **d2Cache**: Jiang, Y., et al. "d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching." arXiv:2509.23094 (2025).
6. **PC-Sampler**: Huang, P., et al. "PC-Sampler: Position-Aware Calibration." arXiv:2508.13021 (2025).
7. **KLASS**: Kim, S.H., et al. "KLASS: KL-Guided Fast Inference." arXiv:2511.05664 (2025).
8. **EB-Sampler**: Ben-Hamu, H., et al. "Accelerated Sampling via Entropy Bounded Unmasking." arXiv:2505.24857 (2025).
9. **WINO**: Hong, F., et al. "Wide-In, Narrow-Out: Revokable Decoding." arXiv:2507.18578 (2025).

---

**文档结束**

本文档提供了 d2Cache 项目的完整代码分析，包括项目概览、分级结构、关键数据结构、扩展开发指南和实用示例。希望这份文档能帮助研究人员和开发者快速理解和使用本项目。

如有问题或建议，请访问项目的 [GitHub Issues](https://github.com/Kamichanw/d2Cache/issues) 或 [Discussions](https://github.com/Kamichanw/d2Cache/discussions)。
