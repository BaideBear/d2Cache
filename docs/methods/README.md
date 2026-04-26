# d2Cache 方法文档

本目录包含 d2Cache 项目中所有缓存方法和解码策略的详细文档。

## 📚 文档组织结构

```
methods/
├── README.md                    # 本文档（顶层索引）
├── caches/                      # 缓存方法文档
│   ├── prefix_cache.md         # PrefixCache / DualCache
│   ├── dllm_cache.md           # dLLM-Cache
│   └── d2cache.md              # d2Cache（本项目核心）
├── decoders/                    # 解码策略文档
│   ├── vanilla.md              # Vanilla / Semi-AR 解码
│   ├── parallel.md             # 并行解码
│   ├── pc_sampler.md           # PC-Sampler
│   ├── klass.md                # KLASS
│   ├── eb_sampler.md           # EB-Sampler
│   ├── wino.md                 # WINO
│   └── ar.md                   # 自回归解码
└── guides/                      # 扩展开发指南
    ├── cache_development.md    # 如何实现新的缓存方法
    └── decoder_development.md  # 如何实现新的解码策略
```

## 🔍 快速导航

### 缓存方法

| 方法 | 文档 | 特点 | 适用场景 |
|------|------|------|----------|
| **PrefixCache** | [caches/prefix_cache.md](caches/prefix_cache.md) | 块级近似 KV 缓存，支持双向注意力 | 快速推理，中等质量要求 |
| **dLLM-Cache** | [caches/dllm_cache.md](caches/dllm_cache.md) | 自适应缓存，特征相似度引导 | 长序列生成，动态更新 |
| **d2Cache** | [caches/d2cache.md](caches/d2cache.md) | 双自适应缓存，两阶段选择策略 | 高质量生成，速度与质量平衡 |

### 解码策略

| 策略 | 文档 | 特点 | 适用场景 |
|------|------|------|----------|
| **Vanilla** | [decoders/vanilla.md](decoders/vanilla.md) | 标准扩散解码，支持半自回归 | 通用生成任务 |
| **Parallel** | [decoders/parallel.md](decoders/parallel.md) | 置信度感知并行解码 | 快速生成 |
| **PC-Sampler** | [decoders/pc_sampler.md](decoders/pc_sampler.md) | 位置感知校准采样 | 逻辑推理任务 |
| **KLASS** | [decoders/klass.md](decoders/klass.md) | KL 引导快速推理 | 推理加速 |
| **EB-Sampler** | [decoders/eb_sampler.md](decoders/eb_sampler.md) | 熵有界解掩码 | 自适应步数 |
| **WINO** | [decoders/wino.md](decoders/wino.md) | 可撤销解码机制 | 质量敏感任务 |
| **AR** | [decoders/ar.md](decoders/ar.md) | 传统自回归解码 | 基线对比 |

### 扩展开发

| 指南 | 文档 | 内容 |
|------|------|------|
| **缓存扩展** | [guides/cache_development.md](guides/cache_development.md) | 如何实现新的缓存方法 |
| **解码扩展** | [guides/decoder_development.md](guides/decoder_development.md) | 如何实现新的解码策略 |

## 📖 文档内容说明

每种方法的文档都包含以下核心内容：

1. **算法逻辑精要**: 3-5 句话高度概括算法的主体逻辑，帮助读者在几秒内快速建立全局认知
2. **算法原理**: 详细介绍算法的理论基础和核心思想
3. **核心参数**: 列出关键参数及其作用
4. **详细代码流程分析**: 按源码文件的行号顺序，逐模块/逐行讲解每行代码的作用；每个代码块附带源文件路径和行号标注，方便对照源码阅读
5. **关键函数**: 解释重要函数和上下文管理器
6. **使用示例**: 提供完整的配置和使用示例
7. **性能特点**: 分析性能优势和适用场景

## 🚀 快速开始

### 选择缓存方法

```bash
# PrefixCache - 快速推理
python eval.py cache=prefix model=llada-base dataset.name=humaneval

# dLLM-Cache - 长序列
python eval.py cache=dllm model=dream-inst dataset.name=gsm8k

# d2Cache - 高质量（推荐）
python eval.py cache=d2cache model=llada-inst dataset.name=gsm8k attn_implementation=eager
```

### 选择解码策略

```bash
# Vanilla - 标准解码
python eval.py generation=vanilla model=llada-inst dataset.name=gsm8k

# Parallel - 快速生成
python eval.py generation=vanilla generation.threshold=0.9 model=llada-inst

# KLASS - 推理加速
python eval.py generation=klass model=llada-inst dataset.name=gsm8k
```

## 🔗 相关资源

- [项目主页](../README.md)
- [代码分析文档](../code_analysis.md)
- [代码阅读指南](../code_reading_guides.md)
- [自定义开发指南](../customization.md)

## 📝 文档贡献

如果您发现文档有误或需要补充，欢迎：
1. 提交 Issue
2. 提交 Pull Request
3. 在 Discussions 中讨论

---

**最后更新**: 2026-04-25
