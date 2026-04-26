# d2Cache 项目代码分析文档规格

## Why
d2Cache 是一个专注于扩散语言模型（Diffusion Language Models, dLLMs）研究的代码库，提供了 KV 缓存机制和解码策略的综合基线实现。创建详尽的代码分析文档可以帮助研究人员和开发者快速理解项目架构、核心数据结构以及扩展开发方法。

## What Changes
- 创建完整的项目架构分析文档
- 详细说明核心数据结构和类的设计
- 提供扩展开发的指南和最佳实践
- 包含代码示例和图表说明

## Impact
- Affected specs: 文档系统
- Affected code: 无代码修改，仅新增文档

## ADDED Requirements

### Requirement: 项目概览与架构分析
文档 SHALL 提供完整的项目概览，包括：
- 项目背景和研究目标
- 支持的模型列表（LLaDA、Dream）
- 支持的 KV 缓存方法（PrefixCache、dLLM-Cache、d2Cache）
- 支持的解码策略（Vanilla、Parallel、PC-Sampler、KLASS、EB-Sampler、WINO）

#### Scenario: 用户了解项目整体结构
- **WHEN** 用户阅读项目概览章节
- **THEN** 用户能够理解项目的核心功能和研究方向

### Requirement: 分级结构分析
文档 SHALL 详细说明项目的目录结构和各模块职责：
- `src/frame.py`: 核心数据结构（Frame、FrameDelta、DecodeRecord）
- `src/cache/`: KV 缓存实现模块
- `src/generation/`: 解码策略实现模块
- `src/models/`: 模型定义和修改
- `src/utils/`: 工具函数和辅助类
- `configs/`: 配置文件系统
- `eval.py`: 评估入口

#### Scenario: 开发者理解模块职责
- **WHEN** 开发者阅读分级结构章节
- **THEN** 开发者能够定位到需要修改或扩展的具体模块

### Requirement: 关键数据结构详解
文档 SHALL 深入分析核心数据结构的设计和实现：
- **Frame**: 存储生成状态的完整信息
- **FrameDelta**: 表示两步之间的变化
- **DecodeRecord**: 聚合整个解码轨迹
- **dCache**: 缓存机制的抽象基类
- **d2Cache**: 双自适应缓存的具体实现

#### Scenario: 开发者理解数据流转
- **WHEN** 开发者阅读数据结构章节
- **THEN** 开发者能够理解生成过程中数据的流转和状态管理

### Requirement: 扩展开发指南
文档 SHALL 提供清晰的扩展开发指南：
- 如何实现自定义缓存机制
- 如何开发新的解码策略
- 如何添加新模型支持
- 配置文件的使用方法

#### Scenario: 研究人员添加新功能
- **WHEN** 研究人员阅读扩展开发章节
- **THEN** 研究人员能够按照指南实现新的缓存方法或解码策略

### Requirement: 代码示例和最佳实践
文档 SHALL 包含实用的代码示例：
- 缓存机制的关键代码片段
- 解码策略的实现模式
- 配置文件的编写示例
- 常见问题和解决方案

#### Scenario: 开发者快速上手
- **WHEN** 开发者参考代码示例
- **THEN** 开发者能够快速实现类似功能

## MODIFIED Requirements
无修改的需求

## REMOVED Requirements
无移除的需求
