# Checklist

## 项目概览
- [x] 文档包含项目背景和研究目标的清晰说明
- [x] 支持的模型列表完整准确（LLaDA-8B、LLaDA-1.5、Dream-v0-7B）
- [x] 支持的 KV 缓存方法列表完整（PrefixCache、dLLM-Cache、d2Cache）
- [x] 支持的解码策略列表完整（Vanilla、Parallel、PC-Sampler、KLASS、EB-Sampler、WINO）
- [x] 说明了项目的核心优势（统一评估框架、清晰代码、活跃维护）

## 分级结构分析
- [x] 顶层目录结构说明完整（assets、configs、docs、requirements、scripts、src、tasks）
- [x] src/frame.py 模块职责说明清晰
- [x] src/cache/ 模块结构和职责说明清晰
- [x] src/generation/ 模块结构和职责说明清晰
- [x] src/models/ 模块结构和职责说明清晰
- [x] src/utils/ 工具函数说明清晰
- [x] configs/ 配置系统说明完整
- [x] eval.py 评估入口说明清晰

## 关键数据结构
- [x] Frame 类的设计和属性说明详细
- [x] FrameDelta 类的设计和属性说明详细
- [x] DecodeRecord 类的设计和属性说明详细
- [x] 数据流转机制说明清晰（从初始帧到最终结果）
- [x] dCache 基类的上下文管理器设计说明详细
- [x] d2Cache 的两阶段选择策略说明详细
- [x] 包含关键代码片段和注释

## 扩展开发指南
- [x] 自定义缓存机制的实现步骤清晰
- [x] 自定义解码策略的实现步骤清晰
- [x] 添加新模型支持的步骤清晰
- [x] 配置文件的使用方法说明详细
- [x] 包含完整的代码示例

## 代码示例和最佳实践
- [x] 缓存机制的关键代码示例完整可运行
- [x] 解码策略的实现模式示例清晰
- [x] 配置文件编写示例完整
- [x] 常见问题和解决方案实用

## 文档质量
- [x] 目录结构清晰，易于导航
- [x] 包含必要的图表和流程图
- [x] 术语使用统一
- [x] 格式规范，可读性强
- [x] 包含快速索引和参考链接
