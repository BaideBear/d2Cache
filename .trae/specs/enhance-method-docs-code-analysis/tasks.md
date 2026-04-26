# Tasks

- [x] Task 1: 为缓存方法文档添加「算法逻辑精要」
  - [x] SubTask 1.1: 在 prefix_cache.md 开头添加算法逻辑精要
  - [x] SubTask 1.2: 在 dllm_cache.md 开头添加算法逻辑精要
  - [x] SubTask 1.3: 在 d2cache.md 开头添加算法逻辑精要

- [x] Task 2: 重新编写缓存方法文档的「详细代码流程分析」章节（逐行/逐模块讲解）
  - [x] SubTask 2.1: 重写 prefix_cache.md 的代码流程分析，逐模块讲解 prefix_cache.py (L1-L126)
  - [x] SubTask 2.2: 重写 dllm_cache.md 的代码流程分析，逐模块讲解 dllm_cache.py (L1-L220)
  - [x] SubTask 2.3: 重写 d2cache.md 的代码流程分析，逐模块讲解 d2cache.py (L1-L298)

- [x] Task 3: 为解码策略文档添加「算法逻辑精要」
  - [x] SubTask 3.1: 在 vanilla.md 开头添加算法逻辑精要
  - [x] SubTask 3.2: 在 parallel.md 开头添加算法逻辑精要
  - [x] SubTask 3.3: 在 pc_sampler.md 开头添加算法逻辑精要
  - [x] SubTask 3.4: 在 klass.md 开头添加算法逻辑精要
  - [x] SubTask 3.5: 在 eb_sampler.md 开头添加算法逻辑精要
  - [x] SubTask 3.6: 在 wino.md 开头添加算法逻辑精要
  - [x] SubTask 3.7: 在 ar.md 开头添加算法逻辑精要

- [x] Task 4: 重新编写解码策略文档的「详细代码流程分析」章节（逐行/逐模块讲解）
  - [x] SubTask 4.1: 重写 vanilla.md 的代码流程分析，逐模块讲解 vanilla.py
  - [x] SubTask 4.2: 重写 parallel.md 的代码流程分析，逐模块讲解（parallel 函数位于 vanilla.py 中）
  - [x] SubTask 4.3: 重写 pc_sampler.md 的代码流程分析（PC-Sampler 函数位于 vanilla.py 中）
  - [x] SubTask 4.4: 重写 klass.md 的代码流程分析，逐模块讲解 klass.py
  - [x] SubTask 4.5: 重写 eb_sampler.md 的代码流程分析（EB-Sampler 函数位于 vanilla.py 中）
  - [x] SubTask 4.6: 重写 wino.md 的代码流程分析，逐模块讲解 wino.py
  - [x] SubTask 4.7: 重写 ar.md 的代码流程分析，逐模块讲解 ar.py

- [x] Task 5: 更新 docs/methods/README.md 反映新的文档结构
  - [x] SubTask 5.1: 在 README 的「文档内容说明」中新增对「算法逻辑精要」的说明
  - [x] SubTask 5.2: 在 README 中更新对代码分析章节的描述强调逐行讲解特性

# Task Dependencies
- Task 2 depends on Task 1
- Task 4 depends on Task 3
- Task 5 depends on Task 2 and Task 4
- Tasks 1, 2, 3, 4 内部各 SubTask 之间无依赖，可以并行执行
- Task 1 与 Task 3 可以并行执行
