---
name: 错误报告 Bug Report (中文)
about: 报告错误以帮助我们改进 FusionBench
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## 描述错误
清晰、简洁地描述错误是什么。

## 复现步骤
复现该问题的详细步骤：

1. **命令或脚本执行方式：**
   ```bash
   # 在此粘贴您的命令
   fusion_bench method=... modelpool=... taskpool=...
   ```

2. **使用的配置文件：**
   ```yaml
   # 在此粘贴相关的配置内容
   ```

3. **错误信息或日志：**
   ```
   请包含完整的堆栈跟踪信息，特别是报错的位置。
   如果从 fusion_bench CLI 启动，请设置环境变量 HYDRA_FULL_ERROR=1
   以获取完整的堆栈跟踪：
   
   HYDRA_FULL_ERROR=1 fusion_bench method=...
   ```

## 预期行为
清晰、简洁地描述您期望发生的情况。

## 环境信息
请提供以下信息：

- **操作系统：** [例如：Ubuntu 22.04, macOS 14.0, Windows 11]
- **Python 版本：** [例如：3.10.12]
- **FusionBench 版本：** [例如：0.2.29，或 git commit hash]
- **PyTorch 版本：** [例如：2.1.0]
- **CUDA 版本（如适用）：** [例如：11.8]
- **其他依赖库版本：** 
  ```bash
  # 运行: pip list | grep -E "transformers|datasets|hydra-core|lightning"
  ```

## 截图或日志
如果适用，添加截图或额外的日志信息以帮助解释问题。

## 附加信息
在此添加关于该问题的其他相关信息：
- 这个问题是持续发生还是偶尔出现？
- 您是否对代码进行了任何修改？
- 其他任何相关信息
