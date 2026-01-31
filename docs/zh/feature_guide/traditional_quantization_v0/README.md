---
toc_depth: 3
---
# 传统量化与模型压缩（V0）

传统量化（V0）是 msModelSlim 的早期服务，基于脚本化的方式实现量化、剪枝、蒸馏等功能。它提供了更细粒度的 API 控制，适合有特殊定制需求的开发者。

> **注意**：modelslim_v0 协议版本即将废弃，不推荐在新项目中使用。建议优先参考 [一键量化（V1）](../02-one_click_quantization_v1/usage.md)。

## 目录

### 1. PyTorch 框架支持
- [大模型脚本量化](pytorch_foundation_model_quant.md): 针对 Llama、ChatGLM 等大语言模型的量化指南，包含 FA3 和低显存模式。
- [常规模型与多模态量化](pytorch_common_model_quantization.md): 针对 ResNet 等卷积模型以及 SD3 等生成模型的量化指南。
- [模型压缩与优化](pytorch_model_optimization.md): 包含剪枝、低秩分解和蒸馏等技术。

### 2. 其他框架支持
- [MindSpore 脚本量化](mindspore_post_training_quantization.md): 针对 MindSpore 框架的脚本量化指南。
- [ONNX 脚本量化](onnx_post_training_quantization.md): 针对 ONNX 模型的脚本量化指南。

## 核心概念

V0 服务主要由以下核心接口组成：
- **AntiOutlier**: 用于离群值抑制，提升量化精度。
- **Calibrator**: 用于量化校准、运行和保存。
- **PruneTorch**: 用于 PyTorch 模型的剪枝优化。
