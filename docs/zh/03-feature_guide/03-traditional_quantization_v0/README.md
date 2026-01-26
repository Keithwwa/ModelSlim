# 传统量化（V0）

传统量化（V0）是 msModelSlim 的早期量化服务，基于脚本化的方式实现量化功能。它将量化过程分为模型加载、离群值抑制和量化校准与保存三个阶段。

## 核心概念

V0 量化服务主要由以下核心接口组成：
- **AntiOutlier**: 用于离群值抑制。
- **Calibrator**: 用于量化校准。

## 支持框架

- [PyTorch 脚本量化](pytorch/foundation_model_post_training_quantization.md): 针对 PyTorch 框架的大模型脚本量化指南。
- [MindSpore 脚本量化](mindspore/post_training_quantization.md): 针对 MindSpore 框架的脚本量化指南。
- [ONNX 脚本量化](onnx/post_training_quantization.md): 针对 ONNX 模型的脚本量化指南。

## 其他特性

- [模型蒸馏](common/model_distillation.md): 支持模型蒸馏功能。
- [模型剪枝](pytorch/importance_based_pruning_and_tuning.md): 基于重要性评估的剪枝调优。
- [低秩分解](pytorch/model_low_rank_factorization.md): 模型低秩分解特性。

> **注意**：modelslim_v0 协议版本即将废弃，不推荐使用。建议优先使用 [一键量化（V1）](../02-one_click_quantization_v1/usage.md)。
