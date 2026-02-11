# 传统模型与大模型压缩特性指南 (v0)

本目录包含了 msModelSlim 工具针对传统模型和大模型的压缩优化文档（相关特性已停止演进）。为了方便查阅，我们将相关文档进行了整合分类。

## 文档总览

### 1. [传统模型量化](./traditional_quantization.md)
涵盖了针对传统 CNN 类模型的量化技术：
- **训练后量化 (PTQ)**：支持 PyTorch, ONNX, MindSpore 框架。
- **量化感知训练 (QAT)**：支持 PyTorch 框架，通过训练提升量化精度。

### 2. [大模型压缩](./foundation_model_compression.md)
涵盖了针对大语言模型 (LLM) 和多模态大模型的全栈压缩方案：
- **训练后量化 (PTQ)**：W8A8, W8A16, W4A16 等多种位宽支持。
- **稀疏量化与权重压缩**：进一步降低模型体积。
- **FA3 量化**：针对 Flash Attention 3 的硬件加速量化。
- **低显存量化**：显存受限场景下的模型加载与量化方案。
- **长序列压缩**：针对 Alibi 和 RoPE 编码的长文本推理优化。

### 3. [模型剪枝与蒸馏](./pruning_and_distillation.md)
涵盖了结构化剪枝与知识蒸馏技术：
- **模型剪枝**：基于重要性评估的剪枝及 Transformer 权重剪枝。
- **模型蒸馏**：通过教师模型引导小模型学习训练。

### 4. [多模态生成模型推理优化](./inference_optimization_for_multimodal_generative_model.md)
针对大规模多模态生成模型（如 OpenSora）的专项优化方案：
- **DiT 缓存优化**：通过缓存中间计算结果加速生成。
- **自适应采样优化**：自适应调整采样步数提升效率。

### 5. [辅助工具与专项指导](./compression_utils.md)
- **低秩分解**：通过矩阵分解降低计算开销。
- **混合校准数据集**：灵活构建高质量校准数据。
- **[伪量化精度测试工具](./fake_quantization_accuracy_testing_tool.md)**：在 NPU 上验证量化模型精度。
- **[稀疏加速训练](./sparse_acceleration_training.md)**：稀疏化训练的实现指导。
- **[MindSpeed 适配器](./mindspeed_adapter.md)**：与 MindSpeed 框架的集成说明。

### 6. [参考与示例]
- **[量化与稀疏量化导入代码示例](./quantization_and_sparse_quantization_scenario_import_code_examples.md)**：各场景下的 API 调用模板。
- **[开源权重格式转换](./convert_opensource_weights_to_msmodelslim_format.md)**：开源模型权重转换指南。

---

## 快速链接
- [安装指南](../../getting_started/install_guide.md)
- [已验证模型列表](../../model_support/foundation_model_support_matrix.md)
- [API 接口文档](../../python_api_v0/common_apis.md)
