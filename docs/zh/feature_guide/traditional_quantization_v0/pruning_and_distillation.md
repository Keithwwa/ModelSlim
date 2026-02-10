# 模型剪枝与蒸馏

## 简介
模型剪枝与蒸馏是进一步优化模型性能和减小模型体积的重要手段。剪枝通过移除冗余参数，蒸馏通过教师模型引导学生模型学习，从而在保持精度的同时提升效率。

---

## 模型剪枝

### 基于重要性评估的剪枝 (PyTorch)
通过评估权重重要性，移除不重要的神经元或通道。
- **支持框架**：PyTorch 2.0.0+
- **使用方法**：
```python
from msmodelslim.pytorch.prune.prune_torch import PruneTorch
# 1. 初始化 PruneTorch
pruner = PruneTorch(model, torch.ones([1, 3, 224, 224]).type(torch.float32))
# 2. 执行剪枝并获取描述信息
desc = pruner.prune(0.8) # 剪枝率0.8
# 3. 后续评估加载剪枝信息
pruner.prune_by_desc(desc)
```

### Transformer 类模型权重剪枝
支持将大型 Transformer 模型的权重裁剪并加载到结构相同的小模型中。
- **支持框架**：MindSpore, PyTorch
- **核心步骤**：
1. **配置剪枝**：定义 `PruneConfig`。
```python
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig
prune_config = PruneConfig()
prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block']).add_blocks_params(pattern="bert.encoder.layer.(\d+).", layer_id_map={0: 0, 1: 2, 2: 4})
```
2. **执行裁剪与加载**：
```python
from msmodelslim.common.prune.transformer_prune.prune_model import prune_model_weight
# bert_model 为初始化的小模型
prune_model_weight(bert_model, prune_config, weight_file_path="large_model.pt")
```

---

## 模型蒸馏

### 简介
用户提供 teacher 模型、student 模型和数据集，通过 API 引导 student 模型学习 teacher 的知识。

### 使用流程
1. **配置蒸馏**：
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
distill_config = KnowledgeDistillConfig()
distill_config.add_output_soft_label({
    "t_output_idx": 1, "s_output_idx": 1,
    "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1, "temperature": 1}]
})
```
2. **获取蒸馏模型**：
```python
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model
distill_model = get_distill_model(teacher_model, student_model, distill_config)
```
3. **训练与提取**：对 `distill_model` 进行训练，完成后调用 `distill_model.get_student_model()` 获取优化后的学生模型。
