# PyTorch 模型压缩与优化（剪枝、低秩分解、蒸馏）

除了量化，msModelSlim 提供了剪枝（Pruning）、低秩分解（Low-Rank Factorization）和模型蒸馏（Distillation）等多种模型压缩技术，用于进一步减少模型规模、提升推理效率。

## 目录

- [基于重要性评估的剪枝](#基于重要性评估的剪枝)
- [模型低秩分解](#模型低秩分解)
- [模型蒸馏](#模型蒸馏)

---

## 基于重要性评估的剪枝

剪枝通过移除模型中不重要的参数（如权重较小的神经元）来减少模型大小。msModelSlim 提供了简单的 API，仅需少量代码即可集成到现有的训练或微调脚本中。

### 操作步骤
1. **导入接口**：
   ```python
   from msmodelslim.pytorch.prune.prune_torch import PruneTorch
   ```
2. **执行剪枝**：在模型加载权重后，调用 `prune` 接口。参数 `0.8` 表示保留 80% 的权重。
   ```python
   # model 为原始模型实例
   # input_data 用于追踪模型结构的样例输入
   pruner = PruneTorch(model, torch.ones([1, 3, 224, 224]))
   desc = pruner.prune(0.8) 
   ```
3. **微调（Fine-tuning）**：剪枝后建议进行少量 epoch 的训练（通常使用较小的学习率），以恢复精度。
4. **加载剪枝模型**：在评估或推理时，使用保存的 `desc` 信息重建剪枝结构：
   ```python
   PruneTorch(model, dummy_input).prune_by_desc(desc)
   ```

---

## 模型低秩分解

低秩分解通过将大的权重矩阵分解为两个或多个小矩阵的乘积，从而减少计算量。

### 操作步骤
1. **导入接口**：
   ```python
   from msmodelslim.pytorch.low_rank.low_rank_decomposition import LowRankDecomposition
   ```
2. **执行分解**：
   ```python
   decomposition = LowRankDecomposition(model)
   # 对模型中的线性层或卷积层进行分解
   decomposition.decompose()
   ```

---

## 模型蒸馏

模型蒸馏（Knowledge Distillation）通过“老师”模型引导“学生”模型学习，使得更轻量级的学生模型能够获得接近老师模型的性能。

### 核心接口
- `DistillConfig`: 配置蒸馏策略（如中间层匹配、Loss 权重等）。
- `DistillCalibrator`: 执行蒸馏校准过程。

> **提示**：蒸馏通常与其他压缩手段（如剪枝或量化）结合使用，以获得更优的精度保持效果。
