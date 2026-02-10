# 模型低秩分解与压缩工具

## 模型低秩分解

### 简介
通过将大矩阵分解为若干个低秩矩阵的乘积，降低存储空间和计算量。适用于 CV 和 NLP 任务。

### 支持框架
MindSpore 和 PyTorch。

### 使用流程
1. **配置分解**：
```python
from msmodelslim.pytorch import low_rank_decompose
decomposer = low_rank_decompose.Decompose(model).from_ratio(0.5) 
```
2. **执行分解**：
```python
model = decomposer.decompose_network() # 替换原模型
```
3. **微调训练**：多卡训练时需先在单卡下保存权重，再在多卡下设置 `do_decompose_weight=False` 加载权重。

---

## 压缩辅助工具

### 混合校准数据集 (CalibrationData)
支持混合多种标准数据集或自定义数据集。

#### Config 文件示例 (mix_config.json)
```json
{
  "configurations": [
    {
      "dataset_name": "boolq",
      "dataset_path": "./boolq/dev.jsonl"
    },
    {
      "dataset_name": "ceval_5_shot",
      "dataset_path": "./ceval_5_shot/"
    },
    {
      "dataset_name": "gsm8k",
      "dataset_path": "./gsm8k/GSM8K.jsonl"
    },
    {
      "dataset_name": "mmlu",
      "dataset_path": "./mmlu/"
    }
  ]
}
```

#### 自定义处理器示例
```python
from msmodelslim.pytorch.llm_ptq.mix_calibration.dataset_processor_base import DatasetProcessorBase

class CustomizedProcessor(DatasetProcessorBase):
    def process_data(self, indexs):
        # 获取一组样本，返回 [{"prompt": p, "ans": a}, ...]
        pass
    
    def verify_positive_prompt(self, prompts, labels):
        # 验证正样本
        pass

# 使用示例
calib_select = CalibrationData(config_path="mix_config.json", tokenizer=tokenizer, model=model)
calib_select.add_custormized_dataset_processor("my_dataset", CustomizedProcessor(path))
calib_select.set_sample_size({"boolq": 4, "my_dataset": 3})
mixed_dataset = calib_select.process()
```

### 稀疏工具 (Sparse Tool)
针对 PyTorch 线性层进行稀疏化。
- **method 可选值**：`magnitude` (默认), `hessian`, `par`, `par_v2`。
- **参数说明**：
    - `sparse_ratio`: 0~1 之间的稀疏率。
    - `progressive`: 是否渐进式稀疏。
    - `uniform`: 是否均匀稀疏。

```python
from msmodelslim.pytorch.sparse.sparse_tools import SparseConfig, Compressor
sparse_config = SparseConfig(method="magnitude", sparse_ratio=0.5)
prune_compressor = Compressor(model, sparse_config)
prune_compressor.compress(dataset=test_dataset)
```

### 导入代码示例
针对大模型量化、稀疏量化等不同场景，提供了一系列导入代码模板，详见 `quantization_and_sparse_quantization_scenario_import_code_examples.md`。
