# PyTorch 大模型脚本量化指南 (V0)

针对 Llama、ChatGLM 等大语言模型（LLM），msModelSlim V0 提供了一套基于脚本的训练后量化（PTQ）方案。本指南将详细介绍如何通过代码实现离群值抑制、量化校准及权重保存。

## 1. 核心操作流程

大模型脚本量化的标准步骤如下：

### 步骤 1：准备环境与模型
确保已安装 `msmodelslim` 及其依赖（transformers, accelerate, torch_npu）。加载您的浮点模型：
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('./model_path', local_files_only=True)
model = AutoModel.from_pretrained('./model_path', torch_dtype=torch.float16).npu()
```

### 步骤 2：离群值抑制 (AntiOutlier)
在量化前，使用 `AntiOutlier` 平滑激活值分布，这是保持精度的关键。
```python
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

# 配置抑制算法（常用 m3）
anti_config = AntiOutlierConfig(anti_method='m3', dev_type='npu', dev_id=model.device.index)
# 传入模型和一小批校准数据
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()
```

### 步骤 3：量化校准 (Calibrator)
配置量化参数并运行校准任务。
```python
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# 配置 W8A8 量化
quant_config = QuantConfig(
    w_bit=8, a_bit=8, 
    dev_type='npu', 
    dev_id=model.device.index, 
    act_method=3, # 3 代表自动混合量化
    disable_names=['transformer.encoder.layers.0.self_attention.query_key_value'] # 排除敏感层
)

# 初始化校准器并运行
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run() # 执行后模型变为伪量化模型
```

### 步骤 4：保存量化权重
```python
# 保存为 safetensors 格式（推荐用于 MindIE 推理）
calibrator.save('./quant_weight', save_type=['safe_tensor'])
```

---

## 2. 高级特性操作

### 2.1 Flash Attention 3 (FA3) 量化步骤
1. **修改 modeling 文件**：在 `Attention` 类的 `__init__` 中添加 `self.fa_quantizer = FAQuantizer(self.config, logger)`。
2. **插入量化逻辑**：在 `forward` 函数计算出 `query_states` 后，调用 `query_states = self.fa_quantizer.quant(query_states, qkv="q")`。
3. **配置接口**：在 `QuantConfig` 后调用 `.fa_quant(fa_amp=5)`。

### 2.2 低显存模式加载
如果显存不足以加载完整模型，使用 `accelerate` 的 `device_map`：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={0: "20GiB", "cpu": "100GiB"} # 限制 NPU 显存使用
)
```

## 3. 常见参数说明

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `anti_method` | str | 离群值抑制方法，可选 'm1', 'm2', 'm3'。LLM 推荐使用 'm3'。 |
| `act_method` | int | 激活量化方法。1: min-max, 2: histogram, 3: auto。 |
| `disable_names` | list | 手动指定不进行量化的层名称列表。 |
| `save_type` | list | 权重保存格式，支持 'numpy' 和 'safe_tensor'。 |
