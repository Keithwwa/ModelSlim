# PyTorch 大模型脚本量化（V0）

本指南详细介绍了如何使用 msModelSlim V0 接口对大语言模型进行脚本化量化。V0 接口基于 `Calibrator` 和 `AntiOutlier` 等核心类，支持 W8A8、W8A16、W4A16 等多种量化方案。

> **说明**：本指南适用于 V0 脚本化量化。对于更简便的使用体验，建议参考 [一键量化完整指南](../quick_quantization_v1/usage.md)。

## 目录

- [使用前准备](#使用前准备)
- [量化流程总览](#量化流程总览)
- [核心功能说明](#核心功能说明)
  - [基础量化示例](#基础量化示例)
  - [Flash Attention 3 (FA3) 量化](#flash-attention-3-fa3-量化)
  - [低显存量化](#低显存量化)
  - [混合校准数据集](#混合校准数据集)
- [精度保持与调优策略](#精度保持与调优策略)
- [量化后权重说明](#量化后权重文件)

---

## 使用前准备

### 1. 硬件支持
- Atlas 推理系列产品（如 Atlas 300I Duo 推理卡）
- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas 800I A2 推理产品

### 2. 环境配置
安装 msModelSlim 及相关依赖：
```bash
pip3 install numpy==1.25.2
pip3 install transformers>=4.29.1
pip3 install accelerate>=0.21.0
pip3 install tqdm==4.66.1
```

---

## 量化流程总览

大模型脚本量化的核心步骤如下：
1. **准备模型与校准数据**：准备原始浮点模型和具有代表性的校准数据集。
2. **离群值抑制（可选）**：使用 `AntiOutlier` 抑制激活值中的离群点，提升精度。
3. **量化配置**：使用 `QuantConfig` 定义量化参数（比特数、算法等）。
4. **执行量化**：使用 `Calibrator` 构建校准对象并运行 `run()`。
5. **保存模型**：调用 `save()` 接口导出量化后的权重和参数。

---

## 核心功能说明

### 基础量化示例

以下是使用 W8A8 per-channel 方案量化 ChatGLM2-6B 的核心代码：

```python
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# 1. 加载模型与分词器
model_path = './chatglm2'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).npu()

# 2. 准备校准数据
calib_list = ["中国的首都在哪里？", "请做一首诗歌："]
dataset_calib = []
for data in calib_list:
    inputs = tokenizer([data], return_tensors='pt').to(model.device)
    dataset_calib.append([inputs.data['input_ids'], inputs.data['attention_mask']])

# 3. 量化配置
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,
    dev_type='npu',
    dev_id=model.device.index,
    act_method=3, # 混合 min-max 和 histogram
)

# 4. 执行量化
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

# 5. 保存结果
calibrator.save('./quant_weight', save_type=['safe_tensor'])
```

### Flash Attention 3 (FA3) 量化

FA3 量化针对 KV-Cache 增强了硬件利用率，提升了推理效率。

**关键步骤**：
1. **修改 modeling 文件**：在 Attention 模块中注入 `FAQuantizer`。
   ```python
   from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import FAQuantizer
   # 在 __init__ 中
   self.fa_quantizer = FAQuantizer(self.config, logger)
   # 在 forward 中
   query_states = self.fa_quantizer.quant(query_states, qkv="q")
   key_states = self.fa_quantizer.quant(key_states, qkv="k")
   value_states = self.fa_quantizer.quant(value_states, qkv="v")
   ```
2. **配置 QuantConfig**：调用 `.fa_quant()` 方法。
   ```python
   quant_config = QuantConfig(w_bit=8, a_bit=8, ...).fa_quant(fa_amp=5)
   ```

### 低显存量化

当显存不足以完整加载千亿级参数模型时，可启用低显存模式。该模式将模型存放于 Host 内存中，仅在计算时使用 NPU。

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={0: "25GiB", "cpu": "500GiB"}
)
```

---

## 精度保持与调优策略

如果量化后出现精度大幅下降或对话乱码，请尝试以下策略：

1.  **适当增加回退层**：某些层对量化极度敏感（如 `mlp.down_proj`），可使用 `disable_names` 手动排除。
2.  **调整 act_method**：`act_method=3` 通常在大模型场景下表现最佳。
3.  **使用离群值抑制**：将 `do_smooth` 设置为 `True`，配合 `AntiOutlier` 使用。
4.  **优化校准集**：确保校准集覆盖了模型实际使用的语言和任务类型。

---

## 量化后权重文件

### npy 格式
设置 `save_type=['numpy']` 时生成。
- `quant_weight.npy`: 量化后的权重。
- `input_scale.npy` / `input_offset.npy`: 激活值量化参数。
- `deq_scale.npy`: 反量化缩放因子。

### safetensors 格式（推荐）
设置 `save_type=['safe_tensor']` 时生成，包含 `.safetensors` 权重文件和 `.json` 描述文件。这是 MindIE 推理引擎的标准格式。
