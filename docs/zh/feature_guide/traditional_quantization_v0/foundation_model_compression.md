# 大模型压缩

## 简介
大模型压缩工具针对大语言模型（LLM）和多模态大模型，提供稀疏、量化、权重压缩及长序列压缩等全栈优化方案，旨在降低显存占用并提升推理速度。

### 特殊模型限制说明
- **MOE 模型**：支持 W8A8 per-token、W8A16 per-channel/per-group 场景，不支持 lowbit 稀疏量化。
- **多模态模型**：仅支持 W8A16 场景，不支持 W8A8 和 lowbit 稀疏量化。
- **硬件支持**：
    - Atlas 推理系列产品（Atlas 300I Duo 推理卡）。
    - Atlas 训练系列产品。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

---

## 训练后量化 (PTQ)

### 简介
将大模型高位浮点权重/激活转为低位定点数（如 W8A8, W8A16, W4A16），在无需训练的前提下完成压缩。

### 使用前准备 {#使用前准备}
```bash
# 核心依赖
pip3 install numpy==1.25.2
pip3 install transformers>=4.29.1
pip3 install accelerate==0.21.0
pip3 install tqdm==4.66.1
```
若需多卡并行量化，需关闭虚拟内存：
```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

### 核心步骤
1. **离群值抑制 (Anti-Outlier)**：使用 `AntiOutlierConfig` (如 `anti_method="m3"`) 生成配置，调用 `AntiOutlier.process()`。
2. **量化配置**：使用 `QuantConfig` 配置位宽。
3. **校准执行**：调用 `Calibrator.run()`。
4. **模型保存**：调用 `save(save_type=['numpy', 'safe_tensor'])`。

### 使用示例 (以 ChatGLM2-6B 为例)
以下是一个标准的 W8A8 per-channel 量化脚本：

```python
import torch 
import torch_npu 
from transformers import AutoTokenizer, AutoModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

# 1. 加载模型
model_path = './chatglm2'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).npu()

# 2. 准备校准数据 (建议 20-40 条)
calib_list = ["中国的首都在哪里？", "请写一段关于秋天的诗。", "如何学习Python？"]
def get_calib_dataset(tokenizer, calib_list):
    dataset = []
    for data in calib_list:
        inputs = tokenizer([data], return_tensors='pt').to(model.device)
        dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])
    return dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)

# 3. 配置量化参数
quant_config = QuantConfig(
    a_bit=8, w_bit=8, dev_type='npu', act_method=3, pr=0.5, mm_tensor=False
)

# 4. 执行量化与保存
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()
calibrator.save('./quant_weight', save_type=['numpy', 'safe_tensor'])
```

### 常见量化场景配置
- **W8A16 / W4A16**: 设置 `a_bit=16`, `w_bit=8/4`。推荐 `w_method='GPTQ'` 或 `'HQQ'`。
- **KV Cache 量化**: 在 `QuantConfig` 中配置 `use_kvcache_quant=True`。
- **低显存量化 (Low VRAM)**: 当显存不足时，利用 `accelerate` 将模型映射到内存：
  ```python
  model = AutoModelForCausalLM.from_pretrained(..., device_map="auto", max_memory={0: "25GiB", "cpu": "500GiB"})
  ```

### 精度调优与定位
如果精度不达标，可使用以下方法：
- **精度定位**：使用 `FakeQuantizeCalibrator` 接口构建伪量化模型进行前向推理测试。支持 W8A8 (per_channel)、W8A16 per-channel (MinMax, GPTQ, HQQ) 场景。

- **调优诊断逻辑**：
    - **现象一：对话乱码或胡言乱语（精度掉点严重）**
        1. 确认浮点模型推理是否正常。
        2. **增加回退层**：优先回退模型靠前/靠后的 decoder layer，以及各层的 `mlp.down_proj` 层。
        3. **混合量化**：将敏感层量化为更高精度（如 w8a16 或 w8a8_dynamic）。
    - **现象二：对话正常但指标部分掉点**
        1. **调整参数**：修改 `act_method`（推荐 LLM 场景设为 3）。
        2. **调整校准数据**：增加数据量（一般 20-40 条），并确保语种与应用场景一致。
        3. **异常值抑制**：开启 `do_smooth=True`。

---

## 稀疏量化与权重压缩

### 简介
通过稀疏算法将不必要参数置零，并结合权重压缩进一步减小体积。权重压缩仅支持在 Atlas 推理系列产品上使用。

### 安全风险提示
权重压缩工具在加载输入权重文件时存在**反序列化攻击安全风险**。工具已通过将保存目录权限设为 750、权重文件设为 400、描述文件设为 600 来消减风险。用户在加载前需交互确认文件安全。

### 流程
1. **稀疏量化**：使用 `QuantConfig` 开启稀疏（如 `pr=2.0`, `co_sparse=True`），生成 `quant_weight.npy`。
2. **权重压缩**：
    - **编译**：进入 `msmodelslim/pytorch/weight_compression/compress_graph/` 执行 `bash build.sh`。
    - **执行**：使用 `CompressConfig` 和 `Compressor`。
    - **导出**：调用 `compressor.export()` 导出 `weight`, `index`, `info` 文件。

---

## FA3 量化 (Flash Attention 3)

### 简介
在 KV-Cache 基础上增强硬件利用率，支持 Llama3.1, Qwen2.5 等模型。仅 Atlas 800I A2 推理产品支持。

### 关键步骤
1. **修改 modeling 文件**：
    - 导入 `FAQuantizer`。
    - 在 Attention 的 `__init__` 中初始化 `self.fa_quantizer`。
    - 在 `forward` 的 `past_key_value.update` 后插入量化代码。
2. **配置 QuantConfig**：调用 `.fa_quant(fa_amp=5)` 开启自动精度回退。

---

## 低显存量化 (Low VRAM)

### 简介
当显存不足以加载完整模型时，通过 `accelerate` 库将模型模块映射到内存。
```python
model = AutoModelForCausalLM.from_pretrained(..., device_map="auto", max_memory={0: "25GiB", "cpu": "500GiB"})
```

---

## 长序列压缩 (RazorAttention)

### 简介
针对 Alibi 或 RoPE 编码模型，识别并压缩对位置信息不敏感的注意力头。
- **全量加速**：压缩后的 KV Cache 可直接用于模型推理。
- **增量加速**：支持只更新和压缩新 token 对应的部分。

### 使用方法
使用 `RACompressor` 或 `RARopeCompressor` 导出压缩窗口 `.pt` 文件。

---

## 附录：量化权重说明

### 文件格式
- **npy 格式**：保存为字典，Key 为权重名，Value 为数值。
- **safetensors 格式**：包含 `quant_model_weight.safetensors` 和描述文件 `json`。

### 命名规则
- **W8A16**: 每个 Linear 生成 `weight`, `weight_scale`, `weight_offset`。
- **W8A8**: 每个 Linear 生成 `weight`, `input_scale`, `input_offset`, `deq_scale`, `quant_bias`。
- **KV Cache**: 生成 `kv_cache_scale`, `kv_cache_offset` 等。
