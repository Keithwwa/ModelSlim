# 大模型训练后量化 (LLM PTQ)

大模型量化工具将高位浮点数转为低位的定点数，例如16bit降低到8bit，直接减少模型权重的体积，生成量化参数和权重文件。在无需训练成本的前提下，完成大模型的训练后压缩并最大程度保障其精度。

## 1. 使用前准备

### 硬件与软件环境
- 仅支持在以下产品中使用：
    - Atlas 推理系列产品（Atlas 300I Duo 推理卡）。
    - Atlas 训练系列产品。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../../getting_started/install_guide.md)。
- 安装必要依赖：
    ```bash
    pip3 install numpy==1.25.2
    pip3 install transformers        # 需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
    pip3 install accelerate==0.21.0  # 若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
    pip3 install tqdm==4.66.1
    ```

### 环境配置
（可选）如果需要在大模型量化工具中使用NPU多卡并行的方式对模型进行量化，需关闭NPU设备中的虚拟内存，并手动配置量化将会执行的设备序列环境。
```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False # 关闭NPU的虚拟内存
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 #配置量化将会执行的设备序列环境
```

## 2. 功能介绍与流程

大模型压缩技术主要针对常规大语言模型进行量化压缩。

### 功能实现流程
![量化接口调用流程](../figures/[pytorch]quantization_api_calling.png)

关键步骤说明如下：
1. **用户准备**：准备原始模型和校准数据。
2. **离群值抑制（可选）**：对LLM模型进行离群值抑制。
    - 使用 `AntiOutlierConfig` 生成配置，调用 `AntiOutlier` 接口生成抑制器，并执行 `process()`。
3. **生成配置**：使用 `QuantConfig` 生成量化配置。
4. **构建校准对象**：调用 `Calibrator` 接口。
5. **执行量化**：调用 `run()` 方法。
6. **保存结果**：调用 `save()` 接口保存权重和参数。

## 3. 量化步骤示例 (以ChatGLM2-6B为例)

### 3.1 准备环境
1. 准备模型权重、配置文件和校准数据。
2. 安装特定依赖（如 protobuf, sentencepiece 等）。

### 3.2 编写量化脚本
以下是一个标准的 W8A8 per_channel 量化脚本示例：

```python
# 导入相关依赖
import torch 
import torch_npu   # 若需要在cpu上进行量化，可忽略此步骤
from transformers import AutoTokenizer, AutoModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig    # 导入量化配置接口

# 加载模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='./chatglm2', local_files_only=True)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path='./chatglm2', local_files_only=True
  ).npu()

# 准备校准数据
calib_list = ["中国的首都在哪里？", "请做一首诗歌：", "我想要学习python"]
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to(model.device)   
        calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])     
    return calib_dataset
dataset_calib = get_calib_dataset(tokenizer, calib_list)

# 配置量化参数
quant_config = QuantConfig(
    a_bit=8, 
    w_bit=8,       
    dev_type='npu',
    act_method=3,
    pr=0.5, 
    mm_tensor=False
  )  

# 执行校准与保存
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
calibrator.run()
calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])
print('Save quant weight success!')
```

### 3.3 更多量化场景示例
其他场景（如 W8A16, KV Cache 量化, Smooth Quant 等）的代码实现请参考 [量化代码样例](#4-常见量化场景代码样例)。

## 4. 常见量化场景代码样例

### 4.1 W8A16 / W4A16 per-channel 量化
在 `QuantConfig` 中设置 `a_bit=16`, `w_bit=8` 或 `4`。推荐使用 `w_method='MinMax'`, `'HQQ'` 或 `'GPTQ'`。

### 4.2 KV Cache 量化
在 `QuantConfig` 中配置 `use_kvcache_quant=True`。
```python
quant_config = QuantConfig(
    ...,
    use_kvcache_quant=True
).kv_quant(kv_sym=True)
```

### 4.3 Smooth Quant
通过设置 `do_smooth=True` 并配置 `AntiOutlier` 来抑制异常值。

### 4.4 低显存量化模式
当显存不足以完整加载千亿级大模型时，可以启用低显存模式。该模式将模型大部分模块存放于内存中，仅计算时使用NPU。
在使用 `from_pretrained` 加载模型时，通过调整 `device_map` 和 `max_memory` 参数控制。
```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map="auto",
    max_memory={
        0: "25GiB", 
        "cpu": "500GiB"
    }
)
```

## 5. 混合校准数据集准备

### 简介
通过 `CalibrationData` 类可以混合指定的数据集（如 boolq, ceval, mmlu, gsm8k），也支持用户自定义数据集。

### 调用示例
```python
from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import CalibrationData

# 配置采样数量
sample_size = {"boolq": 4, "mmlu": 2}
calib_select = CalibrationData(config_path=CONFIG_PATH, save_path=SAVE_PATH, tokenizer=tokenizer, model=model)
calib_select.set_sample_size(sample_size)
mixed_dataset = calib_select.process()
```

## 6. 量化后权重说明

### 权重文件格式
msmodelslim 生成的量化权重包含两种格式：
1. **npy格式**：保存为字典的 `.npy` 文件，key 为权重名称，value 为数值。
2. **safetensors格式**：包含 `quant_model_weight.safetensors` 和 `quant_model_description.json`。

### 量化权重名称规则
- **W8A16**: 每个 Linear 生成 `weight`, `weight_scale`, `weight_offset`。
- **W8A8**: 每个 Linear 生成 `weight`, `input_scale`, `input_offset`, `deq_scale`, `quant_bias`。
- **KV Cache**: 生成 `kv_cache_scale`, `kv_cache_offset` 等。

## 7. 精度保持与调优策略

如果量化后精度下降严重，可尝试以下手段：
1. **增加回退层**：使用 `disable_level` 自动回退或手动在 `disable_names` 中指定敏感层（如 `down_proj`）。
2. **调整量化方法**：尝试不同的 `act_method` (1: min-max, 2: histogram, 3: 自动混合)。
3. **启用异常值抑制**：设置 `do_smooth=True`。
4. **混合量化**：将敏感层量化为更高精度的数据类型（如 w8a16 或 w8a8_dynamic）。
