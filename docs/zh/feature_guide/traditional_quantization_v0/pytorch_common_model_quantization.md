# PyTorch 常规模型与多模态模型脚本量化（V0）

本指南介绍如何对常规卷积模型（如 ResNet）以及多模态生成模型（如 Stable Diffusion 3）进行训练后量化（PTQ）。这些模型通常需要导出为 ONNX 或 Safetensors 格式以进行端侧或服务器端推理。

## 目录

- [常规卷积模型量化](#常规卷积模型量化)
  - [混合精度量化算法](#自动混合精度量化算法)
  - [精度保持策略](#精度保持策略)
  - [代码示例](#代码示例-resnet50)
- [多模态生成模型量化](#多模态量化场景)
  - [SD3 量化示例](#代码示例-sd3)
  - [校准数据获取](#校准数据获取方式)

---

## 常规卷积模型量化

常规模型量化工具可自动识别并量化模型中的卷积层 (`torch.nn.Conv2d`) 和线性层 (`torch.nn.Linear`)。

### 自动混合精度量化算法
内置自动混合精度模块，通过计算每层量化前后的 MSE（均方误差）来衡量敏感度。MSE 最大的层将被自动回退为浮点计算，以平衡精度与性能。

### 精度保持策略
- **Easy Quant**: 优化量化参数以减少张量误差，推荐在 data-free 模式下使用。
- **ADMM**: 迭代更新权重参数，提升 label-free 场景下的效果。
- **Rounding 取整优化**: 使用自适应取整代替普通四舍五入。

### 代码示例 (ResNet50)

```python
import torchvision
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

if __name__ == '__main__':
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    # 配置量化参数
    quant_config = QuantConfig(
        input_shape=[1, 3, 224, 224], 
        amp_num=5,  # 自动回退 5 个敏感层
        sigma=25,   # 使用 sigma 统计方法
        keep_acc={'admm': [False, 1000], 'easy_quant': [True, 1000]}
    )

    calibrator = Calibrator(model, quant_config)
    calibrator.run()

    # 导出可部署的 ONNX 模型
    calibrator.export_quant_onnx("resnet50", "./output", ["input.1"])
```

---

## 多模态量化场景

目前已支持 SD3 和 OpenSora 1.2 等模型。建议在 Atlas 800I A2 等系列硬件上使用。

### 代码示例 (SD3)

```python
import torch
from diffusers import StableDiffusion3Pipeline
from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig

# 加载模型
pipe = StableDiffusion3Pipeline.from_pretrained(
    "path/to/sd3", torch_dtype=torch.float16
).to("npu")
model = pipe.transformer

# 加载捕获的校准数据
calib_dataset = torch.load("sd3_calib_data.pth", map_location="npu")

quant_config = QuantConfig(
    w_bit=8, a_bit=8, 
    act_quant=True, device="npu"
)

calibrator = Calibrator(model, quant_config, calib_dataset)
calibrator.run()

# 导出 Safetensors
calibrator.export_quant_safetensor("./output/")
```

### 校准数据获取方式
对于生成类模型，通常需要捕获推理过程中的中间张量作为校准数据。可以通过构造一个 `Listener` 装饰器来拦截并保存 `forward` 过程中的输入参数。
