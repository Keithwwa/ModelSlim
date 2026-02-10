# 传统模型量化

## 简介
传统模型量化工具支持对卷积神经网络（CNN）等传统模型进行量化，通过将浮点权重和激活值转换为低位定点数，提升推理性能。目前支持 PyTorch、ONNX 和 MindSpore 框架。

---

## 训练后量化（PyTorch）

### 简介
训练后量化工具需要用户提供PyTorch训练脚本或者pth文件，工具可自动对模型中的卷积和线性层（torch.nn.Linear和torch.nn.Conv2d）进行识别并量化，最终导出量化后的onnx模型，量化后的模型可以在推理服务器上运行，达到提升推理性能的目的。量化过程中用户需自行提供模型与数据集，调用API接口完成模型的量化调优。

### 功能介绍
#### 自动混合精度量化算法
为了提升量化精度，训练后量化（PyTorch）算法内置了自动混合精度的模块，自动识别并回退量化敏感层为浮点计算，避免量化敏感层对精度造成较大损失。算法核心是：计算每个量化层量化前后输出的MSE，根据MSE的排序来衡量每一个量化层的量化敏感性，自动回退MSE最大的部分敏感层，从而提升量化的精度。

#### 精度保持策略
为了进一步降低量化精度损失，训练后量化（PyTorch）工具内集成了多种精度保持策略，对权重的量化参数和取整方式进行优化。
- Easy Quant权重优化方法：利用输出相似性优化量化参数，减少输入输出张量的量化误差，推荐在data-free模式下使用，通常能够起到较好的改善效果。
- ADMM权重优化方法：使用交替优化的方法，对权重的量化参数进行迭代更新优化，推荐在label-free模式下使用，适当改善量化效果。
- Rounding取整优化：在量化中普通取整不是最优解，使用自适应取整的方式优化权重的取整能提高量化精度，推荐在label-free模式下使用，适当改善量化效果。

### 调用示例
```python
import torchvision
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator
from ascend_utils.common.security import SafeWriteUmask

if __name__ == '__main__':
    MODEL_ARCH = "resnet50"
    SAVE_PATH = "./output"
    INPUTS_NAMES = ["input.1"]

    model = torchvision.models.resnet50(pretrained=True)
    model.eval()

    disable_names = []
    input_shape = [1, 3, 224, 224]
    keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}

    quant_config = QuantConfig(
        disable_names=disable_names,  # 手动回退的量化层名称
        amp_num=0,  # 混合精度量化回退层数
        input_shape=input_shape,  # 模型输入的shape
        keep_acc=keep_acc,  # 精度保持策略
        sigma=25,  # 统计方法：大于0使用sigma统计；0值使用min-max统计。
    )

    calibrator = Calibrator(model, quant_config)
    calibrator.run()  # 执行量化算法
    calibrator.export_quant_onnx("resnet50", "./output", ["input.1"])  # 导出量化onnx模型
```

### 进阶：多模态量化与校准数据获取
多模态量化场景（如 SD3）通常需要特殊的校准数据获取方式。
#### 校准数据获取流程
1. 加载预训练模型。
2. 添加 `Listener` 类用于捕捉模型输入参数。
3. 配置 `calib_prompts`。
4. 遍历提示词执行前向推理。
5. 保存校准数据。

#### 校准数据获取示例
```python
import torch
from diffusers import StableDiffusion3Pipeline

class Listener(torch.nn.Module):
    def __init__(self, module):
        super(Listener, self).__init__()
        self.module = module
        self.inputs = []
    
    def forward(self, *args, **kwargs):
        sample = {}
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                sample[k] = kwargs[k].cpu()
            else:
                sample[k] = kwargs[k]
        self.inputs.append(sample)
        return self.module(*args, **kwargs)

# 加载模型并添加监听器
pipe = StableDiffusion3Pipeline.from_pretrained("/path/to/sd3/", torch_dtype=torch.float16).to("npu")
pipe.transformer = Listener(pipe.transformer)

# 生成校准数据
calib_prompts = ['a photo of a cat holding a sign that says hello world']
for prompt in calib_prompts:
    pipe(prompt=prompt, num_inference_steps=28)

calib_dataset = pipe.transformer.inputs
```

---

## 训练后量化（ONNX）

### 简介
自动对ONNX模型中的卷积（Conv）和矩阵乘法（Gemm）进行识别和量化，支持 Label-Free 和 Data-Free 模式。

### 使用前准备
注意：当前 ONNX 量化功能暂不支持Python 3.12 及以上版本。
```bash
# 详细依赖列表
pip3 install numpy>=1.23.0       # Python 3.8+
pip3 install onnx>=1.16.2
pip3 install onnxruntime>=1.14.1
pip3 install torch==2.1.0        # CPU版本
pip3 install onnx-simplifier>=0.3.10
```

### 功能实现流程
1. **用户准备**：准备原始 ONNX 模型。
2. **配置参数**：使用 `QuantConfig`。动态 Shape 需开启 `is_dynamic_shape=True` 并配置 `input_shape`。
3. **图优化 (可选)**：通过 `graph_optimize_level` 开启内置图优化。
4. **执行量化**：初始化 `OnnxCalibrator` 并调用 `run()`。
5. **导出模型**：调用 `export_quant_onnx`。

### 已验证模型
目前已支持包括但不限于以下模型。

#### 表格1 已验证模型列表（Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件或Atlas 推理系列产品）

| 模式 | 任务 | 模型 | 源码链接参考 |
| :--- | :--- | :--- | :--- |
| Label-Free模式 | 图像分类 | efficientnet_B | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1) |
| | | googleNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/GoogleNet_for_Pytorch) |
| | | ResNet18 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch) |
| | | ResNet34 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNet34) |
| | | ResNet50 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNet50 v1.5 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNet101 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer) |
| | | senet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SENet) |
| | | squeezeNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Squeezenet1_1) |
| | | ShuffleNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Shufflenetv2_for_Pytorch) |
| | 目标检测 | CenterNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CenterNet) |
| | | Retinanet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet) |
| | | YoloV4 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Yolov4_for_Pytorch) |
| | | YoloV5 | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | | YoloV5m | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | | YoloV5s | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| Label-Free模式 | 图像分类 | Densenet121 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Densenet121_Pytorch_Infer) |
| | | efficientnet_B1 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1) |
| | | efficientnet_B2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/EfficientNet-B2_ID1714_for_PyTorch) |
| | | efficientformer | [Link](https://github.com/snap-research/efficientformer) |
| | | googleNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/cv/classification/Googlenet_ID0447_for_PyTorch) |
| | | InceptionV3 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/InceptionV3_for_Pytorch) |
| | | MobileNetV1 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNet-v1) |
| | | MobileNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/MobileNetV2_for_Pytorch) |
| | | MobileNetV3 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNetV3_large_100) |
| | | ResNet18 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch) |
| | | ResNet34 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNet34) |
| | | ResNet50 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNet50 v1.5 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNeXt-50 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/ResNeXt50) |
| | | ResNet101 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer) |
| | | senet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/SENet) |
| | | squeezeNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Squeezenet1_1) |
| | | ShuffleNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Shufflenetv2_for_Pytorch) |
| | | VGG16 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG16) |
| | | VGG19 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG19) |
| | 图像分割 | Deeplabv3 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/DeeplabV3) |
| | 目标检测 | CenterFace | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CenterFace) |
| | | CenterNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/CenterNet) |
| | | fasterrcnn | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50) |
| | | solov2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/SOLOV2) |
| | | SSD | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD) |
| | | SSDResNet34 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD-Resnet34) |
| | | Retinanet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet) |
| | | YoloV3 | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov3_for_PyTorch) |
| | | YoloV4 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Yolov4_for_Pytorch) |
| | | YoloV5 | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | | YoloV5m | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | | YoloV5s | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | 目标动作 | Siamrpn | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/tracking/SiamRPN) |
| | 自动问答 | Bert-Base Squad | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Uncased_for_Pytorch) |
| | 自然语言处理 | Bert-Base Chinese | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch) |
| | 语义分割 | PSPNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/PSPNet) |
| | | UNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/UNet) |
| | 动作识别 | C3D | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/PoseC3D) |
| | 人体姿态估计 | pose_hrnet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/pose_estimation/HRNet) |

#### 表格2 已验证模型列表（Atlas 200/500 A2推理产品）

| 模式 | 任务 | 模型 | 源码链接参考 |
| :--- | :--- | :--- | :--- |
| Label-Free模式 | 图像分类 | googleNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/GoogleNet_for_Pytorch) |
| | | MobileNetV1 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNet-v1) |
| | | MobileNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/MobileNetV2_for_Pytorch) |
| | | MobileNetV3 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNetV3_large_100) |
| | | ResNet18 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch) |
| | | ResNet50 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNet101 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer) |
| | | squeezeNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Squeezenet1_1) |
| | | ShuffleNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Shufflenetv2_for_Pytorch) |
| | 目标检测 | Retinanet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet) |
| | | YoloV3 | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov3_for_PyTorch) |
| | | YoloV4 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Yolov4_for_Pytorch) |
| | | SSD | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD) |
| Label-Free模式 | 图像分类 | efficientnet_B0 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1) |
| | | googleNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/GoogleNet_for_Pytorch) |
| | | MobileNetV1 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNet-v1) |
| | | MobileNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/MobileNetV2_for_Pytorch) |
| | | MobileNetV3 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/MobileNetV3_large_100) |
| | | ResNet18 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch) |
| | | ResNet50 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer) |
| | | ResNet101 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet101_Pytorch_Infer) |
| | | VGG16 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG16) |
| | | VGG19 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/VGG19) |
| | | squeezeNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/Squeezenet1_1) |
| | | ShuffleNetV2 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Shufflenetv2_for_Pytorch) |
| | 目标检测 | fasterrcnn | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50) |
| | | Retinanet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet) |
| | | VGG16-SSD | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/VGG16_SSD_for_PyTorch) |
| | | YoloV3 | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov3_for_PyTorch) |
| | | YoloV4 | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Yolov4_for_Pytorch) |
| | | YoloV5m | [Link](https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch) |
| | | SSD | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/SSD) |
| | 语义分割 | UNet | [Link](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/segmentation/UNet) |

---

## 训练后量化（MindSpore）

### 简介
支持MindSpore框架模型的量化调优。用户需提供权重文件和一小批验证集。

### 详细操作步骤（以 ResNet50 为例）
1. **脚本准备**：新建 `resnet50_quant.py`，加载模型结构并加载预训练权重。
```python
import mindspore as ms
from resnet import resnet50 # 假设模型定义在resnet.py
model = resnet50(class_num=10)
param_dict = ms.load_checkpoint("resnet50.ckpt")
ms.load_param_into_net(model, param_dict)
model.set_train(False)
```
2. **生成配置**：
```python
from msmodelslim.mindspore.quant.ptq_quant.create_config import create_quant_config
config_file = "./quant_config.json"
create_quant_config(config_file, model)
```
3. **模型转换与校准**：
```python
from msmodelslim.mindspore.quant.ptq_quant.quantize_model import quantize_model
input_data = ms.Tensor(np.random.uniform(size=[1, 3, 224, 224]), ms.float32)
model_calibrate = quantize_model(config_file, model, input_data)

# 使用少量数据校准
for data in dataset.create_dict_iterator(num_epochs=1):
    model_calibrate(data['image'])
```
4. **保存模型**：
```python
from msmodelslim.mindspore.quant.ptq_quant.save_model import save_model
save_model("./quantized_model", model_calibrate, input_data, file_format="AIR")
```

---

## 量化感知训练 (QAT)

### 简介
通过重新训练量化模型来减小模型大小并加快推理。当前支持 PyTorch 框架的 CNN 类模型。

### 操作步骤
1. **模型替换**：在优化器初始化前调用 `qsin_qat`。
```python
from msmodelslim.pytorch.quant.qat_tools import qsin_qat, QatConfig
quant_config = QatConfig(grad_scale=0.001)
model = qsin_qat(model, quant_config, logger).to(model.device)
```
2. **训练与导出**：执行原训练流程，保存权重后使用 `save_qsin_qat_model` 导出 ONNX。
