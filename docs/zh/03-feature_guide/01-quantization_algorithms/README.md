# 量化算法

msModelSlim 提供了丰富的量化算法，支持不同精度的量化需求。以下是支持的算法列表及其详细说明。

## 离群值抑制算法 (Anti-Outlier)

这些算法主要用于在量化前对激活值中的离群值进行抑制，以减少量化误差。

- [Flex Smooth Quant](flex_smooth_quant.md): 进阶方案。支持自动参数搜索，当 Iterative Smooth 效果不佳时尝试。
- [Flex AWQ SSZ](flex_awq_ssz.md): 低比特必备。专为 INT4/W4A8 等场景设计，使用真实量化器评估误差。
- [Iterative Smooth](iterative_smooth.md): 首选方案。运行快，精度较高，适用于绝大多数 W8A8 场景。
- [Smooth Quant](smooth_quant.md): 经典的平滑量化算法。
- [KV Smooth](kv_smooth.md): 针对 KVCache 的平滑算法。
- [QuaRot](quarot.md): 旋转量化。通过数学旋转消除离群点。

## 量化算法 (Quantization)

这些算法定义了如何将浮点权重和激活值转换为低比特表示。

- [MinMax](minmax.md): 简单高效。INT8 场景优先推荐，在保证精度的前提下速度最快。
- [SSZ](ssz.md): 迭代搜索。INT4/W4A8 等低比特场景优先推荐，通过最小化量化误差提升精度。
- [AutoRound](autoround.md): 高精度上限。适用于对精度极度敏感的场景，量化效果最接近浮点。
- [PDMIX](pdmix.md): 激活值阶段间混合量化算法，平衡推理精度与性能。
- [KVCache Quant](kvcache_quant.md): 针对 KVCache 的量化算法。
- [FA3 Quant](fa3_quant.md): FA3 量化算法说明。
- [Float Sparse](float_sparse.md): 大模型浮点稀疏算法。

## 综合方案

- [LAOS (W4A4方案)](laos.md): 针对 W4A4 的综合量化方案。
