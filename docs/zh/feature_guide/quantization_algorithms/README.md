---
toc_depth: 3
---
# 量化算法总览

msModelSlim 支持多种先进的量化算法，涵盖了从离群值抑制到低比特优化的各个环节。下表总结了目前支持的核心算法及其主要特性。

<div class="custom-table">

<table>
  <thead>
    <tr>
      <th style="width: 15%;">算法名称</th>
      <th style="width: 15%;">类型</th>
      <th style="width: 30%;">核心思想</th>
      <th style="width: 25%;">适用场景</th>
      <th style="width: 15%;">详细说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>AutoRound</strong></td>
      <td>权重量化优化</td>
      <td>基于 SignSGD 优化舍入偏移，降低重构误差</td>
      <td>4bit 等超低比特量化</td>
      <td><a href="autoround.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>FA3 Quant</strong></td>
      <td>激活量化</td>
      <td>针对 Attention 激活的 per-head INT8 量化</td>
      <td>长序列、MLA 架构模型</td>
      <td><a href="fa3_quant.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>QuaRot</strong></td>
      <td>离群值抑制</td>
      <td>应用正交旋转矩阵平滑激活值分布</td>
      <td>抑制激活离群值，提升精度</td>
      <td><a href="quarot.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>SmoothQuant</strong></td>
      <td>离群值抑制</td>
      <td>协同缩放激活与权重，平滑离群值</td>
      <td>抑制激活离群值</td>
      <td><a href="smooth_quant.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Iterative Smooth</strong></td>
      <td>离群值抑制</td>
      <td>迭代式平滑缩放，更精细的分布调整</td>
      <td>复杂分布下的精度优化</td>
      <td><a href="iterative_smooth.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Flex Smooth Quant</strong></td>
      <td>离群值抑制</td>
      <td>二阶段网格搜索自动寻找最优 alpha/beta</td>
      <td>灵活适配不同架构</td>
      <td><a href="flex_smooth_quant.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Flex AWQ SSZ</strong></td>
      <td>离群值抑制</td>
      <td>结合 AWQ 与 SSZ，使用真实量化器评估误差</td>
      <td>自动搜索最优平滑参数</td>
      <td><a href="flex_awq_ssz.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>KV Smooth</strong></td>
      <td>KV Cache 优化</td>
      <td>针对 KV Cache 的平滑抑制算法</td>
      <td>降低 KV Cache 显存占用</td>
      <td><a href="kv_smooth.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>KVCache Quant</strong></td>
      <td>KV Cache 量化</td>
      <td>针对 KV Cache 的量化方案</td>
      <td>提升长序列推理效率</td>
      <td><a href="kvcache_quant.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>GPTQ</strong></td>
      <td>权重量化优化</td>
      <td>通过逐列优化和误差补偿最小化量化误差</td>
      <td>高精度权重量化需求</td>
      <td><a href="gptq.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Linear Quant</strong></td>
      <td>基础量化</td>
      <td>对线性层进行权重量化和激活量化</td>
      <td>基础量化场景</td>
      <td><a href="linear_quant.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>PDMIX</strong></td>
      <td>混合阶段量化</td>
      <td>Prefilling 使用动态量化，Decoding 使用静态量化</td>
      <td>大模型推理加速，平衡精度与性能</td>
      <td><a href="pdmix.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Standing High</strong></td>
      <td>精度保持</td>
      <td>针对特定层的精度保护策略</td>
      <td>解决关键层量化损失</td>
      <td><a href="standing_high.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Float Sparse</strong></td>
      <td>稀疏化</td>
      <td>基于 ADMM 算法实现模型浮点稀疏化</td>
      <td>高压缩率需求</td>
      <td><a href="float_sparse.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>Histogram</strong></td>
      <td>激活量化</td>
      <td>分析直方图分布，搜索最优截断区间</td>
      <td>过滤离群值，提高精度</td>
      <td><a href="histogram_activation_quantization.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>MinMax</strong></td>
      <td>基础量化</td>
      <td>统计最大最小值确定量化范围</td>
      <td>基础量化场景，计算开销低</td>
      <td><a href="minmax.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>SSZ</strong></td>
      <td>权重量化</td>
      <td>迭代搜索最优缩放因子和偏移量</td>
      <td>权重分布不均的精度优化</td>
      <td><a href="ssz.md">查看详情</a></td>
    </tr>
    <tr>
      <td><strong>LAOS</strong></td>
      <td>低比特量化</td>
      <td>针对 W4A4 等极低比特场景的优化</td>
      <td>极致压缩需求</td>
      <td><a href="laos.md">查看详情</a></td>
    </tr>
  </tbody>
</table>

</div>

## 算法选择建议

- **初学者**：建议优先使用 [一键量化 (V1)](../quick_quantization_v1/usage.md)，它会自动集成合适的算法组合。
- **追求极致精度**：可以尝试组合使用 **QuaRot** + **AutoRound**。
- **长序列推理**：推荐开启 **FA3 Quant** 和 **KVCache Quant**。

<style>
.custom-table table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #dfe2e5;
}
.custom-table th, .custom-table td {
    border: 1px solid #dfe2e5;
    padding: 8px 12px;
}
.custom-table tr:nth-child(2n) {
    background-color: #f6f8fa;
}
</style>
