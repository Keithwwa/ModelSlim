# GroupProcessor 分组处理器

## 简介

对多个处理器进行分组管理，支持对同一组内的处理器进行统一的配置管理。

## 使用前准备

安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../install_guide.md)。

## 功能介绍

### 使用场景

- 当需要对不同类型的层应用不同的量化策略时，可以使用group进行分组管理，可以降低资源消耗。
- 例如：对self_attention层使用静态量化，对mlp层使用动态量化。

### YAML配置示例

#### W8A8混合量化配置

```yaml
# 定义 W8A8 静态量化配置模板
default_w8a8: &default_w8a8
  act:                         # 激活值配置
    scope: "per_tensor"        # 静态量化标识：整个张量共用量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: False           # 非对称量化：false
    method: "minmax"           # 量化方法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：true
    method: "minmax"           # 量化方法：minmax

# 定义 W8A8 动态量化配置模板
default_w8a8_dynamic: &default_w8a8_dynamic
  act:                         # 激活值配置
    scope: "per_token"         # 动态量化标识：每个 token 独立量化参数
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：true
    method: "minmax"           # 量化方法：minmax
  weight:                      # 权重量化配置
    scope: "per_channel"       # 权重量化粒度：逐通道量化
    dtype: "int8"              # 数据类型：int8
    symmetric: True            # 对称量化：true
    method: "minmax"           # 量化方法：minmax
    
spec:
  process:                     # 处理器列表
    - type: "group"            # 处理器类型：分组处理器
      configs:                 # 组内处理器配置列表
        - type: "linear_quant" # 任务1：对 Attention 层应用静态量化，以获取最佳推理性能
          qconfig: *default_w8a8
          include: ["*self_attn*"]
        - type: "linear_quant" # 任务2：对 MLP 层应用动态量化，以应对激活离群值并保护精度
          qconfig: *default_w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate"]   # 排除门控层，实现更精细的层级控制
```

### YAML配置字段详解

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 处理器类型标识 | 固定值"group"，用于标识这是一个分组处理器 |
| configs | 组内处理器配置列表 | 包含多个处理器配置，支持不同类型处理器组合 |
