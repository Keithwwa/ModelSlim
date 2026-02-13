# 辅助工具与专项指导

## msmodelslim量化权重格式

msmodelslim llm-ptq工具生成的safetensors量化权重文件包含两个文件，quant_model_weight.safetensors权重文件和quant_model_description.json权重描述文件

msmodelslim量化类型说明：  
[W8A16](#w8a16量化): Linear权重int8量化，激活值不量化  
[W8A8](#w8a8-w8a8s量化): Linear权重int8量化，激活值int8量化  
[W8A8S](#w8a8-w8a8s量化): Linear权重int8稀疏量化，激活值int8量化  

**注意** msmodelslim工具生成的量化权重均为signed场景，即int8数据分布范围为-128到127。开源权重若为unsigned场景，对于int8可以考虑将weight和offset权重减去128

脚本convert_example.py提供了将开源ChatGLM2-6B转换成msmodelslim量化权重的示例，权重获取链接见[开源模型README](https://github.com/zai-org/ChatGLM2-6B?tab=readme-ov-file#%E4%BD%8E%E6%88%90%E6%9C%AC%E9%83%A8%E7%BD%B2) 低成本部署章节，使用前请修改224行和225行的输入输出路径。使用方式`python convert_example.py`

## 量化权重、描述文件格式
### safetensors权重格式
权重保存为safetensors格式，内部格式为python的字典 dict，包含量化权重和量化不修改的浮点权重，字典的key值为权重名称，value为具体权重的数值  
以ChatGLM2-6B为例：'transformer.embedding.word_embeddings.weight'为浮点模型中word_embedding层的权重，名称和权重均未修改，对应描述文件量化类型为'FLOAT'；'transformer.encoder.layers.0.self_attention.dense.weight'为原始模型第0层layer的dense层linear的权重，经过量化修改，数据类型为int8，对应描述文件量化类型为'W8A16'；'transformer.encoder.layers.0.self_attention.dense.weight_scale'为原始模型第0层layer的dense层linear量化后新增的量化参数weight_scale，对应描述文件量化类型为'W8A16'

示例 ChatGLM2-6B W8A16量化权重：
```
{
    'transformer.embedding.word_embeddings.weight': tensor([...]),
    'transformer.encoder.final_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.input_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.post_attention_layernorm.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.self_attention.dense.weight_offset': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight_scale': tensor([...]),
    'transformer.encoder.layers.0.self_attention.query_key_value.weight_offset': tensor([...]),
    ...
    剩下几层以此类推
    ...
    'transformer.output_layer.weight': tensor([...]),
    'transformer.rotary_pos_emb.inv_freq': tensor([...])
}
```
### json描述文件格式
json描述文件内部储存格式为python的字典 dict，字典的key值为权重名称，value为权重对应的量化类型。"model_quant_type"描述整体的量化类型，"kv_cache_type"表示kv_cache是否量化，其余为各个权重的类型，"FLOAT"表示来自浮点权重，"W8A8"表示来自W8A8量化，"W8A16"表示来自W8A16量化，"W8A8S"表示来自稀疏量化  
示例 ChatGLM2-6B W8A16量化权重的描述文件：  
描述文件字典内容排序不影响实际使用
```
{
    "model_quant_type": "W8A16",  
    "kv_cache_type": "C8", # 使用kv cache量化后会生成该行  
    "transformer.embedding.word_embeddings.weight": "FLOAT",  
    "transformer.rotary_pos_emb.inv_freq": "FLOAT",
    "transformer.encoder.layers.0.input_layernorm.weight": "W8A16",  
    "transformer.encoder.layers.0.self_attention.query_key_value.weight": "W8A16",  
    # 使用kv cache量化后生成如下4行  
    "transformer.encoder.layers.0.self_attention.query_key_value.k_proj.kv_cache_scale": "W8A16",  
    "transformer.encoder.layers.0.self_attention.query_key_value.k_proj.kv_cache_offset": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.v_proj.kv_cache_scale": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.v_proj.kv_cache_offset": "W8A16",
    "transformer.encoder.layers.0.self_attention.query_key_value.weight_scale": "W8A16",  
    "transformer.encoder.layers.0.self_attention.query_key_value.weight_offset": "W8A16",
    "transformer.encoder.layers.0.post_attention_layernorm.weight": "FLOAT", 
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight": "W8A16",  
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_scale": "W8A16",  
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_offset": "W8A16",  
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight": "W8A16",  
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_scale": "W8A16",  
    "transformer.encoder.layers.0.mlp.dense_4h_to_h.weight_offset": "W8A16", 
    ...
    剩下几层以此类推
    ...
    "transformer.encoder.final_layernorm.weight": "FLOAT",
    "transformer.output_layer.weight": "FLOAT"
}
```

## W8A16量化  
量化工具对于每个量化的Linear生成3个参数，参数名称为：`weight`、`weight_scale`、`weight_offset`，在safetensors权重文件中，完整的权重名称为Linear层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.self_attention.query_key_value.weight_scale"`，`"transformer.encoder.layers.0.self_attention.query_key_value"`为Linear层的名称，`"weight_scale"`为参数的名称

### 权重说明
`weight`为量化后的int8的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size  
`weight_scale`为量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float32，在per_channel场景下，shape为[n]，在per_group场景下，shape为[n, k / group_size]  
`weight_offset`为量化的偏移系数，数据类型为torch.Tensor，dtype和shape和weight_scale一致。对称量化场景下需要构造全0的weight_offset  

### 反量化计算公式 
per_channel 场景：  
```python
deq_weight = (weight - weight_offset) * weight_scale  
```
per_group 场景： 
```python
weight = weight.reshape((-1, group_size))  
weight_offset = weight_offset.reshape((n * k / group_size, 1))  
weight_scale = weight_scale.reshape((n * k / group_size, 1))  
deq_weight = ((weight - weight_offset) * weight_scale).reshape((n, k))
```
**注意** npu量化算子计算时实际的逻辑为(weight + weight_offset) * weight_scale，昇腾推理框架在加载量化权重时进行了取负操作

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_process，请根据开源权重的反量化公式和msmodelslim工具的反量化公式进行相应修改

## W8A8, W8A8S量化
msmodelslim量化工具对于每个量化的Linear生成5个参数，参数名称为：weight, input_scale, input_offset, deq_scale, quant_bias  
在safetensors权重文件中，完整的权重名称为Linear层的名称+参数名称，与W8A16类似

### 权重说明
`weight`为量化后的int8的权重，数据类型为torch.Tensor，dtype为torch.int8，shape和原始浮点的shape一致，记为n, k = weight.shape，k为hidden_size   
`input_scale`为激活值量化的缩放系数，数据类型为torch.Tensor，dtype为torch.float16或torch.bfloat16，shape为[1]  
`input_offset`为激活值量化的偏移系数，数据类型为torch.Tensor，dtype和shape和input_scale一致。  
`deq_scale`为反量化缩放系数，数据类型为torch.Tensor，dtype为torch.int64或torch.float32，shape为[n]。注意为了亲和昇腾量化算子，开源量化若基于fp16，则deq_scale的数据在传给量化算子前需要进行数据类型转换，可以参考示例代码120行进行处理，若开源量化权重为bf16，则不需要数据类型转换
`quant_bias`为反量化的偏移系数，数据类型为torch.Tensor，dtype为torch.int32，shape为[n]  

### 量化、反量化计算公式 
```python
input_quant = input_fp / input_scale + input_offset  
output_quant = input_quant * weight + quant_bias  
output_dequant = output_quant * deq_scale  
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.weight_activation_process，请根据开源权重的计算公式和msmodelslim工具的计算公式进行相应修改

## smooth quant
msmodelslim量化工具使用smooth quant后，对于每个norm层，生成2个参数，module.weight和module.bias。完整的权重名称为norm层的名称+参数名称，例如ChatGLM2-6B量化权重中，`"transformer.encoder.layers.0.input_layernorm.module.weight"`，`"transformer.encoder.layers.0.input_layernorm"`为norm层的名称，`"module.weight"`为量化参数名称

msmodelslim量化工具集成的smooth quant算法针对norm层后的Linear层进行smooth平滑操作，而不是所有Linear层。采取这种量化方案的优势在于可以将原本乘在激活值上的scale等价转移到原始浮点模型norm层的权重`norm.weight`上，从而避免额外引入算子带来的性能开销

`module.weight`为scale后的norm.weight，数据类型、dtype、shape和norm.weight一致  
`module.bias`为引入module.weight后带来的偏移系数，数据类型、dtype、shape和norm.weight一致  
 
为了适配几种特殊的回退情况，msmodelslim生成的smooth quant权重中还包含原始浮点权重norm层的权重，norm.weight。如果开源量化权重不涉及回退场景，设置为None即可

代码实现可以参考demo样例MSModelSlimWeightProcessor.anti_outlier_process，请根据开源权重的设计方案和msmodelslim工具的设计方案进行相应修改

## KV Cache量化
msmodelslim工具提供的KV Cache量化采用int8量化。对于每个attention层，生成4个参数，`k_proj.kv_cache_scale`, `k_proj.kv_offset`, `v_proj.kv_cache_scale`, `v_proj.kv_cache_offset`。对于qkv合并或kv合并的场景，完整的四个参数的名称为合并的Linear名称+参数名；对于qkv分离场景，k_proj的scale、offset完整的参数名称为k对应Linear名称+参数名称，v_proj的scale、offset完整的参数名称为v对应Linear名称+参数名称。例如`"transformer.encoder.layers.0.query_key_value.k_proj.kv_cache_scale"`，`"transformer.encoder.layers.0.query_key_value"`为qkv合并的Linear层名称，`"k_proj.kv_cache_scale"`为参数名称

### 权重说明
`kv_cache_scale`为kvcache量化scale的缩放系数，数据类型为torch.Tensor，dtype为torch.float32或torch.float16，shape为kv channel的size，如果是qkv分开场景，则为k或v层linear的n维（见上文**w8a16量化** 章节 **权重说明** weight的shape说明）
`kv_scale_offset`为kvcache量化scale的偏移系数，数据类型、dtype、shape和scale一致

计算公式：  
量化 
```python
cache_int = cache_fp / cache_scale + cache_offset
```
反量化 
```python
cache_deq = (cache_int - cache_offset) * cache_scale
```

代码实现可以参考demo样例MSModelSlimWeightProcessor.kv_cache_process，请根据开源权重的计算公式和msmodelslim工具的计算公式进行相应修改

## MindSpeed适配器

### 简介
原有的llm_ptq模块主要支持基于transformers框架下的大模型量化压缩功能，本模块提供了针对ModelLink模型的量化适配器，可以直接量化MindSpeed-LLM模型

### 使用前准备
- 仅支持在以下产品中使用。
    - Atlas 训练系列产品。
    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件。

- 安装 msModelSlim 工具，详情请参见[《msModelSlim工具安装指南》](../../getting_started/install_guide.md)。
- 大模型量化工具须执行命令安装如下依赖。
  如下命令如果使用非root用户安装，需要在安装命令后加上--user，例如：pip3 install onnx --user。
```
pip3 install numpy==1.25.2
pip3 install transformers        #需大于等于4.29.1版本，LLaMA模型需指定安装4.29.1版本
pip3 install accelerate==0.21.0  #若需要使用NPU多卡并行方式对模型进行量化，需大于等于0.28.0版本
pip3 install tqdm==4.66.1
```
- 安装MindSpeed-LLM库,[安装指导](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/pytorch/install_guide.md)

### 功能介绍

### 功能约束
当前模型适配器仅验证过支持w8a8的量化，以及异常值抑制模块的m3和m5算法，仅支持NPU执行量化，不支持CPU量化

### 量化步骤（以llama2-7b legacy为例）
1.获取开源权重，转化为MindSpeed-LLM支持的模型,可以使用MindSpeed-LLM的权重[转化脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/convert_ckpt.py)，[此处有转化脚本使用教程](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/pytorch/solutions/checkpoint/checkpoint_convert.md)
```
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_from_hf/llama-2-7b-hf/ \
    --save-dir ./model_weights/llama-2-legacy/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model \
    --model-type-hf llama2
```
2.设计量化函数, 以w8a8为例：
```
def quant(model):
    # 准备校准数据，请根据实际情况修改，W8A16 Label-Free模式下请忽略此步骤
    dataset_calib = [["中国的首都在哪里？"],
                ["请做一首诗歌："],
                ["我想要学习python，该怎么学习？"]]

    from msmodelslim.pytorch.mindspeed_adapter import ModelAdapter, CalibratorAdapter, Linear    # 导入量化配置接口
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
    #转化模型，适配mindspeed-llm
    model = ModelAdapter(model)
    # 配置回退层,此处以回退mlp.dense_4h_to_h为例
    disable_names = []
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    for name, mod in model.named_modules():
        if isinstance(mod, Linear) and "mlp.dense_4h_to_h" in name:
            disable_names.append(name)
    # 量化配置，请根据实际情况修改
    # 使用QuantConfig接口，配置量化参数，并返回量化配置实例
    quant_config = QuantConfig(
        w_bit=8,  
        a_bit=8,         
        disable_names=disable_names, 
        dev_type='npu',   # 在cpu进行量化时，需配置参数dev_type='cpu'，并取消参数dev_id=model.device.index的配置
        mm_tensor=False
    )  
    #使用CalibratorAdapter接口，输入加载的原模型、量化配置和校准数据，定义校准
    calibrator = CalibratorAdapter(model, quant_config, calib_data=dataset_calib, disable_level='L0')  
    calibrator.run()     #使用run()执行量化
    calibrator.save('./quant_weight', save_type=[ 'numpy', 'safe_tensor'])      #使用save()保存模型量化参数，请根据实际情况修改路径
    print('Save quant weight success!')
```

3.将上述量化函数插入推理脚本，以mindspeed-llm的自带推理精度测试脚本[evaluation.py](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/evaluation.py)为例，将quant函数插入main函数。请注意`trust_remote_code`为`True`时可能执行浮点模型权重中代码文件，请确保浮点模型来源安全可靠。
```
 ...
def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load, 
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True, local_files_only=True)
    quant(model) #插入之前写好的量化函数
    rank = dist.get_rank()
    if 'mmlu' in args.task:
        a = time.time()
        mmlu(args, LLMChat(args, model, tokenizer))
        if rank == 0:
            logger.info(f'MMLU Running Time:, {time.time() - a}')
 ...

```

4. 修改MindSpeed-LLM推理evaluate_llama2_7B_ptd.sh执行上述量化，以legacy启动为例，修改脚本模型路径
```
 ...
 TOKENIZER_PATH=./model_from_hf/llama-2-7b-hf/  #huggingface开源模型路径
 CHECKPOINT=./model_weights/llama-2-legacy/  #前面转化生成的权重路径
 ...
 python -m torch.distributed.launch $DISTRIBUTED_ARGS evaluation.py   \
 ...
 --tensor-model-parallel-size 1  \
 --pipeline-model-parallel-size 1  \
 --padded-vocab-size 32000 \ #根据模型配置相应的模型参数
 ...
```
5.执行推理脚本，完成量化，进行伪量化精度验证
```
bash examples/legacy/llama2/evaluate_llama2_7B_ptd.sh
```
