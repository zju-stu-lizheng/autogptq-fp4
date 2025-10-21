# 伪量化功能使用说明

## 概述

伪量化功能允许您先使用GPTQ进行nvfp4量化，然后将量化后的权重和per-block scales转换回float16精度保存。这样可以获得量化带来的压缩效果，同时保持float16的精度用于推理。

## 主要特性

- ✅ **NVFP4量化**: 使用4位浮点数量化，block_size=16
- ✅ **Per-block Scale**: 每个16元素的块都有独立的缩放因子
- ✅ **Float16精度**: 最终保存的模型使用float16精度
- ✅ **压缩效果**: 通过量化获得模型压缩效果
- ✅ **兼容性**: 完全集成到AutoGPTQ框架中

## 使用方法

### 1. 基本使用

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.quantization.config import QUANT_METHOD

# 配置nvfp4量化
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4位量化
    group_size=16,  # 组大小设为16，与nvfp4的block_size一致
    desc_act=False,  # 设为False可以显著提升推理速度
    quant_method=QUANT_METHOD.NVFP4,  # 使用nvfp4量化方法
)

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, 
    quantize_config
)

# 执行量化
model.quantize(examples, use_triton=False, cache_examples_on_gpu=False)

# 保存伪量化模型（会自动重定向到伪量化保存）
model.save_quantized(
    save_dir,
    use_safetensors=True
)
```

### 2. 修改您的autogpt8k.py

您的`autogpt8k.py`文件已经修改为支持伪量化：

```python
# 配置nvfp4量化
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4位量化
    group_size=16,  # 设置为16以匹配nvfp4的block_size
    desc_act=False,
    quant_method=QUANT_METHOD.NVFP4,  # 使用nvfp4量化方法
)

# 执行量化（不使用triton）
model.quantize(examples, use_triton=False, cache_examples_on_gpu=False)

# 保存伪量化模型（会自动重定向到伪量化保存）
model.save_quantized(quantized_model_dir, use_safetensors=True)
```

## 保存的文件结构

伪量化模型会保存以下文件：

```
model_dir/
├── pseudo_quantized_model-4bit-16g.safetensors    # 主模型文件（float16精度）
├── pseudo_quantized_model-4bit-16g_scales.safetensors  # NVFP4 scales (per_block_scale, scale_2, g_idx)
├── config.json                                     # 模型配置
└── quantize_config.json                           # 量化配置
```

## 技术细节

### NVFP4量化器

- **Block Size**: 每个块包含16个元素
- **数值范围**: [-6, 6]（4位浮点数，e2m1格式）
- **Scale计算**: 
  - `scale_2` = `amax` / (6.0 * 448.0) （per-tensor缩放因子）
  - `per_block_scale` = `per_block_amax` / (6.0 * scale_2) （per-block缩放因子）
- **e2m1格式**: 使用特定的边界值 [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5] 进行量化
- **无Zero Point**: NVFP4不需要zero point

### 伪量化过程

1. **量化阶段**: 使用GPTQ算法进行nvfp4量化，权重已经是float16精度
2. **跳过打包**: 跳过pack_model步骤，保持原始nn.Linear层结构
3. **获取阶段**: 直接从GPTQ对象获取已量化的权重和per-block scales
4. **保存阶段**: 保存float16权重和per-block scales

### 关键代码修改

1. **GPTQ类**: 添加了`pseudo_quantize_to_fp16`方法，直接获取已量化的权重和scales
2. **BaseGPTQForCausalLM类**: 
   - 添加了`save_pseudo_quantized`方法
   - 修改了`quantize`方法，对nvfp4跳过pack_model步骤
   - 修改了`save_quantized`方法，自动重定向到伪量化保存
3. **BaseQuantizeConfig**: 添加了`QUANT_METHOD.NVFP4`支持
4. **量化过程**: 修改以支持nvfp4量化方法，跳过打包步骤
5. **NVFP4Quantizer**: 量化后的权重已经是float16精度，无需额外转换

## 优势

1. **压缩效果**: 通过nvfp4量化获得模型压缩
2. **精度保持**: 最终模型使用float16精度
3. **推理效率**: 可以使用标准的float16推理
4. **兼容性**: 与现有AutoGPTQ框架完全兼容

## 注意事项

1. **内存使用**: 量化过程需要额外的内存来存储GPTQ对象
2. **量化精度**: nvfp4的精度可能比标准INT4量化略低
3. **硬件支持**: 确保目标硬件支持float16运算
4. **模型大小**: 伪量化模型的大小与原始float16模型相同

## 示例运行

运行修改后的`autogpt8k.py`：

```bash
python autogpt8k.py --model_name Qwen3-30Ba3 --num_calibrations 8192
```

这将创建一个伪量化的模型，使用nvfp4量化方法，最终保存为float16精度的模型。
