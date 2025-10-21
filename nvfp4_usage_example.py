#!/usr/bin/env python3
"""
NVFP4量化使用示例
展示如何使用修改后的GPTQ进行block_size=16的nvfp4量化
"""

# 使用示例代码（需要安装torch和transformers）

def example_usage():
    """
    使用NVFP4量化的示例代码
    """
    
    # 1. 导入必要的模块
    import torch
    import torch.nn as nn
    from auto_gptq.quantization.gptq import GPTQ, NVFP4Quantizer
    
    # 2. 创建一个线性层进行量化
    layer = nn.Linear(512, 256)
    
    # 3. 创建GPTQ实例，指定使用nvfp4量化方法
    gptq = GPTQ(layer, quant_method="nvfp4")
    
    # 4. 准备校准数据（用于构建Hessian矩阵）
    calibration_data = []
    for _ in range(10):  # 使用10个样本进行校准
        inp = torch.randn(1, 512)
        out = layer(inp)
        gptq.add_batch(inp, out)
    
    # 5. 执行NVFP4量化
    scale, zero, g_idx = gptq.fasterquant(
        blocksize=16,           # 使用16作为块大小
        group_size=16,          # 组大小
        quant_method="nvfp4",   # 指定使用nvfp4量化
        nvfp4_block_size=16,    # NVFP4的块大小
        actorder=True,          # 使用激活顺序
        static_groups=False     # 不使用静态组
    )
    
    # 6. 检查结果
    print(f"量化完成！")
    print(f'layer weight shape: {layer.weight.data.shape}')
    print(f"Scale shape: {scale.shape}")  # 应该是 (256, num_blocks)
    print(f"Zero: {zero}")                # 对于nvfp4应该是None
    print(f"Group indices shape: {g_idx.shape}")
    
    # 7. 保存量化后的模型权重和scale
    quantized_weights = layer.weight.data
    per_block_scales = scale
    
    return quantized_weights, per_block_scales, g_idx

def nvfp4_quantizer_direct_usage():
    """
    直接使用NVFP4Quantizer的示例
    """
    
    import torch
    from auto_gptq.quantization.gptq import NVFP4Quantizer
    
    # 1. 创建测试数据
    batch_size, seq_len = 2, 64
    test_data = torch.randn(batch_size, seq_len) * 5
    
    # 2. 创建NVFP4量化器
    quantizer = NVFP4Quantizer()
    quantizer.configure(bits=4)
    
    # 3. 计算量化参数（per-block scale）
    quantizer.find_params(test_data, weight=False, block_size=16)
    
    # 4. 执行量化
    quantized_data = quantizer.quantize(test_data, block_size=16)
    
    # 5. 检查结果
    print(f"原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
    print(f"量化后数据范围: [{quantized_data.min():.3f}, {quantized_data.max():.3f}]")
    print(f"每个块的scale: {quantizer.scale}")


    ### test_data 和 quantized_data 是否一致
    print(f"原始数据和反量化回float16的值是否一致: {torch.allclose(test_data, quantized_data)}")
    ### 计算误差
    print(test_data.shape, '\n', quantized_data.shape)
    print(test_data[:,:8], '\n', quantized_data[:,:8])
    error = torch.abs(test_data - quantized_data) / torch.abs(test_data)
    # print(f"误差范围: [{error.min():.3f}, {error.max():.3f}]")
    print(f"平均误差: {error.mean()*100:.3f}%")
    return quantized_data, quantizer.scale

def key_features():
    """
    关键特性说明
    """
    print("""
    NVFP4量化实现的关键特性：
    
    1. Block Size = 16: 每个16个元素组成一个块，共享一个scale
    2. Per-block Scale: 每个块都有独立的缩放因子
    3. NVFP4格式: 4位浮点数格式，数值范围[-7, 7]
    4. 无Zero Point: NVFP4不需要zero point，只需要scale
    5. 兼容GPTQ: 完全集成到现有的GPTQ框架中
    
    使用方法：
    - 在GPTQ初始化时设置 quant_method="nvfp4"
    - 在fasterquant中设置 nvfp4_block_size=16
    - 返回的zero参数为None，只使用scale参数
    """)

if __name__ == "__main__":
    print("NVFP4量化使用示例")
    key_features()
    
    # 注意：实际运行需要安装torch和transformers
    print("\n要运行实际示例，请确保已安装必要的依赖：")
    print("pip install torch transformers")
    print("\n然后取消注释下面的代码行：")
    print("# example_usage()")
    print("# nvfp4_quantizer_direct_usage()")
    example_usage()
    ### 验证一下 原始值 和 反量化回float16的值 是否一致
    nvfp4_quantizer_direct_usage()
