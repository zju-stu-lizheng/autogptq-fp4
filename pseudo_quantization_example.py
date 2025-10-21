#!/usr/bin/env python3
"""
伪量化使用示例
展示如何使用nvfp4量化并保存为float16精度的伪量化模型
"""

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.quantization.config import QUANT_METHOD
from transformers import AutoTokenizer
import logging
import torch

# 设置日志
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

def pseudo_quantization_example():
    """
    伪量化示例：使用nvfp4量化，然后保存为float16精度的模型
    """
    
    # 1. 设置模型路径
    pretrained_model_dir = "facebook/opt-125m"  # 使用小模型进行演示
    quantized_model_dir = "opt-125m-pseudo-quantized-nvfp4"
    
    print(f"开始伪量化过程...")
    print(f"原始模型: {pretrained_model_dir}")
    print(f"保存目录: {quantized_model_dir}")
    
    # 2. 准备校准数据
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "AutoGPTQ is an easy-to-use model quantization library with user-friendly APIs, based on GPTQ algorithm.",
            return_tensors="pt"
        )
    ]
    
    # 3. 配置nvfp4量化
    quantize_config = BaseQuantizeConfig(
        bits=4,  # 4位量化
        group_size=16,  # 组大小设为16，与nvfp4的block_size一致
        desc_act=False,  # 设为False可以显著提升推理速度
        quant_method=QUANT_METHOD.NVFP4,  # 使用nvfp4量化方法
    )
    
    print(f"量化配置: {quantize_config}")
    
    # 4. 加载模型
    print("加载模型...")
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_dir, 
        quantize_config,
        trust_remote_code=True
    )
    
    # 5. 执行量化
    print("执行nvfp4量化...")
    model.quantize(
        examples, 
        use_triton=False,  # 对于nvfp4，不使用triton
        cache_examples_on_gpu=False
    )
    
    # 6. 保存伪量化模型
    print("保存伪量化模型...")
    model.save_quantized(
        quantized_model_dir,
        use_safetensors=True
    )
    
    print("伪量化完成！")
    print(f"模型已保存到: {quantized_model_dir}")
    print("保存的文件包括:")
    print("- 主模型文件: pseudo_quantized_model-4bit-16g.safetensors")
    print("- Scales文件: pseudo_quantized_model-4bit-16g_scales.safetensors")
    print("- 配置文件: config.json, quantize_config.json")
    
    return quantized_model_dir

def load_and_test_pseudo_quantized_model(model_dir):
    """
    加载并测试伪量化模型
    """
    print(f"\n加载伪量化模型: {model_dir}")
    
    # 加载模型
    model = AutoGPTQForCausalLM.from_quantized(
        model_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        use_triton=False
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 测试推理
    test_text = "The future of AI is"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {test_text}")
    print(f"输出: {generated_text}")
    
    return model

def compare_model_sizes(original_dir, pseudo_quantized_dir):
    """
    比较原始模型和伪量化模型的大小
    """
    import os
    
    def get_dir_size(directory):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    original_size = get_dir_size(original_dir)
    pseudo_size = get_dir_size(pseudo_quantized_dir)
    
    print(f"\n模型大小比较:")
    print(f"原始模型大小: {original_size / (1024**3):.2f} GB")
    print(f"伪量化模型大小: {pseudo_size / (1024**3):.2f} GB")
    print(f"压缩比: {original_size / pseudo_size:.2f}x")
    
    return original_size, pseudo_size

def main():
    """
    主函数
    """
    print("=" * 60)
    print("AutoGPTQ 伪量化示例")
    print("=" * 60)
    
    try:
        # 执行伪量化
        quantized_dir = pseudo_quantization_example()
        
        # 测试伪量化模型
        model = load_and_test_pseudo_quantized_model(quantized_dir)
        
        # 比较模型大小
        # compare_model_sizes("facebook/opt-125m", quantized_dir)
        
        print("\n✅ 伪量化示例完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
