from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.quantization.config import QUANT_METHOD
from transformers import AutoTokenizer
import logging
import torch
from datasets import load_dataset
from typing import Union, List
from utils import get_calibration_data_gptq
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='量化Qwen3模型')
parser.add_argument('--num_calibrations', type=int, default=8192, 
                    help='用于校准的样本数量，默认为8192')
parser.add_argument('--max_length', type=int, default=4096, 
                    help='用于校准的样本最大长度，默认为4096')
parser.add_argument('--device_num', type=int, default=1, 
                    help='GPU数量，默认为1')
parser.add_argument('--model_name', type=str, default='Qwen3-30Ba3')
parser.add_argument('--batch_size', type=int, default=1, 
                    help='量化批量大小，默认为64')
# desc_act
parser.add_argument('--desc_act', action='store_true', default=False,
                    help='是否使用desc_act')
parser.add_argument('--quantized_model_dir', type=str, default=None,
                    help='量化模型保存目录')
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


if args.model_name == "Qwen3-30Ba3":
    pretrained_model_dir = f"/disk1/model/Qwen3-30B-A3B"
else:
    pretrained_model_dir = f"/disk1/model/Qwen3-0.6B"
examples = get_calibration_data_gptq(pretrained_model_dir=pretrained_model_dir,
                                    num_calibrations=args.num_calibrations,
                                    max_length=args.max_length,
                                    use_shuffle=True)
if args.quantized_model_dir is None:
    quantized_model_dir = f"{args.model_name}-Instruct-nvfp4-pseudo-{args.num_calibrations}-new"
else:
    quantized_model_dir = args.quantized_model_dir

print(pretrained_model_dir, quantized_model_dir)

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=16,  # 设置为16以匹配nvfp4的block_size
    desc_act=args.desc_act,  # set to False can significantly speed up inference but the perplexity may slightly bad
    quant_method=QUANT_METHOD.NVFP4,  # 使用nvfp4量化方法
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
if args.model_name == "Qwen3-30Ba3" or args.model_name == "Qwen3-235B":
    max_memory = {i: "100GB" for i in range(args.device_num)}
    max_memory['cpu'] = '800GB'
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, max_memory=max_memory)
else:
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, max_memory={i: "50GB" for i in range(1)})

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples, use_triton=False, cache_examples_on_gpu=False, batch_size=min(args.batch_size,len(examples)))  # nvfp4不使用triton

del examples
## 清除显存占用
torch.cuda.empty_cache()

# save quantized model using safetensors (会自动重定向到伪量化保存)
model.save_quantized(quantized_model_dir, use_safetensors=True)
## save tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
tokenizer.save_pretrained(quantized_model_dir)