CUDA_VISIBLE_DEVICES=5 python autogpt8k.py --num_calibrations 128 --dataset recite --device_num 1 --batch_size 1 --quantized_model_dir  /disk1/model/AutoGPTQ/Qwen3-30Ba3-Instruct-recite-128-right


CUDA_VISIBLE_DEVICES=5 python autogpt8k.py --num_calibrations 192 --dataset combined --device_num 1 --batch_size 1 --quantized_model_dir  /disk1/model/AutoGPTQ/Qwen3-30Ba3-Instruct-combined-192-right

CUDA_VISIBLE_DEVICES=5 python autogpt8k.py --num_calibrations 192 --dataset combined --device_num 1 --batch_size 1 --only_quantize_mlp --quantized_model_dir  /disk1/model/AutoGPTQ/Qwen3-30Ba3-Instruct-combined-192-mlp-right


CUDA_VISIBLE_DEVICES=0 python /disk1/model/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py \
--pyt_ckpt_path   /disk1/model/newgptq/AutoGPTQ/Qwen3-30Ba3-Instruct-recite-128-1031 --qformat nvfp4 \
 --export_fmt hf --export_path Qwen3-30B-A3B-fp4-recite-128-1031 --trust_remote_code --kv_cache_qformat none


CUDA_VISIBLE_DEVICES=5 python /disk1/model/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py --pyt_ckpt_path /disk1/model/AutoGPTQ/Qwen3-30Ba3-Instruct-combined-192-right --qformat nvfp4 --export_fmt hf --export_path Qwen3-30B-A3B-fp4-combined-192-right --trust_remote_code --kv_cache_qformat none


CUDA_VISIBLE_DEVICES=5 python /disk1/model/TensorRT-Model-Optimizer/examples/llm_ptq/hf_ptq.py --pyt_ckpt_path /disk1/model/AutoGPTQ/Qwen3-30Ba3-Instruct-recite-128-right --qformat nvfp4 --export_fmt hf --export_path Qwen3-30B-A3B-fp4-recite-128-right --trust_remote_code --kv_cache_qformat none
