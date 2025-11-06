

CUDA_VISIBLE_DEVICES=0 python generate_act_scales.py --model-name /disk1/model/Qwen3-30B-A3B/ --dataset-type combined --output-path ./combine_30b_instruct_recite.pt \
--num-samples 192 --seq-len 4096 --dataset-path /disk1/model/bench_res/oldAutoGPTQ/recite_sample_128.jsonl

CUDA_VISIBLE_DEVICES=0 python smooth_model.py --model-name /disk1/model/Qwen3-30B-A3B --smooth-version fp4-combined-mlp --alpha 0.5 --act-scales /disk1/model/smoothquant_fp4/combine_30b_instruct_recite.pt  --output-path ./output_path/ --only-smooth-mlp