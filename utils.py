import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer

def extract_messages(file_path, max_lines=None):
    """
    提取 JSONL 文件中的 messages 属性。
    
    :param file_path: JSONL 文件路径
    :param max_lines: 最大读取行数，默认为 None（表示加载全部）
    :return: 包含所有 messages 的列表
    """
    # 初始化存储 messages 的列表
    all_messages = []

    # 打开 JSONL 文件并逐行读取
    with open(file_path, "r", encoding="utf-8") as file:
        # 如果 max_lines 未指定，则读取全部行
        lines_to_read = file if max_lines is None else [next(file) for _ in range(max_lines)]
        
        # 使用 tqdm 包装文件迭代器以显示进度条
        for line in tqdm(lines_to_read, desc="Processing lines", unit="line"):
            # 解析每一行的 JSON 数据
            data = json.loads(line.strip())
            # 提取 messages 属性并添加到列表中
            if "messages" in data:
                all_messages.append(data["messages"])

    return all_messages

def get_calib_dataset(
    datas,
    tokenizer=None,
    n_samples=8192,
    max_seq_len=4096,
    split="train",
    text_column="text",
):
    samples = []
    n_run = 0
    for line in datas:
        line = line.strip()
        # line_encoded = tokenizer.encode(line, truncation=True, max_length=max_seq_len)
        # if len(line_encoded) > max_seq_len:
        #     continue
        new_encoded = tokenizer(line, truncation=True, max_length=max_seq_len)
        samples.append(new_encoded)
        n_run += 1
        if n_run == n_samples:
            break

    return samples

def get_calibration_data_gptq(pretrained_model_dir, num_calibrations=8192, max_length=4096, use_shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    data_path = "/disk1/model/AutoGPTQ/30b_calibration.json"
    with open(data_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    data = []
    for msg in dataset:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        # text = msg[0]["content"]
        data.append(text.strip())
    examples = get_calib_dataset(data, tokenizer, n_samples=num_calibrations, max_seq_len=max_length)
    # examples = [tokenizer(text, max_length=4096, truncation=True) for text in data]

    print('use ', len(examples), f' examples from {data_path} for calibration')
    return examples
