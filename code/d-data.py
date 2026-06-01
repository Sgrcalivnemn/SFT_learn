import os
# 关键修正：必须在 import datasets 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from datasets import load_dataset

target_dir = "/root/autodl-tmp/LLaMA-Factory/data"
os.makedirs(target_dir, exist_ok=True)

print("正在从国内镜像下载 ChnSentiCorp 数据集...")
cache_path = os.path.join(target_dir, ".hf_cache")
dataset = load_dataset("lansinuote/ChnSentiCorp", cache_dir=cache_path)

print(f"【查看】原始数据集已成功缓存至: {os.path.abspath(cache_path)}")
print(f"【分割信息】可用分割: {dataset.keys()}")  # 应包含 train, validation, test

def convert_split_to_sft(split_data, split_name):
    """将指定分割的数据转换为 SFT 格式"""
    sft_list = []
    for item in split_data:
        label_text = "正向" if item['label'] == 1 else "负向"
        sft_list.append({
            "instruction": "你是一个文本情感分析专家。请阅读用户的评论，判断其情感倾向是【正向】还是【负向】。只需回答这两个词之一，不要说任何多余的话。",
            "input": item['text'],
            "output": label_text
        })
    return sft_list

# 分别处理训练集、验证集、测试集
splits = {
    "train": "chn_senti_sft_train.json",
    "validation": "chn_senti_sft_val.json",
    "test": "chn_senti_sft_test.json"
}

for split_name, filename in splits.items():
    if split_name not in dataset:
        print(f"警告：数据集不包含 '{split_name}' 分割，跳过。")
        continue
    
    print(f"正在处理 {split_name} 分割...")
    sft_data = convert_split_to_sft(dataset[split_name], split_name)
    output_path = os.path.join(target_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存 {len(sft_data)} 条数据至: {output_path}")

print("\n" + "="*50)
print("【成功】所有分割的数据集均已生成！")
print(f"【位置】保存目录: {target_dir}")
print("="*50)
print("训练集样例预览：\n", json.dumps(sft_data[0], ensure_ascii=False, indent=2))