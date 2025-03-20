import os

from datasets import load_dataset
import json

# 加载数据集
dataset = load_dataset("/mnt/private_hk/data/RLHF-V-Dataset")


# 定义一个函数来修改每一行
def add_new_column(example):
    info = json.loads(example["text"])
    # print(info)
    example['problem'] = info['question']
    example['solution'] = '<answer> %s </answer>' % info['chosen']
    image = example['image']
    os.makedirs(os.path.dirname(os.path.join('/mnt/private_hk/data/RLHF-V-Dataset-img/', example['image_path'])),
                exist_ok=True)
    if image.mode == 'P' or image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(os.path.join('/mnt/private_hk/data/RLHF-V-Dataset-img/', example['image_path']))
    return example


# 使用 map 函数应用修改
updated_dataset = dataset.map(add_new_column)

# # 保存更新后的数据集（可选）
updated_dataset.save_to_disk("/mnt/private_hk/data/RLHF-V-Dataset-H")
