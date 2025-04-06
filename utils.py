import os
import json
from glob import glob

def load_json_files(repo_root):
    """
    加载指定 repo 路径下所有 json 类信息文件，过滤掉没有 classDesc 的类。
    """
    json_files = glob(os.path.join(repo_root, "*.json"))
    data = []
    for path in json_files:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                sample = json.load(f)
                if sample and 'classDesc' in sample and sample['classDesc'] and sample['classDesc'].strip():
                    data.append(sample)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to parse {path}")
    return data

def load_all_repos(data_root):
    """
    加载 ./data/ 目录下所有 repo 中的类信息。
    """
    all_data = []
    for repo_name in os.listdir(data_root):
        repo_path = os.path.join(data_root, repo_name)
        if os.path.isdir(repo_path):
            repo_data = load_json_files(repo_path)
            all_data.extend(repo_data)
    return all_data

def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
    """
    将数据集划分为训练、验证和测试集。
    """
    from random import shuffle
    shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

def prepare_input_target_pairs(data, model):
    """
    将原始 json 数据列表转为 (input_text, target_text) 对列表。
    """
    input_texts = []
    target_texts = []
    for sample in data:
        input_texts.append(model.preprocess(sample))
        target_texts.append(sample['classDesc'])
    return input_texts, target_texts

if __name__ == '__main__':
    from model.summarizer import CodeSummaryModel
    model = CodeSummaryModel()
    dataset = load_all_repos("./data")
    print(f"Total loaded classes: {len(dataset)}")

    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train/Val/Test sizes: {len(train_set)}, {len(val_set)}, {len(test_set)}")

    x_train, y_train = prepare_input_target_pairs(train_set, model)
    print("Example:")
    print("Input:", x_train[0])
    print("Target:", y_train[0])
