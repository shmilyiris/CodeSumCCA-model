import csv

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from model.summarizer import CodeSummaryModel
from utils import load_all_repos, split_dataset, prepare_input_target_pairs
from tqdm import tqdm
import os

class CodeSummaryDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_len=512, max_target_len=64):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_enc = self.tokenizer(
            input_text, truncation=True, padding='max_length', max_length=self.max_input_len, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, truncation=True, padding='max_length', max_length=self.max_target_len, return_tensors="pt"
        )

        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze()
        }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("[INFO] Loading dataset...")
    base_model = CodeSummaryModel()
    dataset = load_all_repos("./data")
    train_set, val_set, _ = split_dataset(dataset)
    x_train, y_train = prepare_input_target_pairs(train_set, base_model)
    x_val, y_val = prepare_input_target_pairs(val_set, base_model)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    train_dataset = CodeSummaryDataset(x_train, y_train, tokenizer)
    val_dataset = CodeSummaryDataset(x_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("[INFO] Starting training...")
    log_path = "./train.csv"
    with open(log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), f"./model/t5_code_summary/t5_code_summary_epoch{epoch+1}.pt")
        # 写入 train.csv
        with open(log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, round(avg_loss, 4)])

if __name__ == '__main__':
    train()