# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.summarizer import CodeSummarizer
from utils import CodeSummaryDataset, build_vocab, collate_fn
from torch.optim import Adam

BATCH_SIZE = 8
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = CodeSummaryDataset('./data/', split='train')
vocab = build_vocab(train_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=lambda x: collate_fn(x, vocab), shuffle=True)

model = CodeSummarizer(vocab_size=len(vocab)).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, tgt_input, tgt_output in train_loader:
        src, tgt_input, tgt_output = src.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
