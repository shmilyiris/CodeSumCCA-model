# utils.py
import os
import json
import torch
from torch.utils.data import Dataset
from collections import Counter

class CodeSummaryDataset(Dataset):
    def __init__(self, root_path, split='train'):
        self.samples = []
        for repo in os.listdir(root_path):
            repo_path = os.path.join(root_path, repo)
            if not os.path.isdir(repo_path): continue
            for f in os.listdir(repo_path):
                if not f.endswith('.json'): continue
                with open(os.path.join(repo_path, f), 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    if not data:
                        continue
                    print(data.get('uniName', ''))
                    code = data.get('codeText', '')
                    summary = data.get('classDesc', '')
                    if code and summary:
                        self.samples.append((code, summary))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def tokenize(text):
    return text.strip().split()

def build_vocab(dataset, min_freq=2):
    counter = Counter()
    for code, summary in dataset:
        counter.update(tokenize(code))
        counter.update(tokenize(summary))
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    tokens = ['<SOS>'] + tokenize(text) + ['<EOS>']
    return [vocab.get(tok, vocab['<UNK>']) for tok in tokens]

def collate_fn(batch, vocab):
    codes, summaries = zip(*batch)
    code_ids = [torch.tensor(encode(code, vocab)) for code in codes]
    sum_ids = [torch.tensor(encode(summary, vocab)) for summary in summaries]

    code_ids = torch.nn.utils.rnn.pad_sequence(code_ids, batch_first=True, padding_value=vocab['<PAD>'])
    sum_input = [s[:-1] for s in sum_ids]
    sum_output = [s[1:] for s in sum_ids]

    sum_input = torch.nn.utils.rnn.pad_sequence(sum_input, batch_first=True, padding_value=vocab['<PAD>'])
    sum_output = torch.nn.utils.rnn.pad_sequence(sum_output, batch_first=True, padding_value=vocab['<PAD>'])

    return code_ids, sum_input, sum_output
