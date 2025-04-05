# model/summarizer.py
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        i = torch.arange(0, emb_size, 2)
        angle_rates = pos / (10000 ** (i / emb_size))
        pe = torch.zeros(maxlen, emb_size)
        pe[:, 0::2] = torch.sin(angle_rates)
        pe[:, 1::2] = torch.cos(angle_rates)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CodeSummarizer(nn.Module):
    def __init__(self, vocab_size, emb_size=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        self.transformer = Transformer(d_model=emb_size, nhead=nhead,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1),
                                  src_key_padding_mask=src_mask,
                                  tgt_key_padding_mask=tgt_mask)
        output = self.fc_out(output.transpose(0, 1))
        return output
