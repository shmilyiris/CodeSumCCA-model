import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv

import config


class CodeDataset(Dataset):
    def __init__(self, json_dir: str, graph_dir: str):
        self.json_dir = json_dir
        self.graph_dir = graph_dir
        self.file_names = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file = self.file_names[idx]

        with open(os.path.join(self.json_dir, file), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        graph_path = os.path.join(self.graph_dir, file.replace('.json', '.pt'))
        graph_data = torch.load(graph_path)  # 应为 GraphData 类型

        return json_data, graph_data


class SAFM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(SAFM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # JSON结构化信息编码器
        self.json_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 图结构编码器
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # 融合与解码
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, json_input, graph_data):
        # 假设 json_input 是 token id 序列，graph_data.x 是节点 token id
        json_emb = self.embedding(json_input)  # [B, L, H]
        json_encoded, _ = self.json_encoder(json_emb)

        node_x = self.embedding(graph_data.x)  # [N, H]
        x = self.gcn1(node_x, graph_data.edge_index)
        x = self.gcn2(x, graph_data.edge_index)

        # 融合（简单拼接+线性层）
        combined = torch.cat([json_encoded.mean(dim=1), x.mean(dim=0, keepdim=True).expand(json_encoded.size(0), -1)], dim=1)
        fused = self.fusion(combined).unsqueeze(1).repeat(1, json_input.size(1), 1)

        decoded, _ = self.decoder(fused)
        output = self.output_layer(decoded)
        return output


class HSAM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(HSAM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.field_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.method_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.class_decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, field_summaries, method_summaries):
        field_emb = self.embedding(field_summaries)  # [B, Lf, H]
        method_emb = self.embedding(method_summaries)  # [B, Lm, H]

        field_enc, _ = self.field_encoder(field_emb)
        method_enc, _ = self.method_encoder(method_emb)

        combined = torch.cat([field_enc.mean(dim=1), method_enc.mean(dim=1)], dim=1)  # [B, 2H]
        combined = combined.unsqueeze(1).repeat(1, field_summaries.size(1), 1)

        decoded, _ = self.class_decoder(combined)
        output = self.output_layer(decoded)
        return output


# 示例运行：加载数据 & 构造模型
if __name__ == '__main__':
    dataset = CodeDataset(config.json_path, config.graph_path)
    sample_json, sample_graph = dataset[0]

    vocab_size = 10000
    model_safm = SAFM(vocab_size, 256)
    model_hsam = HSAM(vocab_size, 256)

    # output = model_safm(json_input_tensor, graph_data_tensor)
    # output = model_hsam(field_tensor, method_tensor)
