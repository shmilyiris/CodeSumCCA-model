import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 图卷积操作
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class CodeSummarizer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, vocab_size, embedding_dim):
        super(CodeSummarizer, self).__init__()
        # GCN 编码器
        self.gcn_encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        # LSTM 解码器
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=embedding_dim, batch_first=True)
        # 输出层，将LSTM的输出映射到词汇表空间
        self.fc = nn.Linear(embedding_dim, vocab_size)
        # 使用GPT2模型做语言生成
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, x, edge_index):
        # 编码阶段：通过GCN编码图
        encoded_features = self.gcn_encoder(x, edge_index)

        # LSTM解码阶段：将GCN输出作为LSTM输入
        lstm_out, _ = self.lstm(encoded_features.unsqueeze(1))  # 添加序列维度
        lstm_out = lstm_out.squeeze(1)  # 恢复到原始形状

        # 输出阶段：生成代码摘要的token
        logits = self.fc(lstm_out)

        return logits

    def generate_summary(self, x, edge_index, max_length=50):
        # 生成代码摘要
        with torch.no_grad():
            encoded_features = self.gcn_encoder(x, edge_index)
            lstm_out, _ = self.lstm(encoded_features.unsqueeze(1))
            lstm_out = lstm_out.squeeze(1)
            logits = self.fc(lstm_out)

        # 取最后的token输出，用GPT-2生成摘要
        inputs = self.tokenizer.decode(logits.argmax(dim=-1).cpu().numpy(), skip_special_tokens=True)
        inputs = self.tokenizer.encode(inputs, return_tensors="pt")

        # 使用GPT2生成摘要
        summary = self.gpt2.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        return self.tokenizer.decode(summary[0], skip_special_tokens=True)


# 示例初始化和使用
if __name__ == "__main__":
    # 假设我们有一个图的节点特征（图中的每个节点是AST的一个部分，特征为整数索引）
    # edge_index表示图的连接关系
    x = torch.tensor([[0], [1], [2], [3]], dtype=torch.float)  # 示例节点特征
    edge_index = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)  # 示例边关系

    # 设定输入的特征维度、隐藏层维度、输出维度等参数
    model = CodeSummarizer(in_channels=1, hidden_channels=32, out_channels=64, vocab_size=50257, embedding_dim=128)

    # 生成代码摘要
    summary = model.generate_summary(x, edge_index)
    print("Generated Summary:", summary)