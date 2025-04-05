import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from util.serializer import ASTSerializer

# Hyperparameters
HYPERPARAMS = {
    "embedding_dim": 768,
    "node_type_dim": 128,
    "input_code_length": 400,
    "code_encoder_layers": 12,
    "code_encoder_output_dim": 768,
    "attention_heads": 12,
    "key_value_dim": 64,
    "feedforward_dim": 2048,
    "graph_output_nodes": 300,
    "graph_attention_heads": 8,
    "graph_encoder_layers": 4,
    "decoder_layers": 6,
    "dropout": 0.2,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "optimizer": "Adam",
    "beam_size": 6
}


class CodeSummaryModel(nn.Module):
    def __init__(self, model_name):
        super(CodeSummaryModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.code_encoder = AutoModel.from_pretrained(model_name)
        self.ast_encoder = nn.LSTM(input_size=HYPERPARAMS["embedding_dim"],
                                   hidden_size=HYPERPARAMS["embedding_dim"], batch_first=True)
        self.fc = nn.Linear(2 * HYPERPARAMS["embedding_dim"], HYPERPARAMS["embedding_dim"])
        self.decoder = nn.Linear(HYPERPARAMS["embedding_dim"], self.tokenizer.vocab_size)

    def forward(self, code_text, ast_sequence):
        code_tokens = self.tokenizer(code_text, return_tensors="pt", padding=True, truncation=True)
        code_features = self.code_encoder(**code_tokens).last_hidden_state[:, 0, :]

        ast_embedded = self.tokenizer(ast_sequence, return_tensors="pt", padding=True, truncation=True)
        ast_features, _ = self.ast_encoder(ast_embedded.input_ids.float())
        ast_features = ast_features[:, -1, :]

        combined_features = torch.cat((code_features, ast_features), dim=1)
        combined_features = self.fc(combined_features)

        output = self.decoder(combined_features)
        return output


def train(model, train_data, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for code_text, ast_tree, summary in train_data:
            optimizer.zero_grad()
            ast_sequence = ASTSerializer.serialize_(ast_tree)
            output = model(code_text, ast_sequence)
            summary_tokens = model.tokenizer(summary, return_tensors="pt", padding=True, truncation=True).input_ids
            loss = criterion(output, summary_tokens.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


if __name__ == '__main__':
    model_name = "bert-base-uncased"
    model = CodeSummaryModel(model_name)
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    with open("data/train_data", encoding='utf_8') as f:
        train_data = f.read()
        train(model, train_data, optimizer, criterion)
