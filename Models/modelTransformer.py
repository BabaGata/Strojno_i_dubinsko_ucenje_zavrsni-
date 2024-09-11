import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, n_features, hidden_dim=128, num_layers=16, dropout=0.1):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(n_features, n_features, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        x = x.mean(dim=1)
        return x