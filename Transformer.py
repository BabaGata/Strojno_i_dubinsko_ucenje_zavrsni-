import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        src = self.transformer_encoder(src)
        output = self.decoder(src)
        return output

# Example usage
if __name__ == "__main__":
    batch_size = 32
    sequence_length = 100
    input_dim = 64
    hidden_dim = 128
    num_heads = 8
    num_layers = 4

    model = TransformerModel(input_dim, num_heads, hidden_dim, num_layers)
    input_data = torch.randn(batch_size, sequence_length, input_dim)
    
    # Pass input data through the model
    output = model(input_data)
    
    print(output.shape)  # Expected output shape: (batch_size, sequence_length, input_dim)
