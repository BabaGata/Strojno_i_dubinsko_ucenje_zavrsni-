import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(SelectiveSSM, self).__init__()
        self.linear1 = nn.Linear(input_dim, state_dim)
        self.linear2 = nn.Linear(state_dim, input_dim)
    
    def forward(self, x):
        # Apply linear transformation and activation
        state = F.silu(self.linear1(x))
        # Apply the second linear transformation
        output = self.linear2(state)
        return output

class MAMBA(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(MAMBA, self).__init__()
        self.ssm = SelectiveSSM(input_dim, state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, input_dim)
        )
    
    def forward(self, x):
        # Pass through the Selective SSM block
        x = self.ssm(x)
        # Pass through the MLP block
        x = self.mlp(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 32
    sequence_length = 100
    input_dim = 64
    state_dim = 128

    model = MAMBA(input_dim, state_dim)
    input_data = torch.randn(batch_size, sequence_length, input_dim)
    
    # Pass input data through the model
    output = model(input_data)
    
    print(output.shape)  # Expected output shape: (batch_size, sequence_length, input_dim)
