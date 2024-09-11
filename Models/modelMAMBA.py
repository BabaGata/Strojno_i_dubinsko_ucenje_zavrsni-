import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, n_features, state_dim):
        super(SelectiveSSM, self).__init__()
        self.linear1 = nn.Linear(n_features, state_dim)
        self.linear2 = nn.Linear(state_dim, n_features)
    
    def forward(self, x):
        state = F.silu(self.linear1(x))
        output = self.linear2(state)
        return output

class MAMBA(nn.Module):
    def __init__(self, n_features, state_dim, hidden_size):
        super(MAMBA, self).__init__()
        self.hidden = nn.ModuleList()

        for _ in range(hidden_size):
            self.hidden.append(SelectiveSSM(n_features, state_dim))
            self.hidden.append(nn.Sequential(
                nn.Linear(n_features, state_dim),
                nn.SiLU(),
                nn.Linear(state_dim, n_features)
            ))
    
    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return x
    
class MAMBAModel(nn.Module):
    def __init__(self, n_features, hidden_size=32, output_dim=1):
        super(MAMBAModel, self).__init__()
        state_dim=120
        self.mamba = MAMBA(n_features, state_dim, hidden_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_features * state_dim, output_dim)
    
    def forward(self, x):
        x = self.mamba(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x