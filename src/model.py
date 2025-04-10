# src/model.py
import torch
import torch.nn as nn

class SentimentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x