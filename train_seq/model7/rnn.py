import torch
import torch.nn as nn

class RNNseqNote(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, eos_mask):
        x, _ = self.lstm(x)
        x = self.fc(x)
        # x = x * eos_mask  # ใช้ eos_mask เพื่อจำกัดการคาดการณ์
        x = x * eos_mask.unsqueeze(-1) 
        return x
