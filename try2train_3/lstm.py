from torch import nn
from torchsummary import summary

class PitchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PitchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        out = self.softmax(out)
        return out

# Example usage
input_size = 1  # Input size (e.g., one feature representing amplitude)
hidden_size = 64  # Number of LSTM units
num_layers = 3  # Number of LSTM layers
output_size = 72  # Number of output classes (pitch classes)

# Create an instance of the PitchLSTM model
model = PitchLSTM(input_size, hidden_size, num_layers, output_size)

# Generate sample input data (replace this with your actual data)
sequence_length = 100  # Length of your input sequence
batch_size = 1  # Number of sequences in a batch
input_data = torch.randn(batch_size, sequence_length, input_size)

# Forward pass
output = model(input_data)
print(output)
