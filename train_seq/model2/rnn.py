import torch
import torch.nn as nn
from torchsummary import summary


# กำหนดโมเดล RNN
class PitchSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PitchSeqModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x



# สร้างโมเดล RNN
# if __name__ == "__main__":
#     input_size = mel_spectrogram.n_mels  # จำนวน Mel bins
#     hidden_size = 64  # จำนวน hidden units ของ RNN
#     num_layers = 2  # จำนวน layer ของ RNN
#     num_classes = 12  # จำนวนตัวโน๊ตที่เป็นไปได้ (C ถึง B♭)

#     model = PitchModel(input_size, hidden_size, num_layers, num_classes)
#     model = model.to(device)
