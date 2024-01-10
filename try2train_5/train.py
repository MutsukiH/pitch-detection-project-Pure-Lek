import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio
import os

# -------------------------------------------------------------------
# Step 1: Convert sound WAV files to mel spectrograms
# -------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform
        self.file_list = os.listdir(audio_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # file_path = os.path.join(self.audio_dir, self.file_list[idx])
        file_path = ".\data\wav\wavnote48.wav"
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = self._mix_down_if_necessary(waveform)
        if self.transform:
            mel_spectrogram = self.transform(waveform)
        return mel_spectrogram
    
    def _mix_down_if_necessary(self, waveform): 
        if waveform.shape[0] > 1: # เปลี่ยนจากหลาย ๆ channal ให้มีแค่ 1 channal
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
# -------------------------------------------------------------------
# Define a simple CNN model
# -------------------------------------------------------------------
class PitchCNN(nn.Module):
    def __init__(self, num_classes):
        super(PitchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 13 * 13, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# -------------------------------------------------------------------
# Step 2: Create a transform for converting audio to mel spectrogram
# -------------------------------------------------------------------
transform = torchaudio.transforms.MelSpectrogram( # แปลงเป็น mel-spectrogram เราเพิ่มมันหลังจาก load ata 
        sample_rate=22050,
        n_fft=1024, # frame size    
        hop_length=512,
        n_mels=64 # number of mel
)

# -------------------------------------------------------------------
# Step 3: Build and train the CNN model
# -------------------------------------------------------------------
num_classes = 12
model = PitchCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have a train_loader (replace with your actual data loader)
data = AudioDataset(audio_dir='.\data', 
                    transform=transform)
train_loader = DataLoader(data, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for mel_spectrogram in train_loader:
        optimizer.zero_grad()
        outputs = model(mel_spectrogram)
        # Assuming you have ground truth labels for your data
        labels = torch.randint(0, num_classes, (mel_spectrogram.size(0),))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# -------------------------------------------------------------------
# Step 4: Evaluate the model and make predictions (not shown in this simplified example)
# -------------------------------------------------------------------


# ---------------------------
# test output shape before linear
# ---------------------------
# Assuming width and height are the same
# width = 64
# num_classes = 12
# num_classes = 12

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Define the model architecture
# model = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=(3, 3)),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=(2, 2)),
#     nn.Conv2d(32, 64, kernel_size=(3, 3)),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=(2, 2)),
#     nn.Flatten()
# )
# # model = model(num_classes)
# data = AudioDataset(audio_dir='.\data', 
#                     transform=transform)
# train_loader = DataLoader(data, batch_size=32, shuffle=True)

# # Forward pass to get the size after the second pooling
# data = AudioDataset(audio_dir='.\data', 
#                     transform=transform)
# train_loader = DataLoader(data, batch_size=32, shuffle=True)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for mel_spectrogram in train_loader:
#         optimizer.zero_grad()
#         outputs = model(mel_spectrogram)
#         # Assuming you have ground truth labels for your data
#         labels = torch.randint(0, num_classes, (mel_spectrogram.size(0),))
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# # Print the size of the tensor after the second pooling
# print("Size after second pooling:", output.size())
