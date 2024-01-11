import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio
import pandas as pd
import os
import matplotlib.pyplot as plt
import librosa, librosa.display # version 0.9.1
import IPython.display as ipd

# -------------------------------------------------------------------
# Step 1: Convert sound WAV files to mel spectrograms
# -------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file) # อ่านรายชื่อไฟล์ทั้งหมดที่มี
        self.audio_dir = audio_dir # ชื่อไดเรคทอรี่
        self.device = device # 
        self.transformation = transformation.to(self.device) # ทำ mel-spectrogram กับข้อมูลที่โหลดมาแล้ว และssign ให้เครื่องมือที่จะใช้รัน
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples # เพิ่มมา
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index) # รับ path ของตัวอย่าง audio  
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._mix_down_if_necessary(signal) # ทำให้มันเป็น mono ก่อน 
        signal = signal.to(self.device) # register ให้ tensor มาทำงานสิ่งที่สั่งบนเครื่องมือนี้
        signal = self.transformation(signal) # ทำ mal-spectrogram 
        return signal, label 
        # file_path = os.path.join(self.audio_dir, self.file_list[idx])
        # label = torch.tensor([0])
        # file_path = ".\data\wav\wavnote48.wav"
        # waveform, sample_rate = torchaudio.load(file_path)
        # waveform = self._mix_down_if_necessary(waveform)
        # if self.transform:
        #     mel_spectrogram = self.transform(waveform)
        # return mel_spectrogram, torch.tensor([0])
    
    def _mix_down_if_necessary(self, waveform): 
        if waveform.shape[0] > 1: # เปลี่ยนจากหลาย ๆ channal ให้มีแค่ 1 channal
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def _get_audio_sample_path(self, index):
        fold = f"wav\\" # เอาค่าเลขของ fold , iloc[index, 4] แถวที่และคอลัมที่ในไฟล์ csv
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[ # รวมชื่อ path ให้เป็น str เดียวกัน
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 5] # label ในไฟล์ csv อยู่คอลัม 5

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
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 25, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
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

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

# -------------------------------------------------------------------
# Step 3: Build and train the CNN model
# -------------------------------------------------------------------
num_classes = 12
model = PitchCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

ANNOTATIONS_FILE = ".\data\data1samplenote49.csv"
AUDIO_DIR = ".\data"
SAMPLE_RATE = 22050 # เพิ่มเข้ามา
NUM_SAMPLES = 22050


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")


data = AudioDataset(ANNOTATIONS_FILE,
                        AUDIO_DIR,
                        transform,
                        SAMPLE_RATE,
                        NUM_SAMPLES,
                        device)

train_dataloader = create_data_loader(data, 128)


# Training loop
num_epochs = 15
LEARNING_RATE = 1

# construct model and assign it to device
cnn = PitchCNN(num_classes).to(device)

# initialise loss funtion + optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(),
                            lr=LEARNING_RATE)

# train model
for epoch in range(num_epochs):
    # for mel_spectrogram in train_loader:
    #     optimizer.zero_grad()
    #     outputs = model(mel_spectrogram)
    #     # Assuming you have ground truth labels for your data
    #     labels = torch.randint(0, num_classes, (mel_spectrogram.size(0),))
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    print(f"Epoch {epoch+1}")

    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        print(target)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")
    print("---------------------------")
print("Finished training")

# save model
torch.save(cnn.state_dict(), ".\\try2train_5\cnnnet.pth")
print("Trained feed forward net saved at cnnnet.pth")


# -------------------------------------------------------------------
# Step 4: Evaluate the model and make predictions 
# -------------------------------------------------------------------
class_mapping = [ # ต้องการคำตอบมาเป็นชื่อ 
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
]

def predict(model, input, target, class_mapping): # ฟังก์ชั่นการทำนาย 
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        print(predicted_index)
        print(target)
    return predicted, expected

cnn = PitchCNN(num_classes)
state_dict = torch.load(".\\try2train_5\cnnnet.pth", map_location=torch.device('cpu'))
cnn.load_state_dict(state_dict)

# get a sample from the urban sound dataset for inference
# for i in range(12):
input, target = data[0][0], data[0][1] # [batch size, num_channels, fr, time]
print(input.shape)
input.unsqueeze_(0) # ใช้ tensor 
print(input.shape)
predicted, expected = predict(cnn, input, target,
                                class_mapping)
print(f"Predicted: '{predicted}', expected: '{expected}'")


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
