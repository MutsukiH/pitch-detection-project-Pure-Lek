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
        print(path)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 5] # label ในไฟล์ csv อยู่คอลัม 5


# -------------------------------------------------------------------
# Step 2: Create a transform for converting audio to mel spectrogram
# -------------------------------------------------------------------
transform = torchaudio.transforms.MelSpectrogram( # แปลงเป็น mel-spectrogram เราเพิ่มมันหลังจาก load ata 
        sample_rate=22050,
        n_fft=1024, # frame size    
        hop_length=512,
        n_mels=64 # number of mel
)

ANNOTATIONS_FILE = ".\data\datawav1.csv"
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

# signal2, sr2 = torchaudio.load('.\data\wav\wavnote48.wav')
# data2 = transform(signal2)

# print(data[0].shape)
# fig, axs = plt.subplots(4, 3)
def plot_spectrogram(specgram, title=None, ylabel="freq_bin",ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    print(specgram.shape) #! ได้ output เป็น torch.Size([1, 64, 216])
    ax.imshow(librosa.power_to_db(specgram.view(64, 216)), origin="lower", aspect="auto", interpolation="nearest")
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

j = 0

list_freq = ['261.626', '277.183', '293.665', '311.127', '329.628', '349.228', '369.994', '391.995', '415.305', '440.000', '466.164','493.883']

for i, label in data:
    title = "spectrogram '" + class_mapping[label] + "' freq : "+ list_freq[j] + " Hz"
    print(title)
    print(j)
    plot_spectrogram(i, title=title)
    plt.savefig('.\\try2train_6\MelSpec\\'+'spectrogram_' + class_mapping[label]+'.png')
    j += 1
# plt.show()





# def create_data_loader(train_data, batch_size):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size)
#     return train_dataloader

# train_dataloader = create_data_loader(data, 128)



