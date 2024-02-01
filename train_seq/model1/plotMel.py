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
        
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)# ทำ mal-spectrogram 
        return signal, label 
        # file_path = os.path.join(self.audio_dir, self.file_list[idx])
        # label = torch.tensor([0])
        # file_path = ".\data\wav\wavnote48.wav"
        # waveform, sample_rate = torchaudio.load(file_path)
        # waveform = self._mix_down_if_necessary(waveform)
        # if self.transform:
        #     mel_spectrogram = self.transform(waveform)
        # return mel_spectrogram, torch.tensor([0])
    
    def _cut_if_necessary(self, signal):
        # signal -> tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            last_dim_padding = (0, num_missing_samples) # (1, 1, 2, 2)
            # (1, num_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, waveform): 
        if waveform.shape[0] > 1: # เปลี่ยนจากหลาย ๆ channal ให้มีแค่ 1 channal
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def _get_audio_sample_path(self, index):
        fold = f'.\{self.annotations.iloc[index, 4]}' # เอาค่าเลขของ fold , iloc[index, 4] แถวที่และคอลัมที่ในไฟล์ csv
        print(fold)
        path = fold + self.audio_dir + self.annotations.iloc[index, 0]
        # path = os.path.join(fold, self.audio_dir, self.annotations.iloc[index, 0]) # รวมชื่อ path ให้เป็น str เดียวกัน
            
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
        n_mels=128 # number of mel
)

ANNOTATIONS_FILE = ".\data_seq\wavseq.csv"
AUDIO_DIR = "\wav\\"
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
        fig, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    print(specgram.shape) #! ได้ output เป็น torch.Size([1, 64, 216])
    print(specgram.shape[2])
    # print(librosa.power_to_db(specgram.view(128, 216)))
    # ดู range ของค่าสี
    min_val = librosa.power_to_db(specgram).min()
    max_val = librosa.power_to_db(specgram).max()
    print(min_val, max_val)
    mel = ax.imshow(librosa.power_to_db(specgram.view(specgram.shape[1], specgram.shape[2])), origin="lower", aspect="auto", interpolation="nearest",  cmap='viridis')
    fig.colorbar(mel, format='%+2.0f dB')

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

# list_freq = ['261.626', '277.183', '293.665', '311.127', '329.628', '349.228', '369.994', '391.995', '415.305', '440.000', '466.164','493.883']

for i, label in data:
    title = "spectrogram " + str(label)
    print(title)
    print(j)
    plot_spectrogram(i, title=title)
    
    plt.savefig('.\\train_seq\model1\MelSpec256\\'+'spectrogram_' + str(label)+'.png')
    j += 1
# plt.show()





# def create_data_loader(train_data, batch_size):
#     train_dataloader = DataLoader(train_data, batch_size=batch_size)
#     return train_dataloader

# train_dataloader = create_data_loader(data, 128)



