import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa

class NoteSeqDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        label = torch.tensor(label)
        label = self._right_pad_if_necessary(label)
        signal = self.transformation(signal)
        # ตรวจสอบขนาดของ signal และจัดการให้เหมาะสมสำหรับ RNN
        if len(signal.shape) == 3:  # กรณีเป็น 3D Tensor
            signal = signal.squeeze(1)  # ลบ dimension ที่ 1 ออก
        return signal, label
    
    def _cut_if_necessary(self, signal):
        # signal -> tensor -> (1, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    # def _right_pad_if_necessary(self, signal):
    #     print(signal.shape)
    #     length_signal = signal.shape[1]
    #     if length_signal < self.num_samples:
    #         num_missing_samples = self.num_samples - length_signal
    #         # [1, 1, 1] -> [1, 1, 1, 0, 0]
    #         last_dim_padding = (0, num_missing_samples) # (1, 1, 2, 2)
    #         # (1, num_samples)
    #         signal = torch.nn.functional.pad(signal, last_dim_padding)
    #     return signal

    def _right_pad_if_necessary(self, signal):
        print(signal.shape)
        length_signal = signal.shape[0]  # เข้าถึงความยาวของมิติแรกเสมอ
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        # fold = f"fold{self.annotations.iloc[index, 5]}"
        # path = os.path.join(self.audio_dir, self.annotations.iloc[
        #     index, 0])
        fold = f'.\{self.annotations.iloc[index, 4]}' # เอาค่าเลขของ fold , iloc[index, 4] แถวที่และคอลัมที่ในไฟล์ csv
        # print(fold)
        path = fold + self.audio_dir + self.annotations.iloc[index, 0]
        print(path)
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 5].split()
        label[0] = label[0][1:-1] 
        print(label)
        label = str.split(label[0], ',')
        print(label)
        for i in range(len(label)): # แปลงตัวเลขข้าใน list ที่เป็น string ให้เป็น int
            label[i] = int(label[i])
        print(label)
        return label  # แยก string ของ label เป็น list ของตัวโน๊ต
    
if __name__ == "__main__":

    ANNOTATIONS_FILE = ".\data_seq\wavseq.csv"
    AUDIO_DIR = "\wav\\"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 1024

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=1024 # 256 ผิด 2, 512 ผิด 1
    )
    # ms = mel_spectrogram(Signal)

    usd = NoteSeqDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]

    a = 1

