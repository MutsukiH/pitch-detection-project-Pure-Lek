import os

import torch
from torch.utils.data import Dataset #เพื่อใช้ปรับ custom data set 
import pandas as pd
import torchaudio

# สำหรับใช้ custom data set
class midiDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device): # โครงสร้าง
        self.annotations = pd.read_csv(annotations_file) # อ่านรายชื่อไฟล์ทั้งหมดที่มี
        self.audio_dir = audio_dir # ชื่อไดเรคทอรี่
        self.device = device # 
        self.transformation = transformation.to(self.device) # ทำ mel-spectrogram กับข้อมูลที่โหลดมาแล้ว และssign ให้เครื่องมือที่จะใช้รัน
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples # เพิ่มมา

    def __len__(self): # ใช้อธิบายว่าเราจะใช้ syntax len ได้ยังไง คำนวนยังไง
        return len(self.annotations) # ต้องการคืนค่าจำนวนของ sample ที่มี

    def __getitem__(self, index): # เราจะได้ item มายังไง ex. a_list[1] -> a_list.__getitem(1)
        audio_sample_path = self._get_audio_sample_path(index) # รับ path ของตัวอย่าง audio  
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device) # register ให้ tensor มาทำงานสิ่งที่สั่งบนเครื่องมือนี้
        # signal -> (num_channal, samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal, sr) # ทำให้มั่นใจว่าทุก sample มีการแบ่ง sample mี่เหมือนกัน
        signal = self._mix_down_if_necessary(signal) # ทำให้มันเป็น mono ก่อน 
        signal = self._cut_if_necessary(signal) # ใช้ในกรณีที่ sample ใน audio file มีมากกว่าค่าที่เราต้องการ
        signal = self._right_pad_if_necessary(signal) #ใช้ในกรณีที่ sample ใน audio file มีน้อยกว่าหรือมากกว่าค่าที่เราต้องการ
        signal = self.transformation(signal) # ทำ mal-spectrogram 
        return signal, label 
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples] # slide มัน (1, 50000) -> (1,22050)
        return signal

    def _right_pad_if_necessary(self, signal): #ใช้ในกรณีที่ sample ใน audio file มีน้อยกว่าค่าที่เราต้องการ
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length_signal # คิดว่าต้องเพิ่มอีกเท่าไหร่ 
            last_dim_padding = (0, num_missing_samples) #(1, 2) : [1, 1, 1] -> [0, 1, 1, 1, 0, 0] 
            signal = torch.nn.functional.pad(signal, last_dim_padding) 
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate: # ใช้เมื่อ sample rate ไม่เท่ากับ target_sample_rate
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0] > 1: # เปลี่ยนจากหลาย ๆ channal ให้มีแค่ 1 channal
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"note/octave{self.annotations.iloc[index, 4]}" # เอาค่าเลขของ fold , iloc[index, 4] แถวที่และคอลัมที่ในไฟล์ csv
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[ # รวมชื่อ path ให้เป็น str เดียวกัน
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 5] # label ในไฟล์ csv อยู่คอลัม 5


# if __name__ == "__main__":
#     ANNOTATIONS_FILE = ".\Meta_train1.csv"
#     AUDIO_DIR = ".\data"
#     usd = midiDataset(ANNOTATIONS_FILE, AUDIO_DIR)
#     print(f"There are {len(usd)} samples in the dataset.")
#     signal, label = usd[0]


if __name__ == "__main__":
    ANNOTATIONS_FILE = ".\Meta_train1.csv"
    AUDIO_DIR = ".\data"
    SAMPLE_RATE = 22050 # เพิ่มเข้ามา
    NUM_SAMPLES = 22050

    if torch.cuda.is_available(): # เพิ่มตรงนี้
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram( # แปลงเป็น mel-spectrogram เราเพิ่มมันหลังจาก load ata 
        sample_rate=SAMPLE_RATE,
        n_fft=1024, # frame size    
        hop_length=512,
        n_mels=64 # number of mel
    )

    usd = midiDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
