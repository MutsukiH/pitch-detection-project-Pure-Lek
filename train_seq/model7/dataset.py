import os
import ast

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa
import numpy as np

class NoteDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate
                #  num_samples
                 ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        # self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = self._calculate_max_num_sample()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) 
        signal, sr = torchaudio.load(audio_sample_path)
        # signal = signal.to(self.device)
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        melsignal = self.transformation(signal)
        onset_frames, onset_tensor = self._onset_detection(signal, melsignal)

        signal_tensor = torch.cat((melsignal, onset_tensor.unsqueeze(0).unsqueeze(0).expand_as(melsignal)), dim=1)
        signal_tensor = self._to_width_multiply_high(signal_tensor) # ก่อนแปลงขนาด [1, 2048, 432]

        return signal_tensor, label_tensor
    
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
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        # fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, self.annotations.iloc[
            index, 0])
        print(path)
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 1]
        label = ast.literal_eval(label)
        return label
    
    def _calculate_max_num_sample(self):
        num_samples = 0
        for file_path in range(len(self.annotations)):
            audio_tensor, _ = torchaudio.load(os.path.join(self.audio_dir, self.annotations.iloc[file_path, 0]))
            num_samples = max(num_samples, audio_tensor.shape[1])
        return num_samples

    def _onset_detection(self, signal, mel_signal):
        audio_np = signal.numpy()[0]  # Convert to NumPy array
        onset_frames = librosa.onset.onset_detect(y=audio_np, sr=self.target_sample_rate)

        # Create feature representation combining spectrogram and onset information
        onset_tensor = torch.zeros(mel_signal.shape[2])  # Initialize onset tensor
        onset_tensor[onset_frames] = 1  # Set values corresponding to onsets to 1
        return onset_frames, onset_tensor
    

    def _targetTensor(self, label):
        pitch_indexes = [pitch for pitch in label]
        pitch_indexes.append(12) # EOS
        return torch.LongTensor(pitch_indexes)
    
    def _to_width_multiply_high(self, signal_tensor):
        # คำนวณ high และ width
        high = signal_tensor.shape[1]
        width = signal_tensor.shape[2]
        # เปลี่ยนรูปร่าง tensor โดยใช้ view
        signal_tensor = signal_tensor.view(1, high * width)
        # ตรวจสอบรูปร่าง
        # print(signal_tensor.shape)
        return signal_tensor
    
if __name__ == "__main__":

    ANNOTATIONS_FILE = r"data_seq\wavseq2.csv"
    AUDIO_DIR = r"data_seq\wav_equ"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=512 # 256 ผิด 2, 512 ผิด 1
    )
    # ms = mel_spectrogram(Signal)

    usd = NoteDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE
                            # NUM_SAMPLES,
                            # device
                            )
    print(f"There are {len(usd)} samples in the dataset.")
    for i in range(len(usd)):
        signal, label = usd[i]
        print('input shape : ', signal.shape)
        print('label : ', label)
        print('-----------------------------------------------')


