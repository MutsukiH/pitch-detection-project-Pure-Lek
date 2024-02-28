import os
import ast

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import librosa
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence

all_note = ['C', 'C#', 'D','Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb','B','sos', 'eos']
n_note = len(all_note)

class NoteDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 time_length,
                 all_note,
                 max_length
                #  num_samples,
                #  device
                 ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        # self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        # self.num_samples = num_samples
        self.time_length = time_length
        self.all_note = all_note
        self.max_length = max_length
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) # คือ label ที่ใส่ eos
        input_tensor = self._inputTensor(label) # คือ label ที่ใส่ sos
        signal, sr = torchaudio.load(audio_sample_path)
        # signal = signal.to(self.device)
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)

        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)
        melsignal = self.transformation(signal)
        # print(melsignal.shape)
        signal_tensor = melsignal.reshape(1, melsignal.shape[2], melsignal.shape[1])
        # print(signal_tensor.shape)
        sequences = self.split_melspectrogram(signal_tensor)
        # print(sequences.shape)
        # print(sequences)
        signal_tensor = torch.stack(sequences)
        # print('-'*30)
        # print(signal_tensor)
        # print('-'*30)
        signal_tensor = torch.flatten(signal_tensor, start_dim=1, end_dim=-1)
        # print(signal_tensor.shape)
        # onset_frames, onset_tensor = self._onset_detection(signal, melsignal)
        # signal_tensor = torch.cat((melsignal, onset_tensor.unsqueeze(0).unsqueeze(0).expand_as(melsignal)), dim=1)
        # signal_tensor = self._to_width_multiply_high(signal_tensor) # ก่อนแปลงขนาด [1, 2048, 432]
        return signal_tensor, input_tensor, label_tensor, label
    
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
        return path

    def _get_audio_sample_label(self, index):
        # print(self.annotations.iloc[index, 1])
        label = self.annotations.iloc[index, 1]
        label = ast.literal_eval(label)
        return label

    # ลบ calculate max ออก
    
    def _onset_detection(self, signal, mel_signal):
        audio_np = signal.numpy()[0]  # Convert to NumPy array
        onset_frames = librosa.onset.onset_detect(y=audio_np, sr=self.target_sample_rate)

        # Create feature representation combining spectrogram and onset information
        onset_tensor = torch.zeros(mel_signal.shape[2])  # Initialize onset tensor
        onset_tensor[onset_frames] = 1  # Set values corresponding to onsets to 1
        return onset_frames, onset_tensor
    
    def _targetTensor(self, label):
        pitch_indexes = [int(pitch) for pitch in label]
        pitch_indexes.append(int(self.all_note.index('eos'))) # EOS
        return torch.LongTensor(pitch_indexes)
    
    def _to_width_multiply_high(self, signal_tensor):
        # คำนวณ high และ width
        high = signal_tensor.shape[1]
        width = signal_tensor.shape[2]
        # เปลี่ยนรูปร่าง tensor โดยใช้ view
        signal_tensor = signal_tensor.view(1, high * width)
        # ตรวจสอบรูปร่าง
        # print(signal_tensor.shape)
        # torch.flatten(signal_tensor)
        return signal_tensor
    
    # สร้างตัว input 
    def _inputTensor(self, label):
        # print(label)
        labelwithsos = [all_note.index('sos')]
        for note in label:
            labelwithsos.append(note)
        # print(labelwithsos)
        tensor = torch.zeros(len(labelwithsos), 1, n_note) # ขนาด len(line) แถว 1 หลัก 
        # print('shape input tensor', tensor.shape)
        for li in range(len(labelwithsos)):
            note = labelwithsos[li]
            tensor[li][0][note] = 1
        # print(f'input tensor : {tensor}')
        return tensor
    
    # แยกเมล
    def split_melspectrogram(self, signal):
        seq_len = math.ceil(signal.size(1) / self.time_length)
        total_size = signal.size(1)
        split_size = self.time_length
        remainder = total_size % split_size
        # print(total_size, seq_len, remainder)
        split_tensors = torch.split(signal, split_size, dim=1)

        # If there is a remainder, pad the last tensor
        if remainder > 0:
            last_tensor = split_tensors[-1]
            padding_size = split_size - last_tensor.size(1)
            last_tensor = torch.nn.functional.pad(last_tensor, (0, 0, 0, padding_size))
            split_tensors = split_tensors[:-1] + (last_tensor,)

        return split_tensors
    
    
def create_data_loader(data, batch_size):

    signal_list, onehot_list, label_list = [], [], []
    num_batch = math.ceil(len(data) / batch_size)
    for i in range(len(data)):
        signal, input_ten, label_ten, _ = data[i]
        signal_list.append(signal)
        onehot_list.append(input_ten)
        label_list.append(label_ten)

    train_data_loader = []
    start = 0
    for batch in range(1, num_batch+1):
        print(onehot_list[start:batch_size*batch])
        signal_batch = pad_sequence(signal_list[start:batch_size*batch], padding_value=-1,batch_first=True)
        print(signal_batch.shape)
        onehot_batch = pad_sequence(onehot_list[start:batch_size*batch], padding_value=0,batch_first=True)
        print(onehot_batch.shape)
        # for i in range(onehot_batch.size(0)):
        #     print(onehot_batch[i,:])
        label_batch = pad_sequence(label_list[start:batch_size*batch], padding_value=-1,batch_first=True)
        print(label_batch.shape)
        each_batch = (signal_batch, onehot_batch, label_batch)
        train_data_loader.append(each_batch)

        start = batch_size*batch
    return train_data_loader


if __name__ == "__main__":

    ANNOTATIONS_FILE = r"data_seq\wavseq1.csv"
    AUDIO_DIR = r"data_seq\wav_equ"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050
    time_length = 100
    max_length = 60
    batch_size = 10

    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    # print(f"Using device {device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=256 # 256 ผิด 2, 512 ผิด 1
    )
    # ms = mel_spectrogram(Signal)

    usd = NoteDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )
    print(f"There are {len(usd)} samples in the dataset.")
    for i in range(len(usd)):
        # print('-----------------------------------------------')
        signal, input_ten, label,_ = usd[i]
        print('signal shape : ', signal.shape)
        # print('input  : ', input_ten)
        # print('label : ', label)
        # print('-----------------------------------------------')

    train_data_loader = create_data_loader(usd, batch_size)

    print('LEN train_data_loader: ',len(train_data_loader))

    for batch in train_data_loader:
        signals, target_onehots, target_labels = batch
        print("Signals Shape:", signals.shape)
        print("Onehots Shape:", target_onehots.shape)
        print("Labels Shape:", target_labels.shape)

