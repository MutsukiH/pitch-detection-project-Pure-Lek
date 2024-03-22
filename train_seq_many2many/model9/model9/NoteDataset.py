# ------------------------------------------------------------------------------------------------
# improt library ที่ใช้ทั้หมด
# ------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------


all_note = ['C', 'C#', 'D','Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb','B','sos', 'eos']
n_note = len(all_note)

# ------------------------------------------------------------------------------------------------
# สร้าง class เพื่อเตรียม Dataset
# ------------------------------------------------------------------------------------------------

class NoteDataset(Dataset):

    # กำหนด init ของคลาส ระบุคุณสมบัติเบื้องต้น ------------------------------------------
    def __init__(self, 
                 annotations_file, # csv รวมชื่อไฟล์เสียง
                 audio_dir, # โฟล์เดอร์ของไฟล์
                 transformation, # รูปแบบการแปลงเป็นเมลสเปคโตรแกรม
                 target_sample_rate, # sample rate ปกติ
                 time_length, # ความยาวของเวลาที่จะตัด
                 all_note, # ลิสท์ของ target ทั้งหมดที่เป็นไปได้
                 max_length # ความยาวของไฟล์ที่ยาวที่สุด
                 ):
        # กำหนดค่าให้ตัวแปรในคลาสด้วยตัวแปรที่รับเข้ามา
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.time_length = time_length
        self.all_note = all_note
        self.max_length = max_length
    # -----------------------------------------------------------------------------

    # เป็น method ที่ถ้าเราใช้ len() จะทำให้ได้จำนวนของข้อมูลที่เรามีอยู่ โดยดูจากจำนวนข้อมูลในไฟล์ csv
    def __len__(self):
        return len(self.annotations)
    # -----------------------------------------------------------------------------

    # เป็น method ที่กำหนดการเข้าถึงข้อมูลคลาสนี้ ว่าจะเข้าถึงยังไง ทำอะไรกับข้อมูลที่จะได้ไปบ้าง -----
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label_tensor = self._targetTensor(label) # คือ label ที่ใส่ eos
        input_tensor = self._inputTensor(label) # คือ label ที่ใส่ sos
        signal, sr = torchaudio.load(audio_sample_path)
        
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal) # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        melsignal = self.transformation(signal)
        # print(melsignal.shape)
        # เปลี่ยนจากที่มีหลาย batch ในแกนแรกให้เป็น 1 batch และแปลงขนาดของ Mel-spectrogram จาก (จำนวน batch, n_mel, เวลา) เป็น (1, เวลา, n_mel)
        # signal_tensor = melsignal.reshape(1, melsignal.shape[2], melsignal.shape[1]) 
        sequences = self.split_melspectrogram(melsignal)
        # print(sequences)
        signal_tensor = torch.stack(sequences) # เอาลิสต์ของ tensor ที่ทำการแบ่งก่อนแล้ว ให้มาเป็น tensor ก้อนใหญ่ ๆ โดยจะมีมิติเพิ่มเข้ามา 1 มิติ (ตอนนี้ข้อมูลมี 4 มิติแล้ว)
        signal_tensor = torch.flatten(signal_tensor, start_dim=1, end_dim=-1) # ทำให้อยู่ในรูปแบบของเวกเตอร์ 1 มิติ
        return signal_tensor, input_tensor, label_tensor, label
    # -----------------------------------------------------------------------------

    # เป็น method ที่เอาไว้รวม folder กับ path ที่ได้จาก csv ให้เป็น file path --------------
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[ # ใช้ os จะรวมแบบมี / ให้
            index, 0]) # เอาชื่อโฟล์เดอร์ในตัวแปร audio_dir มารวมกับ ชื่อไฟล์ที่ได้จากคอลัม 0 ในไฟล์ csv 
        return path
    # -----------------------------------------------------------------------------

    # เป็น method ที่เอาไว้เอาค่า label ของแต่ละไฟล์เสียงมา --------------------------------
    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 1] # เอาลาเบลจากไฟล์ csv คอลัม 1 มา
        label = ast.literal_eval(label) # แยก str ให้ออกมาเป็น list ของตัวเลข : '[1, 2]' -> [1, 2]
        return label
    # -----------------------------------------------------------------------------

    # เป็น method ที่เอาไว้เปลี่ยน label ที่เป็น list ให้เป็น tensor และเพิ่ม eos เข้าไปด้วย ------
    def _targetTensor(self, label):
        pitch_indexes = [int(pitch) for pitch in label] # วนให้แน่ใจว่า label เราเป็นลิสต์ตัวเลขแน่ ๆ
        pitch_indexes.append(int(self.all_note.index('eos'))) # เพิ่ม EOS เข้าไปท้ายลิสต์
        return torch.LongTensor(pitch_indexes) # เปลี่ยน list ให้เป็น tensor
    # -----------------------------------------------------------------------------
    
    # เป็น method ที่สร้าง tensor ที่มี [0, 0, ..., 0, 1 , 0, ..., 0] ตามจำนวนของโน๊ต -----
    def _inputTensor(self, label):
        labelwithsos = [all_note.index('sos')] # เอา sos ใส่ไว้หน้าสุดของ list
        for note in label: # วนโน๊ตตาม label 
            labelwithsos.append(note) # เพิ่มโน๊ตลงลิสต์ที่มี sos อยู่
        # สร้าง tensor ขนาด จำนวน label แถว 1 หลัก และหลักนั้นเป็น [0 , ..., 0] ที่ยาวเท่าจำนวน class ที่เป็นไปได้
        tensor = torch.zeros(len(labelwithsos), 1, n_note) 
        for li in range(len(labelwithsos)): # วนตามจำนวนของ label
            note = labelwithsos[li] # เอาเลข class ของ label มาใส่ไว้ในตัวแปร note
            tensor[li][0][note] = 1 # เปลี่ยน [0 , ..., 0] แต่ละอัน ให้เป็น 1 ในตำแหน่งที่เป็น class นั้น ๆ 
        return tensor
    # -----------------------------------------------------------------------------

    # เป็น method ที่เอาไว้ทำให้สัญญาณที่เอาเข้ามาถูกแบ่งด้วย sample rate ที่เท่ากัน -------------
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate: # ถ้า sample rate ที่ได้จากการโหลดเสียงและแปลงมาไม่เท่ากับ sample rate ที่ตั้งไว้
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate) # ให้แปลงใหม่ด้วยค่า sample rate ที่ตั้งไว้
            signal = resampler(signal)
        return signal
    # -----------------------------------------------------------------------------
    
    # เป็น method ที่เอาไว้ทำให้ Channel ของสัญญาณเสียง จาก 2 เป็น 1 channels ------------
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    # -----------------------------------------------------------------------------

    # ไม่ได้ใช้ _cut_if_necessary แล้ว

    # ไม่ได้ใช้ _right_pad_if_necessary แล้ว

    # ไม่ได้ทำ onset ที่หา Attack แล้ว

    # เป็น method ที่ใช้แบ่ง mel-spectrogram ของไฟล์ที่ยาวออกเป็นหลาย ๆ ภาพ ตาม time length ที่กำหนดไว้
    def split_melspectrogram(self, signal):
        # seq_len = math.ceil(signal.size(1) / self.time_length) # คำนวนว่าจะได้กี่ก้อน จริง ๆ ไม่จำเป็น 
        total_size = signal.size(2) # เอาความยาวเวลาทั้งหมดของไฟล์นั้น ๆ มา อยู่ตน. 1 เพราะรูปร่างคือ (1, เวลา, n_mel)
        split_size = self.time_length # เอาความยาวเวลาที่ต้องการแบ่งมาเก็บในตัวแปร
        remainder = total_size % split_size # คำนวนว่าจะเหลือเศษเวลาที่ไม่ครบตามที่กำหนดอยู่เท่าไหร่
        # print(total_size, seq_len, remainder)
        split_tensors = torch.split(signal, split_size, dim=2) # แบ่ง mel-spectrogram ออกตามเวลาที่กำหนด

        # ถ้าเหลือเศษ จะต้อง padding ก้อนเศษให้มีขนาดเท่า ๆ กับก้อนอื่น ๆ 
        if remainder > 0: # เช็คว่าเหลือเศษไหม
            last_tensor = split_tensors[-1] # ถ้าเหลือก็เลือกเอาก้อนสุดท้ายที่เป็นก้อนเศษมา
            padding_size = split_size - last_tensor.size(2) # คำนวนว่าต้อง padding อีกเท่าไหร่
            # กำหนดว่าจะทำ padding ที่มิติไหนบ้าง โดยไม่ทำกับมิติที่ 0 แต่ทำกับมิติที่ 3 ตามจำนวน padding size ที่คำนวนไว้
            last_tensor = torch.nn.functional.pad(last_tensor, (0, padding_size)) 
            split_tensors = split_tensors[:-1] + (last_tensor,) # เอาอันที่ padding มาแทนก้อนเศษอันสุดท้าย
        return split_tensors
    # -----------------------------------------------------------------------------
    


# ------------------------------------------------------------------------------------------------
# ทำให้ข้อมูลเป็น batch
# ------------------------------------------------------------------------------------------------

def create_data_loader(data, batch_size):

    signal_list, input_list, labeltensor_list, labelori_list = [], [], [], [] # สร้าง list เปล่าไว้
    num_batch = math.ceil(len(data) / batch_size) # คำนวนหาจำนวน batch ที่ต้องมี
    for i in range(len(data)): # วนเรียกเอาข้อมูลทีละชุด ๆ 
        signal, input_ten, label_ten, label_ori = data[i] # เอาข้อมูลทุกอันของ sample ที่ i ออกมา
        signal_list.append(signal) # เพิ่มลง list
        input_list.append(input_ten) # เพิ่มลง list
        labeltensor_list.append(label_ten) # เพิ่มลง list
        labelori_list.append(label_ori) # เพิ่มลง list

    train_data_loader = [] # ลิสต์ของข้อมูลที่จะเอาไปเทรน
    start = 0
    for batch in range(1, num_batch+1): # วนตามจำนวน batch ที่คำนวนได้
        # รวม tensor ใน list ตั้งแต่ตำแหน่งต้น batch ถึงท้าย batch ให้เป็น tensor ก้อนใหญ่ ๆ ก้อนเดียว และเติมค่าที่ตั้งไว้ในส่วนที่ไม่มีข้อมูลเพื่อให้ทุก batch มีค่าเท่ากัน 
        signal_batch = pad_sequence(signal_list[start:batch_size*batch], padding_value=-1,batch_first=True) 
        onehot_batch = pad_sequence(input_list[start:batch_size*batch], padding_value=0,batch_first=True) 
        labeltensor_batch = pad_sequence(labeltensor_list[start:batch_size*batch], padding_value=13,batch_first=True)
        each_batch = (signal_batch, onehot_batch, labeltensor_batch)
        train_data_loader.append(each_batch) # เพิ่มก้อน batch เข้าไปในลิสท์ของข้อมูลที่จะเอาไปเทรน
        start = batch_size*batch
    return train_data_loader

# ------------------------------------------------------------------------------------------------
# ทดสอบดึงข้อมูล
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__": # เป็นการตั้งเื่อนไขว่า ถ้ารักไฟล์นี้เป็นไฟล์หลัก จะทำการรันโค้ดด้านล่างต่อไปนี้

    # กำหนดค่าตัวแปรต่าง ๆ 
    ANNOTATIONS_FILE = r"data_seq\wav_seq_multi\train.csv"
    AUDIO_DIR = r"data_seq\wav_seq_multi"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050
    time_length = 100
    max_length = 60
    batch_size = 10

    # กำหนดค่าในการแปลงข้อมูลเป็น mel-spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=256 # 256 ผิด 2, 512 ผิด 1
    )

    # แปลงเป็น mel และทำให้เป็นข้อมูลก้อนนึงใหญ่ ๆ ที่ยังไม่แบ่ง batch
    notedata = NoteDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )

    print(f"There are {len(notedata)} samples in the dataset.")
    print('NoteDataset : ', notedata[0][0].shape)
    print('-----------------------------------------------')

    # แสดงขนาดของข้อมูลในแต่ละ sample
    # for i in range(len(notedata)):
        # print('-----------------------------------------------')
        # signal, input_ten, label,_ = notedata[i]
        # print('signal shape : ', signal.shape)
        # print('input  : ', input_ten)
        # print('label : ', label)
        # print('-----------------------------------------------')

    train_data_loader = create_data_loader(notedata, batch_size)

    print('Num of train_data_loader: ',len(train_data_loader), ' batches') # แสดงจำนวน batch

    # แสดงขนาดของข้อมูลในแต่ละ batch
    for i, batch in enumerate(train_data_loader):
        signals, input_tensor, target_tensor = batch
        print("Batch :", i + 1)
        print("Signals Shape:", signals.shape)
        print("input_tensor (Onehots) Shape:", input_tensor.shape)
        print("Labels Shape:", target_tensor.shape)
        print('-----------------------------------------------')

# 1
# Signals Shape: torch.Size([3, 2, 6400])
# Onehots Shape: torch.Size([3, 4, 1, 14])
# Labels Shape: torch.Size([3, 4])
# shape ข้อมูลถูกแล้ว 
