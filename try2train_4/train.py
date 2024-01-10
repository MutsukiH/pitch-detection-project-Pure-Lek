#! error เยอะมาก 

import torch
import torchaudio
# import tensorflow as tf
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import librosa, librosa.display # version 0.9.1
import IPython.display as ipd
import numpy as np

# from dataset import midiDataset
# from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = ".\Meta_train1.csv"
AUDIO_DIR = ".\data\wav"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


# ----------------------------------
#     กำหนดตัวแปร train, test
# ----------------------------------
sr = 22050
frame_size = 2048 
Hop_size = 512
width = 128
num_classes = 12
trainNote = ".\data\wav\wavnote48.wav"
testNote = ".\data\wav\wavnote48.wav"

ipd.Audio(trainNote)
ipd.Audio(testNote)
x_train, sr = librosa.load(trainNote)
x_test, sr = librosa.load(testNote)

# ทำ mel สำหรับ train
mel_spectrogram_x_train = torchaudio.transforms.MelSpectrogram(x_train, 22050, 2048, 512, 128) # n_mel คือจำนวน mel band ที่จะแบ่ง
# mel_spectrogram_x_train = librosa.feature.melspectrogram(y=x_train, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
# ปรับให้สเกลเป็น log
# log_mel_spectrogram_x_train = librosa.power_to_db(mel_spectrogram_x_train)

# ทำ mel สำหรับ test
mel_spectrogram_x_test = torchaudio.transforms.MelSpectrogram(x_test, 22050, 2048, 512, 128)
# mel_spectrogram_x_test = librosa.feature.melspectrogram(y=x_train, sr=22050, n_fft=2048, hop_length=512, n_mels=128) # n_mel คือจำนวน mel band ที่จะแบ่ง
# ปรับให้สเกลเป็น log
# log_mel_spectrogram_x_test = librosa.power_to_db(mel_spectrogram_x_train)


# x_train = np.array(log_mel_spectrogram_x_train)  
#ใช้ numpy จัดการข้อมูลที่ถูกอ่านมาแล้วแปลงเป็น array
y_train = [0]
# x_test = np.array(log_mel_spectrogram_x_test)  
y_test = [0]
# x_train = x_train.astype('float32') #เปลี่ยนประเภทข้อมูลเป็นเลขทศนิยม
# x_test = x_test.astype('float32')

# พออ่านข้อมูลออกมาแล้วจะได้ไฟล์ RGB หลายขนาด 
# ต้องแปลงข้อมูลให้มีค่าไม่เกิน 1 เพื่อง่ายต่อการใช้งาน
# ด้วยการหารด้วย 255 และจะง่ายด้วยเมื่อเอาไปเข้า keras
# x_train /= 255  
# x_test /= 255  
# x_train = torch.from_numpy(x_train)
# x_test = torch.from_numpy(x_test)

# ----------------------------------
#          สร้างโมเดลเทรน
# ----------------------------------
# model0 = nn.Sequential([
#     nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3)),
#     nn.ReLU(),
#     nn.MaxPool2d(2),
#     nn.Conv2d(in_channels=3,out_channels=128,kernel_size=(3,3)),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=(2, 2)),
#     nn.Flatten(),
#     nn.Softmax(num_classes)])
model0 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3)),  # Adjust in_channels
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(128 * (width // 4) * (width // 4), num_classes),  # Adjust input size for Linear layer
    nn.Softmax(dim=1)  # Adjust the dimension for softmax
)


# ----------------------------------
#        กำหนดการเรียนรู้ของโมเดล
# ----------------------------------
output = model0(x_train)
print(output.shape)
# model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics= ['accuracy'])

# batch_size = 32
epochs = 10

# 
# history = model0.fit(x_train, y_train ,batch_size=BATCH_SIZE, epochs=epochs ,validation_data=(x_test, y_test))
