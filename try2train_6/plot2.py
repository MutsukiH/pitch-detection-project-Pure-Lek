import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio
from torchaudio.utils import download_asset
import pandas as pd
import os
import matplotlib.pyplot as plt
import librosa, librosa.display # version 0.9.1
import IPython.display as ipd

SAMPLE_NOTE = ".\data\wav\wavnote48.wav"

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    print(specgram.shape) # ได้ output เป็น torch.Size([64, 216])
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show()

NOTE_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_NOTE)

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 64
sample_rate = 22050

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(NOTE_WAVEFORM)

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

for i, label in melspec:
    title = "spectrogram '" + class_mapping[label] + "' freq : "+ list_freq[j] + " Hz"
    print(title)
    print(j)
    plot_spectrogram(i, title=title)
    j += 1
plt.show()
