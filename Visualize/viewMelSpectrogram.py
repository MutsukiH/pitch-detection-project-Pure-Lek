import matplotlib.pyplot as plt
import librosa, librosa.display # version 0.9.1
import IPython.display as ipd
import numpy as np

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def plot_melspec(Y, sample_rate, hop_length, title, y_axis = 'mel'):
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(Y, 
                             sr=sample_rate, 
                             hop_length=hop_length, 
                             x_axis='time', 
                             y_axis=y_axis)
    plt.colorbar(format='%+2f')
    plt.title(title)


sr = 22050
frame_size = 2048 
Hop_size = 512

# Show Mel filter banks

# filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
# plt.figure(figsize=(12,5))
# librosa.display.specshow(filter_banks, sr=sr, x_axis='linear')
# plt.colorbar(format='%+2f')
# plt.show()


# show mel-spectrogram
for i in range(48,60): # ช่วง octave 4
    ipd.Audio('./data/wav/wavnote'+str(i)+'.wav')

    signal, sr = librosa.load('./data/wav/wavnote'+str(i)+'.wav')

    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128) # n_mel คือจำนวน mel band ที่จะแบ่ง
    # ปรับให้สเกลเป็น log
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    title = 'Note: '+ NOTES[i%12]

    plot_melspec(log_mel_spectrogram, 22050, Hop_size,  title, y_axis='mel')

plt.show()
