import matplotlib.pyplot as plt
import librosa, librosa.display # version 0.9.1
import IPython.display as ipd
import numpy as np

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

sr = 22050
frame_size = 2048 
Hop_size = 512

for i in range(48,60): # ช่วง octave 4
    ipd.Audio('./data/wav/wavnote'+str(i)+'.wav')
    signal, sr = librosa.load('./data/wav/wavnote'+str(i)+'.wav')
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    # print(mfcc.shape)
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(mfccs, 
                             sr=sr, 
                             x_axis='time')
    plt.colorbar(format='%+2f')
    plt.title('Note: '+ NOTES[i%12])
plt.show()
