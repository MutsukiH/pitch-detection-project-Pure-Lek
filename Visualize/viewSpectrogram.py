import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def plot_spec(Y, sample_rate, hop_length, title, y_axis = 'linear'):
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

for i in range(48,60): # ช่วง octave 4
    ipd.Audio('./data/wav/wavnote'+str(i)+'.wav')

    signal, sr = librosa.load('./data/wav/wavnote'+str(i)+'.wav')

    # Short-Time FT
    S_signal = librosa.stft(signal, n_fft=frame_size, hop_length=Hop_size)
    # print(S_signal)
    # print(type(S_signal[0][0])) # ได้จำนวนเชิงซ้อนแบบทศนิยม

    # Cal Spectrogram
    Y_scale = np.abs(S_signal)**2
    # print(type(Y_scale[0][0])) # ได้จำนวนจริงแบบทศนิยม
    title = 'Note: '+ NOTES[i%12]

    plot_spec(Y_scale, 22050, Hop_size,  title, y_axis='log')

plt.show()
