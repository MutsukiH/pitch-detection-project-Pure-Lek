import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def plot_magnitude(signal, sample_rate, title, f_radio=1):
    X = np.fft.fft(signal)
    # print(X.shape)
    # print(X[0]) # ได้ค่ามาเ็นคู่อันดับของจำนวนจริงและจำนวนจินตภาพ
    X_mag = np.absolute(X) # ใส่ abs เพื่อให้จำนวนจริงที่ติดลบกลายเป็นบวก ทำให้ได้ค่า mag มา

    plt.figure(figsize=(12, 5))

    f = np.linspace(0, sample_rate, len(X_mag))
    f_bin = int(len(X_mag)*f_radio) #กำหนดช่วงของความถี่ที่จะแสดงถึง

    plt.plot(f[:f_bin], X_mag[:f_bin])
    plt.xlabel("Frequency(Hz)")
    plt.title(title)

    # plt.show()

sr = 22050

for i in range(48,60): # ช่วง octave 4
    ipd.Audio('./data/wav/wavnote'+str(i)+'.wav')
    signal, sr = librosa.load('./data/wav/wavnote'+str(i)+'.wav')
    plot_magnitude(signal, 22050, 'Note: '+NOTES[i%12], 0.3)

plt.show()


