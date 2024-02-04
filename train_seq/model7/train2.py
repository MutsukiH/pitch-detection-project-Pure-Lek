import torch
import torch.nn as nn
from rnn import RNNseqNote

from dataset import NoteDataset

from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import math


ANNOTATIONS_FILE = r"data_seq\wavseq2.csv"
AUDIO_DIR = r"data_seq\wav_equ"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050
# NUM_SAMPLES = 607744

# สร้าง input data
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=512)

sound = NoteDataset(ANNOTATIONS_FILE,
                AUDIO_DIR,
                mel_spectrogram,
                SAMPLE_RATE
                # NUM_SAMPLES,
                )

print(f"There are {len(sound)} samples in the dataset.")

data_loader = DataLoader(sound)


# กำหนด parameter ต่าง ๆ
input_size = 442368
hidden_size = 128
num_layers = 2
output_size = 13 # จน.โน๊ต + EOS 

batch_size = 32
num_epochs = 50



model = RNNseqNote(input_size, hidden_size, num_layers, output_size)



# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# เทรนโมเดล
for epoch in range(10):
    for batch_idx, (input_sequences, target_sequences) in enumerate(data_loader):
        optimizer.zero_grad()

        # สร้าง eos_mask
        eos_masks = torch.stack([
            torch.ones(len(seq) + 1, dtype=torch.bool)
            for seq in target_sequences
        ])
        eos_masks[:, -1] = False  # กำหนด False สำหรับตำแหน่ง EOS

        outputs = model(input_sequences, eos_masks)
        print(outputs.shape, target_sequences.shape)
        outputs = outputs[:, :target_sequences.shape[1], :]
        loss = criterion(outputs.view(-1, output_size), target_sequences.view(-1))
        print(outputs.view(-1, output_size))
        print(target_sequences.view(-1))
        loss.backward()
        optimizer.step()

        # แสดงข้อมูลการเทรน
        print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "train_seq\model7\\rnn.pth")
print("Model trained and stored at rnnnet.pth")

