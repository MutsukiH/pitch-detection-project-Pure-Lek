import torch.optim as optim
import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from notedataset import NoteDataset, n_note, all_note, create_data_loader
from network import Encoder, Decoder, Seq2Seq

# ----------------------------------------------------------
# prepare Training
# ----------------------------------------------------------

# การสร้าง DataLoader
BATCH_SIZE = 10
ANNOTATIONS_FILE = r"data_seq\wavseq2.csv"
AUDIO_DIR = r"data_seq\wav_equ"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050
time_length = 100
max_length = 60
n_mels=256

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=256
)

noteseq = NoteDataset(
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    time_length,
    all_note,
    max_length
)

train_data_loader = create_data_loader(noteseq, BATCH_SIZE)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


# โมเดล
input_dim = n_mels*time_length # Signals Shape: torch.Size([1, 2, 25600])
output_dim = n_note
encoder_embedding_dim = 256
decoder_embedding_dim = 14
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
).to(device)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
).to(device)

model = Seq2Seq(encoder, decoder, device).to(device)


# การตั้งค่าอื่น ๆ
LEARNING_RATE = 0.001
EPOCHS = 100
CLIP = 1.0
TEACHER_FORCING_RATIO = 0

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # ใช้ ignore_index เพื่อไม่คำนวณ loss ในตำแหน่งที่เป็น padding

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm.tqdm(train_data_loader):
        signals, target_onehots, target_labels = batch
        signals, target_onehots, target_labels = signals.to(device), target_onehots.to(device), target_labels.to(device)
        # print(target_labels)
        optimizer.zero_grad()
        output = model(signals, target_onehots, TEACHER_FORCING_RATIO)

        # เปลี่ยนรูปร่าง output เพื่อให้สอดคล้องกับ target_labels
        output = output.view(-1, output.shape[-1])
        target_labels = target_labels.view(-1)

        loss = criterion(output, target_labels)
        print('output: ',output)
        print('target: ', target_labels)
        # loss.requires_grad = True
        loss.backward()

        # Clip gradients เพื่อป้องกัน gradient exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss}")

# Save the trained model if needed
# torch.save(model.state_dict(), "seq2seq_model.pth")
