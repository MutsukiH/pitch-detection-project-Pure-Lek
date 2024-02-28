import random
import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from NoteDataset import NoteDataset, n_note, all_note, create_data_loader
from Network import EncoderRNN, DecoderRNN

# ----------------------------------------------------------
# prepare Training
# ----------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0005

ANNOTATIONS_FILE = r"data_seq\wavseq5.csv"
AUDIO_DIR = r"data_seq\wav_equ"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050
time_length = 100
max_length = 60
batch_size = 10
n_fft=1024
hop_length=512
n_mels=1024

# instantiating our dataset object and create data loader
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=1024
) # จะแก้พารามิเตอร์ n_fft=1024 -> 512, hop_length=512 -> 256

noteseq = NoteDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )

train_data_loader = create_data_loader(noteseq, batch_size)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

input_dim = n_mels*time_length # Signals Shape: torch.Size([1, 2, 25600])
output_dim = len(all_note)
hidden_dim = 512
n_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=10, plot_every=10):
    # start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate) # สร้าง optimizer ที่ใช้ในการอัปเดตพารามิเตอร์ของโมเดล EncoderRNN 
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate) # สร้าง optimizer ที่ใช้ในการอัปเดตพารามิเตอร์ของโมเดล DecoderRNN
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%d %d%% %.4f' % (epoch, epoch / n_epochs * 100, print_loss_avg))

        # 
        # if epoch % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        signal_tensor, target_onehot, target_tensor = data

        encoder_optimizer.zero_grad() # ใช้สำหรับล้างค่า gradient ของพารามิเตอร์ในโมเดล เพื่อให้ไม่เกิดการสะสม gradient ของการอัปเดตพารามิเตอร์ข้ามรอบ
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(signal_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_onehot)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), # แปลงจาก [1, 2, 14] --> [2, 14] 
            target_tensor.view(-1) # แปลงจาก [1, 2] --> [2] 
        )
        # print('decoder_outputs: ', decoder_outputs.view(-1, decoder_outputs.size(-1)))
        # print('target_tensor', target_tensor.view(-1))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    print('-'*40)
    return total_loss / len(dataloader)


# ----------------------------------------------------------------------
# เรียกใช้ฟังก์ชั้นเพื่อ Train และเซฟโมเดล
# ----------------------------------------------------------------------

encoder = EncoderRNN(input_dim, hidden_dim).to(device)
decoder = DecoderRNN(hidden_dim, output_dim).to(device)

train(train_data_loader, encoder, decoder, 80, print_every=5, plot_every=5)

torch.save(encoder.state_dict(), f"train_seq_many2many\model4\enmodel.pth")
torch.save(decoder.state_dict(), f"train_seq_many2many\model4\demodel.pth")
