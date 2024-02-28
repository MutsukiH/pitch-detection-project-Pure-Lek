import random
import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from notedataset import NoteDataset, n_note, all_note, create_data_loader
from Network import EncoderRNN, DecoderRNN, BahdanauAttention, AttnDecoderRNN


# ----------------------------------------------------------
# prepare Training
# ----------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0005

ANNOTATIONS_FILE = r"data_seq\wavseq1.csv"
AUDIO_DIR = r"data_seq\wav_equ"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050
time_length = 100
max_length = 60
batch_size = 10

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

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, _,target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor.long()) # เพรา่ะ embedded ต้องการ LongTensor หรือ IntTensor
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
            #                             epoch, epoch / n_epochs * 100, print_loss_avg))
            print('(%d %d%%) %.4f' % (epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)


hidden_size = 128
batch_size = 32

# for batch in train_data_loader:
#     signals, target_onehots, target_labels = batch
    # print("Signals Shape:", signals.shape)
    # print("Onehots Shape:", target_onehots.shape)
    # print("Labels Shape:", target_labels.shape)

    # input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(25600, hidden_size) # Signals Shape: torch.Size([1, 2, 25600])
decoder = AttnDecoderRNN(hidden_size, 14) # จำนวนโน๊ตที่เป็นไปได้ + sos, eos

train(train_data_loader, encoder, decoder, 80, print_every=5, plot_every=5)
