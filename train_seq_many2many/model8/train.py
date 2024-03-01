import random
import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
# import numpy as np

from NoteDataset import NoteDataset, n_note, all_note, create_data_loader
from Network import EncoderRNN, DecoderRNN

from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/model8_2')

# ----------------------------------------------------------
# prepare Training
# ----------------------------------------------------------

BATCH_SIZE = 128
# EPOCHS = 10
LEARNING_RATE = 0.0005

ANNOTATIONS_FILE_train = r"data_seq\wav_seq_multi\train.csv"
ANNOTATIONS_FILE_test = r"data_seq\wav_seq_multi\test.csv"
AUDIO_DIR = r"data_seq\wav_seq_multi"
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

noteseq = NoteDataset(ANNOTATIONS_FILE_train, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )

noteseq_test = NoteDataset(ANNOTATIONS_FILE_test, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )

train_data_loader = create_data_loader(noteseq, batch_size)
test_dataloader = create_data_loader(noteseq_test, batch_size)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

input_dim = n_mels*time_length # Signals Shape: torch.Size([1, 2, 25600])
output_dim = len(all_note)
hidden_dim = 512
n_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_dataloader, test_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=10, plot_every=10):
    # start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate) # สร้าง optimizer ที่ใช้ในการอัปเดตพารามิเตอร์ของโมเดล EncoderRNN 
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate) # สร้าง optimizer ที่ใช้ในการอัปเดตพารามิเตอร์ของโมเดล DecoderRNN
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%d %d%% %.4f' % (epoch, epoch / n_epochs * 100, print_loss_avg))

        
        if epoch % plot_every == 0:
            # ...log the running loss
            writer.add_scalar('training loss',
                            plot_loss_total / len(train_dataloader),
                            epoch)
            plot_loss_total = 0

        encoder.eval()
        decoder.eval()
 
        with torch.no_grad():
            total_loss_val = 0

            for b in test_dataloader:
                signal_val, target_onehot_val, target_val = b
                encoder_outputs_val, encoder_hidden_val = encoder(signal_val)
                decoder_outputs_val, _ = decoder(encoder_outputs_val, encoder_hidden_val, target_onehot_val)
                
                loss_val = criterion(
                    decoder_outputs_val.view(-1, 14),
                    target_val.view(-1)
                    )
                total_loss_val += loss_val.item()

            if epoch % plot_every == 0:
                writer.add_scalar('test loss',
                            total_loss_val / len(test_dataloader),
                            epoch)
                plot_loss_total = 0
                # acc_val = multiclass_accuracy(decoder_outputs_val.view(-1, 14), 
                #                               target_val.view(-1), 
                #                               num_classes=len(rhythm_dict), ignore_index=-1)
                # total_acc_val += acc_val

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        signal_tensor, target_onehot, target_tensor = data

        encoder_optimizer.zero_grad() # ใช้สำหรับล้างค่า gradient ของพารามิเตอร์ในโมเดล เพื่อให้ไม่เกิดการสะสม gradient ของการอัปเดตพารามิเตอร์ข้ามรอบ
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(signal_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_onehot)

        # print(decoder_outputs.size())
        # print('decoder_outputs : ', decoder_outputs.shape) # --> torch.Size([3, 4, 14])
        # print('decoder_outputs.view(-1, 14) : ',decoder_outputs.view(-1, 14).shape) # --> torch.Size([12, 14])
        # print('target_tensor : ', target_tensor.shape) # --> torch.Size([3, 4])
        # print('target_tensor.view(-1) : ', target_tensor.view(-1).shape) # --> torch.Size([12])

        loss = criterion( # คำนวนค่า loss ระหว่างค่าที่โมเดลทายออกมากับค่าจริง
            decoder_outputs.view(-1, 14), # แปลงจาก [1, 2, 14] --> [2, 14] 
            target_tensor.view(-1) # แปลงจาก [1, 2] --> [2] 
        )
        # print('decoder_outputs: ', decoder_outputs.view(-1, decoder_outputs.size(-1)))
        # print('target_tensor', target_tensor.view(-1))
        loss.backward() # เริ่มกระบวนการ backpropagation เพื่อคำนวณ gradient ของพารามิเตอร์ทุกตัวภายในโมเดล

        encoder_optimizer.step() # อัปเดตพารามิเตอร์ของโมเดล
        decoder_optimizer.step()
        
        total_loss += loss.item() # เอาค่า loss มาบวกเข้าไป
    
    return total_loss / len(dataloader)


# ----------------------------------------------------------------------
# เรียกใช้ฟังก์ชั้นเพื่อ Train และเซฟโมเดล
# ----------------------------------------------------------------------

encoder = EncoderRNN(input_dim, hidden_dim).to(device)
decoder = DecoderRNN(hidden_dim, output_dim).to(device)

train(train_data_loader, test_dataloader, encoder, decoder, 100, print_every=5, plot_every=5)

torch.save(encoder.state_dict(), f"train_seq_many2many\model8\enmodel.pth")
torch.save(decoder.state_dict(), f"train_seq_many2many\model8\demodel.pth")
