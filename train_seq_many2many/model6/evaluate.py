import random
import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from NoteDataset import NoteDataset, n_note, all_note, create_data_loader
from Network import EncoderRNN, DecoderRNN

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

BATCH_SIZE = 128
# EPOCHS = 10
LEARNING_RATE = 0.0005

ANNOTATIONS_FILE = r"data_seq\wav_seq_multi\train.csv"
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

noteseq = NoteDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            time_length,
                            all_note,
                            max_length
                            )

train_data_loader = create_data_loader(noteseq, batch_size)


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

input_dim = n_mels*time_length # Signals Shape: torch.Size([1, 2, 25600])
output_dim = len(all_note)
hidden_dim = 512
n_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderRNN(input_dim, hidden_dim).to(device)
decoder = DecoderRNN(hidden_dim, output_dim).to(device)

state_dicten = torch.load(r"train_seq_many2many\model6\enmodel.pth")
encoder.load_state_dict(state_dicten)
state_dictde = torch.load(r"train_seq_many2many\model6\demodel.pth")
decoder.load_state_dict(state_dictde)

def samples(encoder, decoder, train_data_loader):
    count = 0
    correct = 0
    for signal_tensor, target_onehot, target_tensor in train_data_loader:
        prediction = evaluate(encoder, decoder, signal_tensor, target_onehot, target_tensor)
        
        
        prediction_list = prediction.tolist()
        target_list = target_tensor.tolist()
        for k in range(len(prediction_list)):
            print('prediction: ', prediction_list[k])
            print('targer: ', target_list[k])

            if prediction_list[k] == target_list[k]:
                correct += 1
            count += 1
    print('-'*30)
    print(f'Accuracy : {correct/count*100}')

def evaluate(encoder, decoder, signal_tensor, target_onehot, target_tensor):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(signal_tensor)
        decoder_outputs, decoder_hidden = decoder(encoder_outputs, encoder_hidden, target_onehot)

        _, topi = decoder_outputs.topk(1)
        # print(topi.shape)
        decoded_ids = topi.squeeze() # ลบมิติที่มีค่าเท่ากับ 1 ออกจาก tensor เพื่อให้ได้รูปร่างของ tensor ที่แน่นอนมากขึ้น
        # print(decoded_ids.shape)
        # print(decoded_ids)

    return decoded_ids



samples(encoder, decoder, train_data_loader)
