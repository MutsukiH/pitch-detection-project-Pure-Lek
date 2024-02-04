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

def evaluate(model, input_tensor):

    for input, target in input_tensor:
        model.eval()

        # สร้าง eos_mask สำหรับ input_tensor
        eos_mask = torch.ones(input.shape[1] + 1, dtype=torch.bool)
        eos_mask[-1] = False  # กำหนด False สำหรับตำแหน่ง EOS

        # คาดการณ์ sequence ด้วยโมเดล
        outputs = model(input, eos_mask)

        # แปลง outputs เป็น sequence ของตัวเลขที่บ่งบอกคลาส
        output_sequence = torch.argmax(outputs.squeeze(0), dim=1).tolist()

        # หยุดการคาดการณ์เมื่อเจอ EOS
        output_sequence = output_sequence[:-1]  # ตัด EOS ออก
        print('output :', output_sequence)
        print('target :', target)

    return output_sequence


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


input_size = 442368
hidden_size = 128
num_layers = 2
output_size = 13 # จน.โน๊ต + EOS 

data_loader = DataLoader(sound)
criterion = nn.CrossEntropyLoss()

model = RNNseqNote(input_size, hidden_size, num_layers, output_size)
state_dict = torch.load(r"train_seq\model7\rnn.pth")
model.load_state_dict(state_dict)


output_sequence = evaluate(model, sound)
# print(output_sequence)  # เช่น [6, 2, 12] หรือ [1, 4, 12]

