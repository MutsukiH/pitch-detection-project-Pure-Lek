import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

from notedataset import n_note, all_note, NoteDataset
from Network import RNN

# ----------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = r"data_seq\wavseq1.csv"
AUDIO_DIR = r"data_seq\wav_equ"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


# instantiating our dataset object and create data loader
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=256
) # จะแก้พารามิเตอร์ n_fft=1024 -> 512, hop_length=512 -> 256

noteseq = NoteDataset(ANNOTATIONS_FILE,
                        AUDIO_DIR,
                        mel_spectrogram,
                        SAMPLE_RATE)

train_dataloader = create_data_loader(noteseq, BATCH_SIZE)


# ------------------------------------------------------------------------------
# test
# ------------------------------------------------------------------------------

input_size = 221312
hidden_size = 128
output_size = n_note

rnn = RNN(input_size, hidden_size, output_size)
state_dict = torch.load(r"train_seq\model11\rnn.pth")
rnn.load_state_dict(state_dict)

max_length = 20

# Sample from a category and starting letter
def sample(input_tensor):
    with torch.no_grad():  # no need to track history in sampling
        hidden = rnn.initHidden()
        # output_name = start_letter
        list_note = []
        for i in range(max_length):
            output, hidden = rnn(input_tensor, hidden) # เอาข้อมูลที่ต้องการเทสเข้าโมเดล
            topv, topi = output.topk(1) # เอามาแค่ 1 ตัว จากลิสต์ยาว ๆ
            topi = topi[0][0]
            if topi == n_note - 1:
                break
            else:
                note = all_note[topi]
                list_note.append(note)
            # input = inputTensor(letter)

        return list_note

# Get multiple samples from one category and multiple starting letters
def samples(input_tensor, target_tensor):
    for input in input_tensor:
        print("predict : ", sample(input))
        print(target_tensor)

for input, target in train_dataloader:
    samples(input, target)
    print(f'end')
