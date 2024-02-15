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

ANNOTATIONS_FILE = r"data_seq\wavseq14.csv"
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
    n_mels=1024
) # จะแก้พารามิเตอร์ n_fft=1024 -> 512, hop_length=512 -> 256

noteseq = NoteDataset(ANNOTATIONS_FILE,
                                        AUDIO_DIR,
                                        mel_spectrogram,
                                        SAMPLE_RATE
                                    )

# One-hot matrix of first to last letters not including EOS for input
def inputTensor(label):
        tensor = torch.zeros(len(label), 1, n_note) # ขนาด len(line) แถว 1 หลัก 
        print('shape input tensor', tensor.shape)
        for li in range(len(label)):
            note = label[li]
            print(note)
            tensor[li][0][all_note.index(note)] = 1
        # print(f'input tensor : {tensor}')
        return tensor
# ------------------------------------------------------------------------------
# test
# ------------------------------------------------------------------------------

max_length = 20
rnn = RNN(884736, 128, n_note)
state_dict = torch.load(r"train_seq\model12\rnn.pth")
rnn.load_state_dict(state_dict)

# Sample from a category and starting letter
def sample(category_tensor):
    start_letter=['sos']
    with torch.no_grad():  # no need to track history in sampling
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = []

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden) # เอาข้อมูลที่ต้องการเทสเข้าโมเดล
            topv, topi = output.topk(1) # เอามาแค่ 1 ตัว จากลิสต์ยาว ๆ
            topi = topi[0][0]
            print(topi)
            if topi == n_note - 1:
                break
            else:
                letter = all_note[topi]
                output_name.append(letter)
            print(letter)
            input = inputTensor([letter])

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(noteseq):
    for category_tensor, _, _, target in noteseq:
        print('prediction: ', sample(category_tensor))
        target_name = []
        for i in target :
            target_name.append(all_note[i])
        print('targer: ', target_name)

samples(noteseq)

