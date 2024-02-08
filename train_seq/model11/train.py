import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

from notedataset import n_note, NoteDataset
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

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(input_tensor, target_tensor):
    target_tensor.unsqueeze_(-1) # ยุบมิติ
    hidden = rnn.initHidden() # สร้าง intial weight 

    rnn.zero_grad() 

    loss = torch.Tensor([0]) # หรือจะใช้ loss = 0 ก็ได้ เป็นการกำหนดค่า loss เริ่มต้น

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden) # เข้าโมเดล
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)


rnn = RNN(n_note, 128, n_note)

n_iters = 100000
print_every = 5000
plot_every = 500 # ช่วงของการแสดงค่า loss ว่าจะแสดงในทุก ๆ 
all_losses = []
total_loss = 0 # ต้องรีเซ็ทใหม่ทุกรอบ

for iter in range(1, n_iters + 1):
    for input, target in train_dataloader:
        output, loss = train(input, target)
        total_loss += loss

    if iter % print_every == 0:
        print('%d %d%% %.4f' % (iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


# save model
torch.save(rnn.state_dict(), r"train_seq\model9\rnnsample.pth")
print("Trained feed forward net saved at rnn.pth")


