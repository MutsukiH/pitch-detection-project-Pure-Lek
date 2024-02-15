import torch
import torchaudio
import torch.nn as nn
import random
from torch.utils.data import DataLoader

from notedataset import NoteDataset, n_note, all_note
from Network import RNN

# ----------------------------------------------------------
# prepare Training
# ----------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0005

ANNOTATIONS_FILE = r"data_seq\wavseq_all_equ.csv"
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
# train_dataloader = create_data_loader(noteseq, BATCH_SIZE)

# for signal_tensor, label_tensor in train_dataloader:
#     print(signal_tensor.shape)
#     print(len(label_tensor[0][]))
#     break
# print(train_dataloader.shape)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1) # ยุบมิติ
    hidden = rnn.initHidden() # สร้าง intial weight 

    rnn.zero_grad() 

    loss = torch.Tensor([0]) # หรือจะใช้ loss = 0 ก็ได้ เป็นการกำหนดค่า loss เริ่มต้น
    # print('---------------------------')
    # print(f'input shape {input_line_tensor.shape}')
    # print('---------------------------')
    for i in range(input_line_tensor.size(0)):
        # print(input_line_tensor.size(0))
        # print(i, input_line_tensor[i])
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden) # เข้าโมเดล
        # print(output)
        # print( target_line_tensor[i])
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


rnn = RNN(884736, 128, n_note)
# print(n_note+128+n_note)

n_iters = 500
print_every = 5
plot_every = 50 # ช่วงของการแสดงค่า loss ว่าจะแสดงในทุก ๆ 
all_losses = []
total_loss = 0 # ต้องรีเซ็ทใหม่ทุกรอบ

for iter in range(1, n_iters + 1):
    for category_tensor, input_line_tensor, target_line_tensor, _ in noteseq:
        output, loss = train(category_tensor, input_line_tensor, target_line_tensor)
    total_loss += loss

    if iter % print_every == 0:
        print('---- iter:%d %d%% %.4f-----' % (iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


# save model
torch.save(rnn.state_dict(), r"train_seq\model13\rnntry2overfit.pth")
print("Trained feed forward net saved at rnntrainAll.pth")



# ------------------------------------------------------------------------------
# test
# ------------------------------------------------------------------------------


# max_length = 20

# # Sample from a category and starting letter
# def sample(category, start_letter='A'):
#     with torch.no_grad():  # no need to track history in sampling
#         category_tensor = categoryTensor(category) # ควรเอาไปใส่ใน prepare แล้วนำเข้าข้อมูลตอน trst เป็น tensor 
#         input = inputTensor(start_letter)
#         hidden = rnn.initHidden()

#         output_name = start_letter

#         for i in range(max_length):
#             output, hidden = rnn(category_tensor, input[0], hidden) # เอาข้อมูลที่ต้องการเทสเข้าโมเดล
#             topv, topi = output.topk(1) # เอามาแค่ 1 ตัว จากลิสต์ยาว ๆ
#             topi = topi[0][0]
#             if topi == n_letters - 1:
#                 break
#             else:
#                 letter = all_letters[topi]
#                 output_name += letter
#             input = inputTensor(letter)

#         return output_name

# # Get multiple samples from one category and multiple starting letters
# def samples(category, start_letters='ABC'):
#     for start_letter in start_letters:
#         print(sample(category, start_letter))

# samples('Russian', 'RUS')

# samples('German', 'GER')

# samples('Spanish', 'SPA')

# samples('Chinese', 'CHI')
