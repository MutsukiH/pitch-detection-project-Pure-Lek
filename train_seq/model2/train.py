import numpy as np

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import NoteSeqDataset
from rnn import PitchSeqModel

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.01

ANNOTATIONS_FILE = ".\data_seq\wavseq.csv"
AUDIO_DIR = "\wav\\"
SAMPLE_RATE = 22050
NUM_SAMPLES = 1024

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        loss = train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch)
        print("---------------------------")
        # if i % 50 == 0:
        #     print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
    print("Finished training")

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch):
    for input, target in data_loader:
        # input, target = input.to(device), target.to(device)
        input = input.view(-1, input.shape[2], input.shape[3])
        optimiser.zero_grad()
        output = model(input)
        print(target.shape)
        print(type(target))
        if isinstance(target, list):
            target = np.array(target)
        target = torch.tensor(target)
        loss = loss_fn(output.transpose(1, 2), target)  # Transpose เพื่อให้ output อยู่ในรูป sequence
        loss.backward()
        optimiser.step()
        writer.add_scalar("Loss/train", loss, epoch)

    print(f"loss: {loss.item()}")

    return loss



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=1024
    ) # จะแก้พารามิเตอร์ n_fft=1024 -> 512, hop_length=512 -> 256

    usd = NoteSeqDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    
    # construct model and assign it to device
    input_size = mel_spectrogram.n_mels  # จำนวน Mel bins
    hidden_size = 64  # จำนวน hidden units ของ RNN
    num_layers = 2  # จำนวน layer ของ RNN
    num_classes = 12  # จำนวนตัวโน๊ตที่เป็นไปได้ (C ถึง B♭)

    model = PitchSeqModel(input_size, hidden_size, num_layers, num_classes)
    model = model.to(device)


    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters())

    # train model
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    train(model, train_dataloader, loss_fn, optimiser, device, EPOCHS)
    writer.flush()

    # save model
    torch.save(model.state_dict(), r"train_seq\model2\rnn.pth")
    print("Trained feed forward net saved at rnn.pth")
