import torch
import torchaudio

from rnn import PitchSeqModel
from dataset import NoteSeqDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [ # ต้องการคำตอบมาเป็นชื่อ 
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted = predictions[0].argmax(0)
    return predicted, expected

def detect_pitch_sequence(model, input, target):
    # Feed windows to model
    input = input.view(-1, input.shape[2], input.shape[3])
    predictions = model(input)

    # Decode predictions to note sequence
    note_sequence = torch.argmax(predictions, dim=1)
    return note_sequence, target


if __name__ == "__main__":
    # load back the model
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=1024
    ) # จะแก้พารามิเตอร์ n_fft=1024 -> 512, hop_length=512 -> 256
    input_size = mel_spectrogram.n_mels  # จำนวน Mel bins
    hidden_size = 64  # จำนวน hidden units ของ RNN
    num_layers = 2  # จำนวน layer ของ RNN
    num_classes = 12  # จำนวนตัวโน๊ตที่เป็นไปได้ (C ถึง B♭)

    model = PitchSeqModel(input_size, hidden_size, num_layers, num_classes)
    state_dict = torch.load(r"train_seq\model2\rnn.pth")
    model.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=1024
    )

    usd = NoteSeqDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")
    sum_correct = 0
    num_test = 0
    for i in range(12):
        # get a sample from the urban sound dataset for inference
        input, target = usd[i][0], usd[i][1] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        # make an inference
        predicted, expected = detect_pitch_sequence(model, input, target)
        print(f"Predicted: '{predicted}', expected: '{expected}'")
        # if predicted == expected:
        #     sum_correct += 1
        # num_test += 1
    
    # accuracy = sum_correct/num_test *100
    # print(f'accurency : {accuracy}')

