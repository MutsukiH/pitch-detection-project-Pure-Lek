import torch
import torchaudio

from cnn import CNNNetwork
from mididataset import midiDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

# การเรียกใช้โมเดล จากคลิป 3 

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


def predict(model, input, target, class_mapping): # ฟังก์ชั่นการทำนาย 
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        # print(predictions)
        # print(predictions)
        # print(int(predictions[0].argmax(0)))
        # print(predictions[0].argmax(0))
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index%12]+str(target//12)
        expected = class_mapping[target%12]+str(target//12)
        # print(target)
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("./try2train_2/cnnnet.pth", map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = midiDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")


    for i in range(720):
        # get a sample from the urban sound dataset for inference
        input, target = usd[i][0], usd[i][1] # [batch size, num_channels, fr, time]
        # print(target)
        input.unsqueeze_(0) # ใช้ tensor 

        # make an inference
        predicted, expected = predict(cnn, input, target,
                                  class_mapping)
        if predicted==expected:
            print(f"Predicted: '{predicted}', expected: '{expected}'")
