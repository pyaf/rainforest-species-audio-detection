import os
from resnest.torch import resnest50
import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.transform import resize
import csv

fmin =  84
fmax = 15056
fft = 2048
hop = 512
sr = 48000
length = 10 * sr
num_birds = 24

def load_test_file(f):
    wav, sr = librosa.load('data/test/' + f, sr=None)

    # Split for enough segments to not miss anything
    segments = len(wav) / length
    segments = int(np.ceil(segments))

    mel_array = []

    for i in range(0, segments):
        # Last segment going from the end
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]

        # Same mel spectrogram as before
        mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
        mel_spec = resize(mel_spec, (224, 400))

        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)

        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))

        mel_array.append(mel_spec)

    return mel_array


# Loading model back
model = resnest50(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, num_birds)
)

#model = torch.load()
ckpt_path = 'weights/test/model.pth'
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
import pdb; pdb.set_trace()
model.load_state_dict(state["state_dict"])

model.eval()

# Scoring does not like many files:(
# if save_to_disk == 0:
#     for f in os.listdir('/kaggle/working/'):
#         os.remove('/kaggle/working/' + f)

if torch.cuda.is_available():
    model.cuda()

# Prediction loop
print('Starting prediction loop')
with open('submission.csv', 'w', newline='') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',')
    submission_writer.writerow(['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                               's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])

    test_files = os.listdir('data/test/')
    print(len(test_files))

    # Every test file is split on several chunks and prediction is made for each chunk
    for i in tqdm(range(0, len(test_files))):
        data = load_test_file(test_files[i])
        data = torch.tensor(data)
        data = data.float()
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)

        # Taking max prediction from all slices per bird species
        # Usually you want Sigmoid layer here to convert output to probabilities
        # In this competition only relative ranking matters, and not the exact value of prediction, so we can use it directly
        maxed_output = torch.max(output, dim=0)[0]
        maxed_output = maxed_output.cpu().detach()

        file_id = str.split(test_files[i], '.')[0]
        write_array = [file_id]

        for out in maxed_output:
            write_array.append(out.item())

        submission_writer.writerow(write_array)

        if i % 100 == 0 and i > 0:
            print('Predicted for ' + str(i) + ' of ' + str(len(test_files) + 1) + ' files')

print('Submission generated')
