import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import wavfile

EGMD_DATASET_DIR = 'dataset/e-gmd-v1.0.0' # The original E-GMD dataset, where the audio files are stored.
SLIM_METADATA_PATH = 'dataset/e-gmd-v1.0.0-slim.csv' # A curated subset of the E-GMD metadata rows.
LABEL_MAPPING_PATH = 'dataset/label_mapping.csv' # Drum instrument label IDs, names, and MIDI note numbers.
CHOPPED_DATASET_PATH = 'dataset/chopped.csv' # file_path, begin_frame, num_frames, label, slim_id

SAMPLE_RATE = 44_100 # All audio files in the E-GMD dataset are 44.1 kHz.
CLIP_LENGTH_FRAMES = 1 * SAMPLE_RATE # All clips are set to 1 second long.
N_MEL = 128 # Number of mel bins to use for the mel spectrogram.
N_MEL_FRAMES = 32 # Empirically determined from the mel spectrogram. TODO calculate this programatically.

class WaveformDataset(Dataset):
    def __init__(self, chopped_df, label_mapping_df):
        self.chopped_df = chopped_df
        self.label_mapping_df = label_mapping_df

    def __len__(self):
        return len(self.chopped_df)

    def __getitem__(self, idx):
        row = self.chopped_df.iloc[idx]
        sample_rate, data_int16 = wavfile.read(f'{EGMD_DATASET_DIR}/{row.file_path}')
        begin_frame = row.begin_frame
        num_frames = min(row.num_frames, len(data_int16) - begin_frame)
        waveform_np = data_int16[begin_frame:begin_frame+num_frames] / (2**15) # Convert from int16 to float32
        # If the clip is shorter than the clip length, pad it with zeros.
        # If the clip is longer than the clip length, truncate it.
        if num_frames < CLIP_LENGTH_FRAMES:
            waveform_np = np.pad(waveform_np, (0, CLIP_LENGTH_FRAMES - num_frames))
        elif num_frames > CLIP_LENGTH_FRAMES:
            waveform_np = waveform_np[:CLIP_LENGTH_FRAMES]
        waveform = torch.from_numpy(waveform_np).float()
        # waveform, sample_rate = torchaudio.load(file_path, frame_offset=begin_frame, num_frames=num_frames)
        # assert(sample_rate == 44_100)
        # assert(waveform.size(0) == CLIP_LENGTH_FRAMES)
        return waveform, row.label, sample_rate

# Audio feature extraction pipeline.
# 1. Resample audio
# 2. Convert to power spectrogram
# 3. Convert to mel-scale
class WaveformFeatures(nn.Module):
    def __init__(
        self,
        input_freq=44_100,
        resample_freq=16_000,
        n_fft=1024,
        n_mel=N_MEL,
    ):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spectrogram = T.Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        spec = self.spectrogram(resampled)
        mel = self.mel_scale(spec)
        return mel

class AudioClassifier(nn.Module):
    def __init__(self, n_mel, num_classes):
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Calculate the flattened size for the fully connected layer
        time_dim_after_pooling = N_MEL_FRAMES // (2**3)
        conv_output_size = n_mel // (2**3)
        fc1_input_size = 64 * conv_output_size * time_dim_after_pooling
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten for the fully connected layers.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, train_loader, waveform_features, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for waveforms, labels, _ in train_loader:
            waveforms = torch.stack([waveform.unsqueeze(0) for waveform in waveforms]).to(device)
            labels = labels.to(device)
            features = waveform_features(waveforms)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH)
    chopped_df = pd.read_csv(CHOPPED_DATASET_PATH)
    dataset = WaveformDataset(chopped_df=chopped_df, label_mapping_df=label_mapping_df)
    device = torch.device('cpu')
    model = AudioClassifier(n_mel=N_MEL, num_classes=len(label_mapping_df)).to(device)

    train(
        model=model,
        train_loader=DataLoader(dataset, batch_size=32, shuffle=True),
        waveform_features=WaveformFeatures(n_mel=N_MEL).to(device),
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        num_epochs=10
    )

    # Empirically determine the time dimension of the mel spectrogram.
    # waveform_features = WaveformFeatures(n_mel=N_MEL).to(device)
    # sample_waveform, _, _ = dataset[0]
    # sample_waveform = sample_waveform.unsqueeze(0).to(device)
    # mel_spectrogram = waveform_features(sample_waveform)
    # print(mel_spectrogram.size())  # [batch_size, 1, n_mel, time_dim]
