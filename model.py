import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

EGMD_DATASET_DIR = 'dataset/e-gmd-v1.0.0' # The original E-GMD dataset, where the audio files are stored.

SAMPLE_RATE = 44_100 # All audio files in the E-GMD dataset are 44.1 kHz.
CLIP_LENGTH_FRAMES = SAMPLE_RATE // 2 # All clips are set to a half second long.

# These are not used here, but are provided as a single source of truth for the paths to the dataset files.
SLIM_METADATA_PATH = 'dataset/e-gmd-v1.0.0-slim.csv' # A curated subset of the E-GMD metadata rows.
LABEL_MAPPING_PATH = 'dataset/label_mapping.csv' # Drum instrument label IDs, names, and MIDI note numbers.
CHOPPED_DATASET_PATH = 'dataset/chopped.csv' # file_path, begin_frame, num_frames, label, slim_id

class WaveformDataset(Dataset):
    def __init__(self, chopped_df, label_mapping_df):
        self.chopped_df = chopped_df
        self.label_mapping_df = label_mapping_df

    def __len__(self):
        return len(self.chopped_df)

    def __getitem__(self, idx):
        row = self.chopped_df.iloc[idx]
        begin_frame = row.begin_frame
        num_frames = min(row.num_frames, CLIP_LENGTH_FRAMES)
        waveform, sample_rate = torchaudio.load(f'{EGMD_DATASET_DIR}/{row.file_path}', frame_offset=begin_frame, num_frames=num_frames)
        num_frames = waveform.size(1)
        if num_frames < CLIP_LENGTH_FRAMES:
            # If the waveform is shorter than the clip length, pad it with zeros.
            waveform = F.pad(waveform, (0, CLIP_LENGTH_FRAMES - num_frames))
        return waveform, row.label, sample_rate

# Audio feature extraction pipeline.
# 1. Resample audio
# 2. Convert to power spectrogram
# 3. Convert to mel-scale
class WaveformFeatures(nn.Module):
    def __init__(
        self,
        input_freq=44_100,
        resample_freq=32_000,
        n_fft=2048,
        n_mel=256,
    ):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=n_fft // 4)
        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        spec = self.spectrogram(resampled)
        mel = self.mel_scale(spec)
        return mel

class AudioClassifier(nn.Module):
    def __init__(self, num_classes, n_mel, n_mel_frames):
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        time_dim_after_pooling = n_mel_frames // (2**3)
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
