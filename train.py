import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import numpy as np

EGMD_DATASET_DIR = 'dataset/e-gmd-v1.0.0' # The original E-GMD dataset, where the audio files are stored.
SLIM_METADATA_PATH = 'dataset/e-gmd-v1.0.0-slim.csv' # A curated subset of the E-GMD metadata rows.
LABEL_MAPPING_PATH = 'dataset/label_mapping.csv' # Drum instrument label IDs, names, and MIDI note numbers.
CHOPPED_DATASET_PATH = 'dataset/chopped.csv' # file_path, begin_frame, num_frames, label, slim_id
CHECKPOINTS_DIR = 'checkpoints'

SAMPLE_RATE = 44_100 # All audio files in the E-GMD dataset are 44.1 kHz.
CLIP_LENGTH_FRAMES = SAMPLE_RATE // 2 # All clips are set to a half second long.
N_MEL = 256 # Number of mel bins to use for the mel spectrogram.
N_MEL_FRAMES = 32 # Empirically determined from the mel spectrogram. TODO calculate this programatically.

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
        n_mel=N_MEL,
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
    def __init__(self, n_mel, num_classes):
        super(AudioClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

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

def run_epoch(model, loader, waveform_features, criterion, optimizer, device, is_training, print_freq=100):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss, running_loss, batch_count = 0, 0, 0
    all_losses = []
    for waveforms, labels, _ in tqdm(loader, desc="Train" if is_training else "Evaluate", leave=False):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        features = waveform_features(waveforms)
        outputs = model(features)
        loss = criterion(outputs, labels)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        batch_count += 1
        if batch_count % print_freq == 0:
            print(f'\tBatch {batch_count}: Running avg loss: {running_loss / print_freq:.4f}')
            running_loss = 0  # Reset running loss after logging

        total_loss += loss.item()
        all_losses.append(loss.item())

    return all_losses

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

# Returns a pair of (training_losses, validation_losses).
# Each element is a list of lists of losses, with the outer lists indexed by epoch and the inner lists indexed by batch.
def train(model, train_loader, val_loader, waveform_features, criterion, optimizer, num_epochs, device):
    all_train_losses, all_val_losses = [], []

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_train_losses = run_epoch(model, train_loader, waveform_features, criterion, optimizer, device, is_training=True)
        epoch_train_duration = time.time() - epoch_start_time
        epoch_val_losses = run_epoch(model, val_loader, waveform_features, criterion, optimizer, device, is_training=False)
        epoch_total_duration = time.time() - epoch_start_time
        mean_epoch_train_loss = np.mean(epoch_train_losses)
        mean_epoch_val_loss = np.mean(epoch_val_losses)
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'\tTrain loss: {mean_epoch_train_loss:.4f}')
        print(f'\tVal loss: {mean_epoch_val_loss:.4f}')
        print(f'\tTrain time: {epoch_train_duration:.3f} s')
        print(f'\tVal time: {epoch_total_duration - epoch_train_duration:.3f} s')
        print(f'\tTotal time: {epoch_total_duration:.3f} s')

        all_train_losses.append(epoch_train_losses)
        all_val_losses.append(epoch_val_losses)

        save_checkpoint(model, optimizer, epoch, filename=f'{CHECKPOINTS_DIR}/epoch_{epoch+1}.pth')

    return all_train_losses, all_val_losses

def plot_losses(all_train_losses, all_val_losses, is_final_eval=False, filename='loss_plot.png'):
    mean_epoch_train_loss = [np.mean(epoch_losses) if epoch_losses else 0 for epoch_losses in all_train_losses]
    mean_epoch_val_loss = [np.mean(epoch_losses) if epoch_losses else 0 for epoch_losses in all_val_losses]
    epochs = list(range(1, len(mean_epoch_train_loss) + 1))

    validation_label = 'Test' if is_final_eval else 'Validation'
    plt.figure(figsize=(12, 6))

    # Plot epoch losses.
    plt.plot(epochs, mean_epoch_train_loss, label='Epoch Train Loss', color='blue', marker='o')
    plt.plot(epochs, mean_epoch_val_loss, label=f'Epoch {validation_label} Loss', color='red', marker='o')

    # Plot batch losses.
    # for epoch, epoch_losses in enumerate(all_train_losses):
    #     batch_indices = np.linspace(epoch, epoch + 1, len(epoch_losses))
    #     plt.scatter(batch_indices, epoch_losses, color='lightblue', alpha=0.5, label='Batch Train Loss' if epoch == 1 else "")

    # for epoch, epoch_losses in enumerate(all_val_losses):
    #     batch_indices = np.linspace(epoch, epoch + 1, len(epoch_losses))
    #     plt.scatter(batch_indices, epoch_losses, color='lightcoral', alpha=0.5, label=f'Batch {validation_label} Loss' if epoch == 1 else "")

    plt.title(f'Training and {validation_label} Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH)
    chopped_df = pd.read_csv(CHOPPED_DATASET_PATH)
    slim_df = pd.read_csv(SLIM_METADATA_PATH)
    # Merge the 'split' column from `slim_df` into `chopped_df` based on the `slim_id` (0-indexed row of the slim metadata).
    chopped_df['split'] = chopped_df.slim_id.map(slim_df['split'])
    train_df = chopped_df[chopped_df['split'] == 'train']
    val_df = chopped_df[chopped_df['split'] == 'validation']
    test_df = chopped_df[chopped_df['split'] == 'test']

    # If `evaluate_final` is True, train on the entire training + validation set and evaluate on the test set.
    evaluate_final = True
    train_df = pd.concat([train_df, val_df]) if evaluate_final else train_df
    val_df = test_df if evaluate_final else val_df

    train_dataset = WaveformDataset(train_df, label_mapping_df)
    val_dataset = WaveformDataset(val_df, label_mapping_df)

    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(n_mel=N_MEL, num_classes=len(label_mapping_df)).to(device)
    waveform_features = WaveformFeatures(n_mel=N_MEL).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        waveform_features=waveform_features,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )

    plot_losses(train_losses, val_losses, evaluate_final)

    # Empirically determine the time dimension of the mel spectrogram.
    # waveform_features = WaveformFeatures(n_mel=N_MEL).to(device)
    # sample_waveform, _, _ = train_dataset[0]
    # sample_waveform = sample_waveform.unsqueeze(0).to(device)
    # mel_spectrogram = waveform_features(sample_waveform)
    # print(mel_spectrogram.size())  # [batch_size, 1, n_mel, time_dim]
