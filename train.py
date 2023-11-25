import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import numpy as np

from model import WaveformDataset, WaveformFeatures, AudioClassifier, LABEL_MAPPING_PATH, CHOPPED_DATASET_PATH, SLIM_METADATA_PATH

CHECKPOINTS_DIR = 'checkpoints'

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

    n_mel = 256
    n_mel_frames = 32 # Empirically determined from the mel spectrogram. TODO calculate this programatically.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(num_classes=len(label_mapping_df), n_mel=n_mel, n_mel_frames=n_mel_frames).to(device)
    waveform_features = WaveformFeatures(n_mel=n_mel).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        waveform_features=waveform_features,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=16,
        device=device
    )

    plot_losses(train_losses, val_losses, evaluate_final)

    # Empirically determine the time dimension of the mel spectrogram.
    # waveform_features = WaveformFeatures(n_mel=N_MEL).to(device)
    # sample_waveform, _, _ = train_dataset[0]
    # sample_waveform = sample_waveform.unsqueeze(0).to(device)
    # mel_spectrogram = waveform_features(sample_waveform)
    # print(mel_spectrogram.size())  # [batch_size, 1, n_mel, time_dim]
