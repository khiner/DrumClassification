import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import WaveformDataset, WaveformFeatures, AudioClassifier, LABEL_MAPPING_PATH, CHOPPED_DATASET_PATH, SLIM_METADATA_PATH

if __name__ == '__main__':
    label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH)
    chopped_df = pd.read_csv(CHOPPED_DATASET_PATH)
    slim_df = pd.read_csv(SLIM_METADATA_PATH)
    chopped_df['split'] = chopped_df.slim_id.map(slim_df['split'])
    inference_df = chopped_df[chopped_df['split'] == 'test']
    inference_dataset = WaveformDataset(inference_df, label_mapping_df)
    inference_loader = DataLoader(inference_dataset, batch_size=256, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassifier(num_classes=len(label_mapping_df), n_mel=256, n_mel_frames=32).to(device)
    waveform_features = WaveformFeatures(n_mel=256).to(device)

    checkpoint_path = 'pretrained/final.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_samples, total_correct = 0, 0
    for waveforms, labels, _ in inference_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            mel_spectrograms = waveform_features(waveforms)
            logits = model(mel_spectrograms)
            _, predicted_labels = torch.max(logits, 1)
            num_samples = len(labels)
            num_correct = torch.sum(predicted_labels == labels).item()
            total_samples += num_samples
            total_correct += num_correct
            print(f'Predicted labels: {predicted_labels}')
            print(f'True labels: {labels}')
            print(f'Batch accuracy: {num_correct / num_samples}')
            print('')

    print(f'{total_correct} of {total_samples} correct predictions for final accuracy: {total_correct / total_samples}')
