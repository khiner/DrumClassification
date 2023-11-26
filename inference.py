import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

from model import WaveformDataset, WaveformFeatures, AudioClassifier, LABEL_MAPPING_PATH, CHOPPED_DATASET_PATH, SLIM_METADATA_PATH
from logistic_regression import LogisticRegressionModel

def confusion_matrix(labels, preds):
    num_classes = len(np.unique(labels))
    matrix = np.zeros([num_classes, num_classes], dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix

def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    _, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logistic_regression', action='store_true', help='Use logistic regression model for inference')
    args = parser.parse_args()

    label_mapping_df = pd.read_csv(LABEL_MAPPING_PATH)
    chopped_df = pd.read_csv(CHOPPED_DATASET_PATH)
    slim_df = pd.read_csv(SLIM_METADATA_PATH)
    chopped_df['split'] = chopped_df.slim_id.map(slim_df['split'])
    inference_df = chopped_df[chopped_df['split'] == 'test']
    inference_dataset = WaveformDataset(inference_df, label_mapping_df)
    inference_loader = DataLoader(inference_dataset, batch_size=128, shuffle=False, pin_memory=True)

    n_mel, n_mel_frames = 256, 32
    num_classes = len(label_mapping_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform_features = WaveformFeatures(n_mel=n_mel).to(device)

    if args.logistic_regression:
        # Use the logistic regression model
        input_size = n_mel * n_mel_frames
        model = LogisticRegressionModel(input_size, num_classes).to(device)
        # Load your logistic regression model weights here if available
    else:
        # Use the checkpointed CNN model
        model = AudioClassifier(num_classes=num_classes, n_mel=n_mel, n_mel_frames=n_mel_frames).to(device)
        checkpoint_path = 'pretrained/final.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    all_predictions, all_labels = [], []
    for waveforms, labels, _ in inference_loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            mel_spectrograms = waveform_features(waveforms)
            if args.logistic_regression:
                # Reshape for logistic regression
                mel_spectrograms = mel_spectrograms.reshape(mel_spectrograms.size(0), -1)
            logits = model(mel_spectrograms)
            _, predicted_labels = torch.max(logits, 1)
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    class_names = [label_mapping_df.iloc[i]['name'] for i in range(len(label_mapping_df))]
    plot_confusion_matrix(cm, class_names)

    total_correct = np.sum(np.diag(cm))
    total_samples = np.sum(cm)
    print(f'{total_correct} of {total_samples} correct predictions for final accuracy: {total_correct / total_samples}')