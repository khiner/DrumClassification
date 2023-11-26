import torch
from torch import nn
import torchaudio.transforms as T

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512):
        super().__init__()
        self.spectral_centroid = T.SpectralCentroid(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length)
        self.num_features = 2  # Spectral Centroid and Zero Crossing Rate

    def forward(self, waveforms):
        # Spectral Centroid - Averaging across time frames
        centroid = self.spectral_centroid(waveforms).mean(dim=1)

        # Zero Crossing Rate - Counting zero crossings in each frame and then averaging
        sign_changes = torch.sign(waveforms[:, :, 1:] * waveforms[:, :, :-1])
        zero_crossings = (sign_changes == -1).float().mean(dim=2)

        # Combine the two features
        features = torch.cat((centroid, zero_crossings), dim=1)

        return features
