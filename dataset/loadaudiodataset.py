import os
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sample_rate = torchaudio.load(path)
        label = 1 if 'fake' in path.lower() else 0
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label


def load_train_dataset(transform=None):
    return AudioDataset("dataset/train", transform=transform)


def load_test_dataset(transform=None):
    return AudioDataset("dataset/test", transform=transform)


def load_execution_dataset(transform=None):
    return AudioDataset("dataset/exec", transform=transform)