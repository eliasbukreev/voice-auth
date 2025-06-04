import os
import librosa
import torch
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, file_list, labels, n_mfcc=20, max_len=128):
        self.file_list = file_list
        self.labels = labels
        self.n_mfcc = n_mfcc
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]

        return torch.tensor(mfcc).float(), torch.tensor(label).long()