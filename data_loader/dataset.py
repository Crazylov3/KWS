import os
import random
import torch
import pandas as pd
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

# log mel - 30ms win 10ms shift 1/
class KWSDataset(Dataset):
    def __init__(self, root: str, csv_file: str, sample_rate=8000, win_length=256, n_mels=40, hop_length=80,
                 use_augment=True):
        super().__init__()
        self.root = root
        self.df = pd.read_csv(csv_file)
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.use_augment = use_augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        audio_path = os.path.join(self.root, self.df.iloc[n, 0])
        label = self.df.iloc[n, 1]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0:1, :]
        waveform = self._shift_augment(waveform) if self.use_augment else waveform
        log_mel = self.prepare_wav(waveform, sample_rate, self.sample_rate)
        log_mel = self._spec_augment(log_mel) if self.use_augment else log_mel
        return log_mel, label

    @staticmethod
    def _spec_augment(spectrum, p=0.8):
        f_masker = transforms.FrequencyMasking(freq_mask_param=20)
        t_masker = transforms.TimeMasking(time_mask_param=20)
        t_stretch = transforms.TimeStretch()
        if random.uniform(0, 1) > p:
            spectrum = f_masker(spectrum)
        if random.uniform(0, 1) > p:
            spectrum = t_masker(spectrum)
        # if random.uniform(0, 1) > p:
        #     spectrum = t_stretch(spectrum, random.uniform(0.8, 1.2))

        return spectrum

    @staticmethod
    def _shift_augment(waveform):
        shift = random.randint(-100, 100)
        waveform = torch.roll(waveform, shift)
        if shift > 0:
            waveform[0][:shift] = 0
        elif shift < 0:
            waveform[0][shift:] = 0
        return waveform

    @staticmethod
    def prepare_wav(waveform, sample_rate, SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        to_mel = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=256, f_max=8000, n_mels=40, win_length=256,
                                           hop_length=80)
        log_mel = (to_mel(waveform) + 1e-9).log2()
        return log_mel


#
# def pad_sequence(batch):
#     batch = [item.permute(2, 1, 0) for item in batch]
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
#     return batch.permute(0, 3, 2, 1)
#
#
# def collate_fn(batch):
#     tensors, targets = [], []
#     for log_mel, label in batch:
#         tensors.append(log_mel)
#
#     tensors = pad_sequence(tensors)
#     targets = torch.LongTensor(targets)
#
#     return tensors, targets


if __name__ == "__main__":
    root = "../data/train_sample"
    csv_file = "../data/test.txt"
    dataset = KWSDataset(root, csv_file)
    mel, labels = dataset[0]
    assert mel.shape == (1, 40, 151)
