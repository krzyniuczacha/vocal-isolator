import musdb
import random
import librosa
import numpy as np


class Dataset:

    def __init__(
        self, path, isTrain=True, cut_length=2, size_multiplier=50, fft_size=2048, hop_size=None, window_size=None
        ):
        self.mus = musdb.DB(root=path, is_wav=True, subsets='train', split="train" if isTrain else "valid")
        self.isTrain = isTrain
        self.path = path
        self.cut_length = cut_length
        self.size_multiplier = size_multiplier
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size

        if not window_size:
            self.window_size = self.fft_size
        
        if not hop_size:
            self.hop_size = self.window_size // 4
        
    def __len__(self):
        return len(self.mus) * self.size_multiplier


    def __getitem__(self, idx):
        idx = idx % len(self.mus)
        track = self.mus.tracks[idx]
            
        track.chunk_duration = self.cut_length 
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)

        mixed= track.audio
        vocals = track.targets['vocals'].audio

        mixed_mono = np.mean(mixed, axis=1)
        vocals_mono = np.mean(vocals, axis=1)

        mixed_spec = self.to_spec(mixed_mono)
        vocals_spec = self.to_spec(vocals_mono)
            
        mask = vocals_spec / (mixed_spec + 1e-7)
        mask = np.clip(mask, 0, 1)

        mask_tensor = np.expand_dims(mask, axis=0)
        mixed_tensor = np.expand_dims(mixed_spec, axis=0)

        return mixed_tensor, mask_tensor

    
    def to_spec(self, signal):
        spec = librosa.stft(signal, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.window_size)
        spec_norm = np.abs(spec)

        return spec_norm

            
        