import musdb
import random
import librosa
import numpy as np
import soundfile as sf


class Dataset:

    def __init__(
        self, path, isTrain=True, cut_length=2, size_multiplier=50, fft_size=2048, hop_length=None, win_length=None
        ):
        self.mus = musdb.DB(root=path, is_wav=True, subsets='train', split="train" if isTrain else "valid")
        self.isTrain = isTrain
        self.path = path
        self.cut_length = cut_length
        self.size_multiplier = size_multiplier
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.chunks = []

        if not win_length:
            self.win_length = self.fft_size
        
        if not hop_length:
            self.hop_length = self.win_length // 4

        if not self.isTrain:
            for i in range(len(self.mus)):
                if self.mus.tracks[i].duration < self.cut_length:
                    continue

                cut_num = int(self.mus.tracks[i].duration // self.cut_length)

                for j in range(0, cut_num):
                    self.chunks.append((i, j * self.cut_length))
        

    def __len__(self):
        if self.isTrain:
            return len(self.mus) * self.size_multiplier
        else:
            return len(self.chunks)

        


    def __getitem__(self, idx):

        if self.isTrain: 
            track_idx = idx % len(self.mus)
            track = self.mus.tracks[track_idx]
            start_frame = int(random.uniform(0, track.duration - self.cut_length) * track.rate)
        else:
            track_idx, start = self.chunks[idx]
            track = self.mus.tracks[track_idx]
            start_frame = int(start * track.rate)

        
        num_frames = int(self.cut_length * track.rate)

            
        mixed = sf.read(track.path, start=start_frame, frames=num_frames, dtype='float32')[0]
        vocals = sf.read(track.sources['vocals'].path, start=start_frame, frames=num_frames, dtype='float32')[0]

        mixed_mono = np.mean(mixed, axis=1)
        vocals_mono = np.mean(vocals, axis=1)

        mixed_spec = self.to_spec(mixed_mono)
        vocals_spec = self.to_spec(vocals_mono)
            
        mask = vocals_spec / (mixed_spec + 1e-7)
        mask = np.clip(mask, 0, 1)

        mask_tensor = np.expand_dims(mask, axis=0)
        mixed_tensor = np.expand_dims(mixed_spec, axis=0)

        return mixed_tensor.astype(np.float32), mask_tensor.astype(np.float32)

    
    def to_spec(self, signal):
        spec = librosa.stft(signal, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length, window='hann', center=True, pad_mode='constant')
        spec_norm = np.abs(spec)

        return spec_norm

