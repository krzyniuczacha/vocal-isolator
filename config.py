# paths
PATH = "C:/Users/dtwar/Desktop/studia/projects/vocal-isolator"
DATA_PATH = PATH + "/data_stems"
DIR_NAME = "checkpoints"

# audio & STFT
fs = 44100
fft_size = 2048
win_length = 2048
hop_length = 512
window = 'hann'
center = True
pad_mode = 'constant'

# data
random_seed = 42
cut_length = 2
size_multiplier = 50

# model
first_layer_channels = 16
depth = 6
log_eps = 1e-5
log_type = 'log10'

# training
num_workers = 4
max_epochs = 200
patience = 5
early_stopping = 15
lr = 0.001
scheduler_factor = 0.5
batch_t = 16
batch_v = 32
resume_from_checkpoint = False





