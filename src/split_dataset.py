'''
Helper script for dataset split and feature extraction
Used by train.py and train_QAT.py
'''

import os
import random
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

species_list = ["Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix"] # , "Unknown"] #, "music"]

# Dataset
DATA_DIR = "./sounds_32khz"
OUTPUT_DIR = "./models"

# Settings
TRAIN_SPLIT = 0.8  # train split percent -> how much of the dataset should be used for training
NUM_WORKERS = 10   # number of threads for feature extraction

SR = 32000         # or whatever sampling rate is being used on the microcontroller
FFT_SIZE = 1024
NUM_BANDS = 128
NUM_FRAMES = 32 
SEGMENT_SIZE = SR  # SEGMENT_SIZE = (NUM_FRAMES // 2 + 1) * HOP_SIZE  # 16384
HOP_SIZE = (SEGMENT_SIZE - FFT_SIZE) // (NUM_FRAMES - 1)
STEP_SIZE = SR // 4  # advance 0.5 s between features

# Decibel clipping ranges for quantized version
MIN_DB = 30
MAX_DB = 158


# Save random extracted features to check that everything is alright
def save_spectrogram_image(spec, source, output_dir="extracted_spectrograms"):
    # with 0.001% chance, save as image
    if random.random() < 0.001:
        # ensure output folder exists
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(source, str):
            folder_name = os.path.basename(os.path.dirname(source)) or "root"
            base_name = os.path.splitext(os.path.basename(source))[0]
        else:
            folder_name = "array"
            base_name = "spec"
        filename = f"{folder_name}_{base_name}.png"
        path = os.path.join(output_dir, filename)
        plt.imsave(path, spec.squeeze(), cmap='viridis')


# Audio augmentation function
def augment_audio(y, sr):
    # Time Stretching
    if random.random() > 0.5:
        y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    
    # Pitch Shift
    if random.random() > 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-3, 3))
    
    # Add Noise
    if random.random() > 0.5:
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise
    
    return y


# time/freq masking or frame/band occlusion - random chance
def post_augment_spec(spec):
    # SpecAugment: time masking
    t = spec.shape[1]
    tm = random.randint(1, max(1, t // 10))
    t0 = random.randint(0, t - tm)
    spec[:, t0:t0+tm] = 0
    # SpecAugment: frequency masking
    f = spec.shape[0]
    fm = random.randint(1, max(1, f // 10))
    f0 = random.randint(0, f - fm)
    spec[f0:f0+fm, :] = 0

    # Frame occlusion: drop entire random frame
    if random.random() < 0.001:
        idx = random.randrange(NUM_FRAMES)
        spec[:, idx] = 0
    # Band occlusion: drop entire random band
    if random.random() < 0.001:
        idx = random.randrange(NUM_BANDS)
        spec[idx, :] = 0

    return spec


# Compute spectrograms as on the microcontroller
def extract_features(source, augment=False, quantize=False):
    # Load or assign
    if isinstance(source, str):
        y, _ = librosa.load(source, sr=SR, mono=True)
    else:
        y = source.astype(np.float32)

    # Optional waveform augment
    if augment:
        y = augment_audio(y, SR)

    # Pad to always take exactly SEGMENT_SIZE samples
    if len(y) < SEGMENT_SIZE:
        y = np.pad(y, (0, SEGMENT_SIZE - len(y)))

    # Precompute hann window
    hann = np.hanning(FFT_SIZE)
    bins_per_band = (FFT_SIZE // 2) // NUM_BANDS

    features = []

    offset_db = 20*np.log10(2**15)    # ≈ 90.309 dB - based on INMP441 setup -> 16bit sampling resolution

    # Slide 1 s windows every 0.5 s - outputs 2 spectrograms per second
    for start in range(0, len(y) - SEGMENT_SIZE + 1, STEP_SIZE):
        segment = y[start : start + SEGMENT_SIZE]
        # ensure C‐contiguous memory for stride
        segment = np.ascontiguousarray(segment)

        #Build a 2D view of all frames in one go
        frame_stride = segment.strides[0]
        frames = as_strided( segment, shape=(NUM_FRAMES, FFT_SIZE), strides=(HOP_SIZE * frame_stride, frame_stride), writeable=False )

        # Window + FFT in batch
        windowed = frames * hann                                # (NUM_FRAMES, FFT_SIZE)
        fft_res   = np.fft.rfft(windowed, n=FFT_SIZE, axis=1)   # (NUM_FRAMES, FFT_SIZE//2+1)
        power     = np.abs(fft_res) ** 2                        # same shape

        # Aggregate into bands
        usable = power[:, : bins_per_band * NUM_BANDS]                      # drop extra
        bands  = usable.reshape(NUM_FRAMES, NUM_BANDS, bins_per_band)       # (F, B, P)
        spec   = (10.0 * np.log10(np.sum(bands, axis=2) + 1e-12)).T         # (B, F)
        if quantize:
            spec  += offset_db                                  # ≈ +90.31 dB offset - match with arduino

        # (Optional) spec‐level augment
        if augment:
            spec = post_augment_spec(spec)

        # "Quantize" spectrograms on request, e.g. for train_QAT 
        if quantize:
            spec_uint8 = np.clip(spec, MIN_DB, MAX_DB)
            spec_uint8 = np.round((spec - MIN_DB) * (256 / (MAX_DB - MIN_DB)))
            spec_uint8 = np.clip(spec, 0, 255).astype(np.uint8)
            spec = spec_uint8

        # Save spectrogram image ar random to allow for inspection
        save_spectrogram_image(spec, source)

        features.append(spec[..., np.newaxis])  # (B, F, 1)

    return features


def split(quantize=False):
    # Map species to class labels
    species_labels = {species.replace(" ", "_"): i for i, species in enumerate(species_list)}

    X_train, y_train = [], []
    X_test, y_test = [], []

    base_path = os.path.abspath(os.path.join(os.getcwd(), DATA_DIR))

    for species_name, label in species_labels.items():
        species_path = os.path.join(base_path, species_name)
        all_files = [os.path.join(species_path, f)
                     for f in os.listdir(species_path) if f.endswith(".wav")]

        print(f"#Files for {species_name}: {len(all_files)}")
        random.shuffle(all_files)
        num_train = int(len(all_files) * TRAIN_SPLIT)
        train_files = all_files[:num_train]
        test_files  = all_files[num_train:]

        # Parallel feature extraction for training files
        print(f"Processing training files for {species_name} with {NUM_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            results = list(tqdm(exe.map(lambda f: extract_features(f, augment=True, quantize=quantize),
                                        train_files), total=len(train_files)))
        for feats in results:
            for feat in feats:
                X_train.append(feat)
                y_train.append(label)

        # Parallel feature extraction for test files
        print(f"Processing test files for {species_name} with {NUM_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
            results = list(tqdm(exe.map(lambda f: extract_features(f, augment=False, quantize=quantize),
                                        test_files), total=len(test_files)))
        for feats in results:
            for feat in feats:
                X_test.append(feat)
                y_test.append(label)

    # Convert to numpy arrays
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    return X_train, X_test, y_train, y_test
