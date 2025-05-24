'''
Post quantization script
Gets in input a tensorflow model, and outputs the quantized version
It's strongly advised to not use this script
Use train_QAT.py instead
'''

import os
import random
import librosa
import numpy as np
import tensorflow as tf

OUTPUT_DIR = "./models"
DATA_DIR = "./sounds"
SPECIES = "Larus_michahellis"  # Replace with your species of interest

# Settings
SR = 16000       # MIC sample rate
WINDOW_SIZE = 1  # seconds
FRAME_LENGTH = SR * WINDOW_SIZE
N_MFCC = 12      # feature count -> check sound classification methods

# Load the model
model_path = os.path.join(os.getcwd(), OUTPUT_DIR, "model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Pretrained model not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Summarize the model
model.summary()

def extract_features(file_path):
    """ Extract Mel spectrogram features from a file. """
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    mel_features = []
    for i in range(0, len(y) - FRAME_LENGTH, FRAME_LENGTH):
        mel_spec = librosa.feature.melspectrogram(y=y[i:i+FRAME_LENGTH], sr=SR, n_mels=N_MFCC)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize the Mel spectrogram to have zero mean and unit variance
        mean = np.mean(mel_spec)
        std = np.std(mel_spec)

        if std > 0:
            mel_spec = (mel_spec - mean) / std
        else:
            mel_spec = mel_spec - mean  # Subtract mean but don't divide by zero
        
        mel_features.append(np.expand_dims(mel_spec, axis=-1))

    return mel_features

def representative_dataset():
    """ Prepare a representative dataset for quantization. """
    species_dir = os.path.join(DATA_DIR, SPECIES)
    all_files = [os.path.join(species_dir, f) for f in os.listdir(species_dir) if f.endswith(".wav")]
    
    # Select 10 random files
    selected_files = random.sample(all_files, 10)
    
    for file_path in selected_files:
        features = extract_features(file_path)
        for feature in features:
            # Add an extra dimension for batch_size (shape should be [1, height, width, channels])
            yield [feature.astype(np.float32)]

# Convert to TFLite with post-training quantization to uint8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]  # Ensure uint8 quantization
converter.representative_dataset = representative_dataset  # Set the representative dataset
tflite_model = converter.convert()

# Save the TFLite model
with open("model_quantized_uint8.tflite", "wb") as f:
    f.write(tflite_model)

print("Model saved as model_quantized_uint8.tflite")
