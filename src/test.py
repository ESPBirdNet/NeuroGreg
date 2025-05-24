'''
Load and test the model from the computer, using the integrated or headset microphone
'''

import sounddevice as sd
import numpy as np
import tensorflow.lite as tflite
import queue
import time
from termcolor import colored
import sys

from split_dataset import extract_features  

# ===== SETTINGS =====
SR = 32000              # 32 kHz
DURATION = 1            # 1 s rolling window
INTERVAL = 0.5          # run inference every 0.5 s
BLOCKSIZE = int(SR * 0.1)  # 0.1 s chunks → 3200 samples

MODEL_PATH = "./models/qat_bird_detector.tflite"
QUANTIZE = True
THRESHOLD = 0.0         # minimum top‐score to accept

species_list = [ "Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix", ]

# ===== LOAD TFLITE MODEL =====
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== ROLLING BUFFER & QUEUE =====
audio_buffer = np.zeros(SR * DURATION, dtype=np.float32)
q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].copy())


def classify_audio():
    global audio_buffer
    last_run = time.time()
    
    while True:
        try:
            chunk = q.get(timeout=1)
        except queue.Empty:
            continue

        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk

        if time.time() - last_run >= INTERVAL:
            last_run = time.time()
            feats = extract_features(audio_buffer, augment=False, quantize=QUANTIZE)
            mel_spec = feats[0].astype(np.float32)  # take first segment

            inp = np.expand_dims(mel_spec, axis=0)  # shape [1, H, W, 1]
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            
            probs = interpreter.get_tensor(output_details[0]['index'])[0]  # shape (5,)
            
            if probs.shape[0] != len(species_list):
                print(f"Warning: Model outputs {probs.shape[0]} classes, but species_list has {len(species_list)}.")
                continue  # skip this inference to avoid crashing

            top_idx = int(np.argmax(probs))
            top_score = probs[top_idx]

            # Build colored lines for each species
            lines = []
            for i, sp in enumerate(species_list):
                p = probs[i]
                # choose color by score
                if   p > 0.7: col = 'green'
                elif p > 0.4: col = 'yellow'
                else:         col = 'red'
                lines.append(colored(f"{sp:<20} {p:.2f}", col))

            # Only clear & redraw if at least one is above threshold
            if probs.max() >= THRESHOLD:
                # ANSI: clear screen and move cursor to top-left
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            # else: do nothing (keep previous display)


# === START STREAMING & CLASSIFYING ===
with sd.InputStream( callback=audio_callback, channels=1, samplerate=SR, blocksize=BLOCKSIZE ):
    print("Listening for birds... press Ctrl+C to stop.")
    try:
        classify_audio()
    except KeyboardInterrupt:
        print("\nStopped.")
