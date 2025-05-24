import os
import numpy as np
import tensorflow as tf
from split_dataset import split
from split_dataset import OUTPUT_DIR

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow_model_optimization as tfmot

CACHE_FILE = os.path.join(OUTPUT_DIR, "cached_dataset.npz")

LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32

species_list = ["Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix"] #, "Unknown"] # unknown being random noises

class_weight = {
    0: 1.0,
    1: 2.0,
    2: 2.0,
    3: 1.0,
    4: 1.0
}


# Loss function
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Focal modulation
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1. - y_pred, gamma)
        loss = weight * cross_entropy
        
        # Sum over classes, average over batch
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the CNN model
INPUT_SHAPE = (128, 32, 1)
NUM_SPECIES = len(species_list)  # Assuming species_list is defined

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=INPUT_SHAPE),

    # Block 0
    tf.keras.layers.DepthwiseConv2D((3, 5), padding='same', activation='relu'),     # focus more on frequencies
    tf.keras.layers.SeparableConv2D(4, 1, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),

    # Block 1
    tf.keras.layers.DepthwiseConv2D((5, 3), padding='same', activation='relu'),     # focus more on time
    tf.keras.layers.SeparableConv2D(8, 1, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),

    # Block 2 (deeper)
    tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    tf.keras.layers.SeparableConv2D(16, 1, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),

    # Block 3 (deeper)
    tf.keras.layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    tf.keras.layers.SeparableConv2D(32, 1, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    # Head 0
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    # Final classification
    tf.keras.layers.Dense(NUM_SPECIES, activation='softmax')
])

# Load and split the dataset, avail of caches - takes a long time
if os.path.exists(CACHE_FILE):
    print("Loading cached dataset...")
    data = np.load(CACHE_FILE)
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
else:
    print("Preprocessing dataset...")
    print(species_list)
    X_train, X_test, y_train, y_test = split()
    
    print("Saving dataset to cache...")
    np.savez_compressed(CACHE_FILE,  X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# Drop randomly 25% of samples if not fitting in GPU mem 
rng = np.random.default_rng(42)
n = X_train.shape[0]
keep_n = int(n * 0.75)
idx = np.sort(rng.choice(n, keep_n, replace=False))
X_train_reduced, y_train_reduced = X_train[idx], y_train[idx]

drop_n = n - keep_n
mask = np.ones(n, dtype=bool)
mask[rng.choice(n, drop_n, replace=False)] = False
X_train, y_train = X_train[mask], y_train[mask]

# One hot encoding
y_train = to_categorical(y_train, num_classes=len(species_list))
y_test  = to_categorical(y_test,  num_classes=len(species_list))

# Define cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay( initial_learning_rate=LEARNING_RATE, decay_steps=len(X_train) * BATCH_SIZE )

# Compile model
loss_fn = categorical_focal_loss(alpha=0.25, gamma=2.0)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss_fn, metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=25, batch_size=128, validation_data=(X_test, y_test), verbose = 1, callbacks=[early_stopping], class_weight=class_weight)

# Save Model
model_path = os.path.join(os.getcwd(), OUTPUT_DIR, "model.h5")
model.save(model_path)

print(f"Model saved at {model_path}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = os.path.join(os.getcwd(), OUTPUT_DIR, "bird_detector.tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("Model saved as bird_detector.tflite to " + os.path.join(os.getcwd(), OUTPUT_DIR))