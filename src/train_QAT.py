'''
Quantization Aware Training
Output a fully quantized model
Better then simple quantization
'''

import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from split_dataset import split, OUTPUT_DIR
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score

CACHE_FILE = os.path.join(OUTPUT_DIR, "QAT_cached_dataset.npz")
species_list = [ "Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix" ]
NUM_SPECIES = len(species_list)

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32


# Loss function definition
def categorical_focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1. - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return focal_loss


# Custom QuantizeConfig: skip BatchNorm, let default transforms fold them
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer): return []
    def get_activations_and_quantizers(self, layer): return []
    def set_quantize_weights(self, layer, quantize_weights): pass
    def set_quantize_activations(self, layer, quantize_activations): pass
    def get_output_quantizers(self, layer): return []
    def get_config(self): return {}


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original float32 model
dl_model = tf.keras.models.load_model(
    os.path.join(OUTPUT_DIR, "model.h5"),
    custom_objects={'focal_loss': categorical_focal_loss}
)

# Rename layers to avoid duplicate names
def rename_layer(layer):
    cfg = layer.get_config()
    cfg['name'] = layer.name + '_orig'
    new_layer = layer.__class__.from_config(cfg)
    return new_layer

model = clone_model(
    dl_model,
    clone_function=rename_layer
)
model.set_weights(dl_model.get_weights())

# Annotate only BatchNormalization layers
def clone_and_annotate(layer):
    # Annotate Conv2D, DepthwiseConv2D, Dense, ReLU, etc.
    if isinstance(layer, (tf.keras.layers.Conv2D,
                          tf.keras.layers.DepthwiseConv2D,
                          tf.keras.layers.Dense,
                          tf.keras.layers.ReLU,
                          tf.keras.layers.Activation,
                          tf.keras.layers.BatchNormalization)):
        return quantize_annotate_layer(layer, NoOpQuantizeConfig())
    return layer

with tfmot.quantization.keras.quantize_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
    annotated_model = clone_model(
        model,
        clone_function=clone_and_annotate
    )
    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# Load or build dataset
if os.path.exists(CACHE_FILE):
    print("Loading cached dataset...")
    data = np.load(CACHE_FILE)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
else:
    print("Preprocessing dataset...")
    X_train, X_test, y_train, y_test = split(quantize = True)
    np.savez_compressed(CACHE_FILE,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=NUM_SPECIES)
y_test = to_categorical(y_test, num_classes=NUM_SPECIES)

# Define cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay( initial_learning_rate=LEARNING_RATE, decay_steps=len(X_train) * BATCH_SIZE )

# Compile QAT model
loss_fn = categorical_focal_loss(alpha=0.25, gamma=2.0)
qat_model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss="categorical_crossentropy", metrics=['accuracy'] )
qat_model.summary()

# Fine-tune with QAT
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
qat_model.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[early_stopping] )

# Save QAT model
qat_path = os.path.join(OUTPUT_DIR, "qat_model.h5")
qat_model.save(qat_path)
print(f"QAT model saved at {qat_path}")

# ========================= TFLite full-integer conversion ========================= #
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Representative data gen for calibration
def representative_data_gen():
    for spect in X_train[:500]:
        yield [np.expand_dims(spect, axis=0).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.experimental_new_quantizer = True  # Better full-integer conversion

# Convert and save
tflite_model = converter.convert()
uint8_path = os.path.join(OUTPUT_DIR, "qat_bird_detector.tflite")
with open(uint8_path, 'wb') as f:
    f.write(tflite_model)
print(f"Fully integer-quantized model saved at {uint8_path}")

# ========================= EVALUATION ON TEST SET ========================= #
# Predict and compute metrics
y_pred_probs = qat_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report (includes precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=species_list))

# Overall metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted Precision: {precision:.4f}")
