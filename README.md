# Setup

General description of the required setup, provided scripts and the recommended pipeline to create, train and prepare a model that can be used by a microcontroller.

---

### Setting up the environment:

Create a python virtual environment:

> python -m venv "tinyml"

Enable the environment:

> source tinyml/bin/activate

Install the required packages:

> pip install -r requirements.txt

---

### Avoiding models compatibility issues

Use the legacy version of keras to avoid version compatibility issues between the unquantized model and the quantization process.
For a single environment session:

> export TF_USE_LEGACY_KERAS=1

For a permanent change:

> echo 'export TF_USE_LEGACY_KERAS=1' >> ~/.bashrc

---

### Recommended pipeline:

Once:

1. Download the dataset - *donwload_dataset.py*
2. Download additional audio files for the dataset

For each new model:

1. Ensure the dataset has been downloaded
2. Ensure everything is setup as desired inside the various scripts (sampling rate, number of features...)
3. Train the model - *train.py*
4. Optionally test the trained model before quantizing - *test.py*
5. Do a Quantization Aware Training to prepare the model for the ESP32 - *train_QAT.py*
6. Optionally test the QAT model - *test.py*
7. Convert the QAT model to a binary encoding - *model_convert.py*
8. Place the resulting file *model.h* inside the ESP32 project folder 

---

### Available scripts

Run the scripts from inside the project folder:

> python3 ./src/<script_name>.py

The available scripts are the following:

- **download_dataset.py**: downloads the used dataset from xeno-canto inside the script's specified folder *base_folder*. The data is downloaded in an MP3 format and then converted automatically to stereo, 8kHz, WAV format
- **train.py**: performs the model definition, initialization and training on a specified bird species. It outputs both a keras model and a tensorflow lite model, respectively *model.h5* and *bird_detector.tflite*
- **train_QAT.py**: takes the previously *model.h5* trained model, and performs a *Quantize Aware Training* to prepare the model for the microcontroller. It's important due to the microcontroller being optimized to perform inference using uint8 weights, due to the lack of float32 hardware acceleration and limited memory. uint8 is 4 times smaller than float32, and due to the limited RAM this is needed. 
- **split_dataset.py**: a common script used by both *train.py* and *train_QAT.py*. It defines how and which data is being loaded, how the audio features are being extracted (Mel Spectrograms), and the relative audio settings (8kHz sampling rate, 1s sliding window frames...)
- **model_convert.py**: converts a *.tflite* model to a binary encoded model placed inside a C header file, ready for use by placing it inside the microcontroller project folder. It outputs *model.h*
- **test.py**: runs the trained model using whatever microphone is connected to the computer as the input source.

---

### Observations

- It is preferable to use layer.Flatten() when aiming for accuracy, but due to memory constraints layers.GlobalAveragePooling2D() is needed. The number of parameters that way are reduced from around 1.5m to around 200k for a two 8 and 16 units 3x3 convolution layers and a 16 units dense layer.
- Using SeparableConv2D convolution layers instead of standard convolution layers reduces the number of parameters further from around 200k to around 8k without any noticeable loss of accuracy.
- Qauntization reduces the memory usage further by a factor of 4
- Training the model with only positives or only negatives introduces severe overfitting
- The dataset is not balanced
- An 8000 elemetns frame buffer for a 1s sliding window takest too long to compute on an ESP32 
- The best training results so far are stored in train.log, with a value_accuracy of 94.5% and a value loss of 16.5%