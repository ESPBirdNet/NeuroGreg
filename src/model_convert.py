'''
Use to convert a TFLite model into a binary model wrapped inside a C header file
Useful to copy paste the model directly in the microcontroller project folder
'''

import os

# Input & Output file paths
MODEL_PATH = "./models/qat_bird_detector.tflite"
OUTPUT_PATH = "model.h"

# Read the .tflite model
with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

# Convert to a C++ byte array
cpp_array = ", ".join(f"0x{b:02X}" for b in model_data)
model_len = len(model_data)

# Create the C++ file
cpp_code = f"""\
#ifndef TENSORFLOW_LITE_MICRO_MODEL_H_
#define TENSORFLOW_LITE_MICRO_MODEL_H_

extern const unsigned char model_data[];
extern const int model_data_len;

#endif  // TENSORFLOW_LITE_MICRO_MODEL_H_

const int model_data_len = {model_len};

alignas(8) const unsigned char model_data[] = 
{{ 
{cpp_array}
}};
"""

# Save the output file
with open(OUTPUT_PATH, "w") as f:
    f.write(cpp_code)

print(f"Model converted successfully! Saved as {OUTPUT_PATH}")
