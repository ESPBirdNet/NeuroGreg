import os
import tensorflow as tf
import tensorflow.lite

OUTPUT_DIR = "./models"


interpreter = tf.lite.Interpreter(model_path=os.path.join(os.getcwd(), OUTPUT_DIR,"qat_bird_detector.tflite"))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print("Input dtype:", inp["dtype"], "Quant:", inp["quantization_parameters"])
print("Output dtype:", out["dtype"], "Quant:", out["quantization_parameters"])
