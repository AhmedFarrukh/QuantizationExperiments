import numpy as np
import time
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import argparse

# Generate a fake dataset
def generate_fake_dataset(num_samples=100, input_shape=(1, 3, 224, 224)):
    dataset = [np.random.rand(*input_shape).astype(np.float32) for _ in range(num_samples)]
    return dataset

def run_inference_onnx(model_path, dataset):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    for data in dataset:
        session.run(None, {input_name: data})
    end_time = time.time()

    return end_time - start_time

def run_inference_tflite(model_path, dataset):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    dataset = [np.transpose(data, (0, 2, 3, 1)) for data in dataset]

    start_time = time.time()
    for data in dataset:
        interpreter.set_tensor(input_index, data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)
    end_time = time.time()

    return end_time - start_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_model', help='The path to the tflite model', required=True)
    parser.add_argument('--onnx_model', help='The path to the onnx model', required=True)
    args = parser.parse_args()
    # Paths to local models
    onnx_model_path = args.tflite_model  # Path to ONNX model
    tflite_model_path = args.onnx_model  # Path to TFLite model

    # Generate fake dataset
    print("Generating fake dataset...")
    dataset = generate_fake_dataset(num_samples=1000)

    # Run inference and measure time for ONNX
    print("Running inference with ONNX...")
    onnx_time = run_inference_onnx(onnx_model_path, dataset)
    print(f"ONNX Inference Time: {onnx_time:.4f} ms")

    # Run inference and measure time for TFLite
    print("Running inference with TFLite...")
    tflite_time = run_inference_tflite(tflite_model_path, dataset)
    print(f"TFLite Inference Time: {tflite_time:.4f} ms")

    # Comparison
    print("\n--- Inference Time Comparison ---")
    print(f"ONNX Total Time: {onnx_time:.4f} ms")
    print(f"TFLite Total Time: {tflite_time:.4f} ms")
