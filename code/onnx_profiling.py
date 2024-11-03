import onnxruntime as ort
import numpy as np
import argparse
import os

tflite_model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
onnx_model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
pytorch_model_names = ["mobilenet_v2", "inception_v3", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]

def profile(tflite_dir, onnx_dir, pytorch_dir):
    if tflite_dir:
        for tflite_model in tflite_model_names:
            print(f"Profiling TFlite Model: {tflite_model}")
            tflite_model_path = os.path.join(tflite_dir, f"tflite_{tflite_model}.onnx")
            run_profiler(tflite_model_path, f"tflite_{tflite_model}_profiling.json")
            
            print(f"Profiling Quantized TFlite Model: {tflite_model}")
            tflite_model_path_quant = os.path.join(tflite_dir, f"tflite_{tflite_model}_quant.onnx")
            run_profiler(tflite_model_path_quant, f"tflite_{tflite_model}_quant_profiling.json")
    if onnx_dir:
        for onnx_model in onnx_model_names:
            print(f"Profiling ONNX Model: {onnx_model}")
            onnx_model_path = os.path.join(onnx_dir, f"{onnx_model}.onnx")
            run_profiler(onnx_model_path, f"onnx_{onnx_model}_profiling.json")
            
            print(f"Profiling Quantized ONNX Model: {onnx_model}")
            onnx_model_path_quant = os.path.join(onnx_dir, f"{onnx_model}_quant.onnx")
            run_profiler(onnx_model_path_quant, f"onnx_{onnx_model}_quant_profiling.json")
    if pytorch_dir:
        for pytorch_model in pytorch_model_names:
            print(f"Profiling PyTorch Model: {pytorch_model}")
            pytorch_model_path = os.path.join(pytorch_dir, f"pytorch_{pytorch_model}.onnx")
            run_profiler(pytorch_model_path, f"pytorch_{pytorch_model}_profiling.json")
            
            print(f"Profiling Quantized PyTorch Model: {pytorch_model}")
            pytorch_model_path_quant = os.path.join(pytorch_dir, f"pytorch_{pytorch_model}_quant.onnx")
            run_profiler(pytorch_model_path_quant, f"pytorch_{pytorch_model}_quant_profiling.json")

def run_profiler(model_path, output_name):
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True  # Enable profiling

    # Initialize the ONNX runtime session with profiling enabled
    session = ort.InferenceSession(model_path, sess_options=session_options)

    # Prepare a sample input
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]  # Replace None with 1
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    # Retrieve the profiling file path and rename it
    profiling_file = session.end_profiling()
    new_profiling_file = os.path.join(os.path.dirname(profiling_file), output_name)
    os.rename(profiling_file, new_profiling_file)
    print(f"Profiling data saved to: {new_profiling_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', help='The directory where the ONNX models converted from TFLite are saved')
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models are saved')
    parser.add_argument('--pytorch_dir', help='The directory where the ONNX models converted from PyTorch are saved')
    args = parser.parse_args()

    profile(args.tflite_dir, args.onnx_dir, args.pytorch_dir)
