import onnxruntime as ort
import numpy as np
import argparse
import os

onnx_model_names = ["MobileNetV2", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

def profile(onnx_dir, results_dir, n):
    results_dir = results_dir or '.'
    n = int(n) if n else 10
    
    for onnx_model in onnx_model_names:
        print(f"Profiling {onnx_model} models")
        for i in range(n):
            onnx_model_path = os.path.join(onnx_dir, f"{onnx_model}.onnx")
            onnx_results_path = os.path.join(results_dir, f"onnx_{onnx_model}_profiling_{i}.json")
            run_profiler(onnx_model_path, onnx_results_path)
            
            onnx_model_path_quant = os.path.join(onnx_dir, f"{onnx_model}_quant.onnx")
            onnx_results_path_quant = os.path.join(results_dir, f"onnx_{onnx_model}_quant_profiling_{i}.json")
            run_profiler(onnx_model_path_quant, onnx_results_path_quant)


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
    os.rename(profiling_file, output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models are saved', required=True)
    parser.add_argument('--results_dir', help='The directory where the ONNX profiling are to be saved')
    parser.add_argument('--num_repetitions', type=int, help='The number of repetitions for profiling')
    args = parser.parse_args()

    if not os.path.exists(args.onnx_dir):
        raise FileNotFoundError(f"Error: The input path '{args.onnx_dir}' does not exist.")

    profile(args.onnx_dir, args.results_dir, args.num_repetitions)