import os
import argparse

model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

def convert(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: The input path '{input_path}' does not exist.")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output directory '{output_path}'.")

    for model_name in model_names:
        # Original Model
        tflite_model_path = os.path.join(input_path, f"{model_name}.tflite")
        onnx_model_path = os.path.join(output_path, f"tflite_{model_name}.onnx")
        exit_code = os.system(f"python3 -m tf2onnx.convert --opset 17 --tflite {tflite_model_path} --output {onnx_model_path}")
        
        if exit_code != 0:
            print(f"Error: Conversion failed for model {model_name}.")
            continue

        print(f"Conversion successful for model {model_name}.")

        # Quantized Model
        tflite_model_path_quant = os.path.join(input_path, f"{model_name}_quant.tflite")
        onnx_model_path_quant = os.path.join(output_path, f"tflite_{model_name}_quant.onnx")
        exit_code_quant = os.system(f"python3 -m tf2onnx.convert --opset 17 --tflite {tflite_model_path_quant} --output {onnx_model_path_quant}")
        
        if exit_code_quant != 0:
            print(f"Error: Conversion failed for quantized model {model_name}_quant")
        else:
            print(f"Quantized conversion successful for model {model_name}_quant.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', help='The directory where the tflite models are saved', required=True)
    parser.add_argument('--onnx_dir', help='The directory where the onnx models should be saved', required=True)
    args = parser.parse_args()

    convert(args.tflite_dir, args.onnx_dir)
