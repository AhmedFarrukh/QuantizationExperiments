import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import pathlib
import argparse
import requests

modelNames = {"MobileNetV2": "https://github.com/onnx/models/raw/main/Computer_Vision/mobilenetv2_100_Opset16_timm/mobilenetv2_100_Opset16.onnx",
              "InceptionV3": "https://github.com/onnx/models/raw/main/Computer_Vision/inception_v3_Opset17_timm/inception_v3_Opset17.onnx",
              "ResNet50": "https://github.com/onnx/models/raw/main/Computer_Vision/resnet50_Opset16_timm/resnet50_Opset16.onnx",
              "ResNet101": "https://github.com/onnx/models/raw/main/Computer_Vision/resnet101_Opset17_torch_hub/resnet101_Opset17.onnx", 
              "ResNet152": "https://github.com/onnx/models/raw/main/Computer_Vision/resnet152_Opset17_timm/resnet152_Opset17.onnx",
               "VGG16": "https://github.com/onnx/models/raw/main/Computer_Vision/vgg16_Opset16_timm/vgg16_Opset16.onnx", 
               "VGG19": "https://github.com/onnx/models/raw/main/Computer_Vision/vgg19_Opset16_timm/vgg19_Opset16.onnx"}

def quantize(save_dir):
    onnx_models_dir = pathlib.Path(save_dir)
    onnx_models_dir.mkdir(exist_ok=True, parents=True)

    for modelName in modelNames:
        original_model_path = os.path.join(save_dir, f"{modelName}.onnx")
        preprocessed_model_path = os.path.join(save_dir, f"{modelName}_preprocessed.onnx")
        quantized_model_path = os.path.join(save_dir, f"{modelName}_quant.onnx")

        try:
            # Send a GET request to the URL
            response = requests.get(modelNames[modelName], stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Write the content to a file
            with open(original_model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully as {original_model_path}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

        quant_pre_process(original_model_path, preprocessed_model_path)

        # Perform dynamic range quantization
        quantize_dynamic(
            preprocessed_model_path,  # input model
            quantized_model_path,  # save quantized model
            weight_type=QuantType.QUInt8  # Quantize weights to 8 bits
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory to save the models', required=True)
    
    args = parser.parse_args()

    quantize(args.dir)
