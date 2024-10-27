import torch
from torchvision import models
import argparse
import os

model_names = ["mobilenet_v2", "inception_v3", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]

def convert(input_dir, output_dir):
    for model_name in model_names:
        model_path = os.path.join(input_dir, f"{model_name}.pth")
        quant_model_path = os.path.join(input_dir, f"{model_name}_quant.pth")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
        if not os.path.exists(quant_model_path):
            print(f"Quantized model file not found: {quant_model_path}")
            continue

        # Load and export the original model
        try:
            model_class = getattr(models, model_name)
            model = model_class()  # Model initialized without pretrained weights
            model.eval()

            # Set input shape based on model type
            torch_input = torch.randn((1, 3, 299, 299) if model_name.startswith('inception') else (1, 3, 224, 224))

            # Export the original model
            model.load_state_dict(torch.load(model_path))
            torch.onnx.export(model, torch_input, os.path.join(output_dir, f"pytorch_{model_name}.onnx"), opset_version=15)
            print(f"Exported ONNX model for {model_name}")

            # Load and export the quantized model
            model.load_state_dict(torch.load(quant_model_path))
            torch.onnx.export(model, torch_input, os.path.join(output_dir, f"pytorch_{model_name}_quant.onnx"), opset_version=15)
            print(f"Exported quantized ONNX model for {model_name}")

        except Exception as e:
            print(f"Error exporting {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_dir', help='The directory where the PyTorch models are saved', required=True)
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models should be saved', required=True)
    args = parser.parse_args()

    convert(args.pytorch_dir, args.onnx_dir)
