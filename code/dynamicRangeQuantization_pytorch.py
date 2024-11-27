import torch
import torch.quantization
import os
from torchvision import models
import pathlib
import argparse

modelNames = ["mobilenet_v2", "inception_v3", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]
    

def quantize(save_dir):
    torch_models_dir = pathlib.Path(save_dir)
    torch_models_dir.mkdir(exist_ok=True, parents=True)
    
    for modelName in modelNames:
        model_class = getattr(models, modelName)
        model = model_class(pretrained=True)
        model.eval()

        # Save the original model
        original_model_path = os.path.join(save_dir, f"{modelName}.pth")
        torch.save(model.state_dict(), original_model_path)
        
        # Apply dynamic range quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, dtype=torch.qint8
        )
        
        # Save the quantized model
        quantized_model_path = os.path.join(save_dir, f"{modelName}_quant.pth")
        torch.save(quantized_model.state_dict(), quantized_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory to save the models', required=True)
    
    args = parser.parse_args()

    quantize(args.dir)