import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import argparse
import os

pytorch_model_names =  {"MobileNetV2": "mobilenet_v2",
                        "InceptionV3": "inception_v3",
                        "ResNet50": "resnet50", 
                        "ResNet101": "resnet101", 
                        "ResNet152": "resnet152", 
                        "VGG16": "vgg16", 
                        "VGG19": "vgg19"}

def profile_models(pytorch_dir, results_dir, n):
    results_dir = results_dir or '.'
    n = int(n) if n else 10
    
    for pytorch_model in pytorch_model_names:
        print(f"Profiling {pytorch_model} models")
        for i in range(n):
            pytorch_model_path = os.path.join(pytorch_dir, f"{pytorch_model}.pth")
            pytorch_results_path = os.path.join(results_dir, f"pytorch_{pytorch_model}_profiling_{i}")
            run_profiler(pytorch_model_names[pytorch_model], pytorch_model_path, pytorch_results_path)
            
            pytorch_model_path_quant = os.path.join(pytorch_dir, f"{pytorch_model}_quant.pth")
            pytorch_results_path_quant = os.path.join(results_dir, f"pytorch_{pytorch_model}_quant_profiling_{i}")
            run_profiler(pytorch_model_names[pytorch_model], pytorch_model_path_quant, pytorch_results_path_quant)


def run_profiler(model_name, model_path, output_file_path):
    model_class = getattr(models, model_name)
    model = model_class()  # Model initialized without pretrained weights
    model.eval()

    # Set input shape based on model type
    torch_input = torch.randn((1, 3, 299, 299) if model_name.startswith('inception') else (1, 3, 224, 224))

    # Export the original model
    model.load_state_dict(torch.load(model_path))

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(torch_input)
        
    with open(f'{output_file_path}.txt', 'w') as f:
        f.write(prof.key_averages().table())
    
    prof.export_chrome_trace(f'{output_file_path}.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_dir', help='The directory where the PyTorch models are saved', required=True)
    parser.add_argument('--results_dir', help='The directory where the PyTorch profiling are to be saved')
    parser.add_argument('--num_repetitions', type=int, help='The number of repetitions for profiling')
    args = parser.parse_args()

    if not os.path.exists(args.pytorch_dir):
        raise FileNotFoundError(f"Error: The input path '{args.pytorch_dir}' does not exist.")

    profile_models(args.pytorch_dir, args.results_dir, args.num_repetitions)