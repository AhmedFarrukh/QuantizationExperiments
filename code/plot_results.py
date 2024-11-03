import onnxruntime as ort
import numpy as np
import argparse
import os
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

tflite_model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
onnx_model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]
pytorch_model_names = ["mobilenet_v2", "inception_v3", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]

def plot(results_dir, save_dir, framework, model_names):
    if not os.path.exists(results_dir):
        print(f"Error: The input path '{results_dir}' does not exist.")
        return
    
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    orig_model_loading_uri = []
    orig_session_initialized = []
    orig_model_run = []
    quant_model_loading_uri = []
    quant_session_initialized = []
    quant_model_run = []

    for model in model_names:
        result_path = os.path.join(results_dir, f"{model}_profiling.json")
        with open(result_path, 'r') as file:
            data = json.load(file)

        for entry in data:
            if entry.get('name') == 'model_loading_uri':
                orig_model_loading_uri.append(entry['dur'])
            if entry.get('name') == 'session_initialization':
                orig_session_initialized.append(entry['dur'])
            if entry.get('name') == 'model_run':
                orig_model_run.append(entry['dur'])

    for model in model_names:
        result_path = os.path.join(results_dir, f"{model}_quant_profiling.json")
        with open(result_path, 'r') as file:
            data = json.load(file)

        for entry in data:
            if entry.get('name') == 'model_loading_uri':
                quant_model_loading_uri.append(entry['dur'])
            if entry.get('name') == 'session_initialization':
                quant_session_initialized.append(entry['dur'])
            if entry.get('name') == 'model_run':
                quant_model_run.append(entry['dur'])
    
    n_groups = len(model_names)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_model_loading_uri, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_model_loading_uri, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'{framework}-Model Loading Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'{framework}-ModelLoadingTime.png')
    plt.savefig(f'{save_dir}/{framework}-ModelLoadingTime.png')
    plt.close()

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_session_initialized, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_session_initialized, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'{framework}-Session Intialization Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'{framework}-SessionIntializationTime.png')
    plt.savefig(f'{save_dir}/{framework}-SessionIntializationTime.png')
    plt.close()

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_model_run, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_model_run, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'{framework}-Model Run Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'{framework}-ModelRunTime.png')
    plt.savefig(f'{save_dir}/{framework}-ModelRunTime.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_dir', help='The directory where the TFLite models\' results are saved')
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models\' results are saved')
    parser.add_argument('--pytorch_dir', help='The directory where the PyTorch models\' results are saved')
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    args = parser.parse_args()

    if args.tflite_dir:
        plot(args.tflite_dir, args.save_dir, "tflite", ['tflite_' + name for name in tflite_model_names])
    
    if args.pytorch_dir:
        plot(args.pytorch_dir, args.save_dir, "pytorch", ['pytorch_' + name for name in tflite_model_names])

    if args.onnx_dir:
        plot(args.onnx_dir, args.save_dir, "onnx", ['onnx_' + name for name in tflite_model_names])