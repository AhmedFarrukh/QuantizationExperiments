import onnxruntime as ort
import numpy as np
import argparse
import os
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

model_names = ["MobileNet", "InceptionV3", "ResNet50", "ResNet101", "ResNet152", "VGG16", "VGG19"]

def plot(results_dir, save_dir, num_repetitions):
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

        orig_model_loading_uri.append(0)
        orig_session_initialized.append(0)
        orig_model_run.append(0)

        for i in range(num_repetitions):
            result_path = os.path.join(results_dir, f"onnx_{model}_profiling_{i}.json")
            with open(result_path, 'r') as file:
                data = json.load(file)

            for entry in data:
                if entry.get('name') == 'model_loading_uri':
                    orig_model_loading_uri[-1] += entry['dur']
                if entry.get('name') == 'session_initialization':
                    orig_session_initialized[-1] += entry['dur']
                if entry.get('name') == 'model_run':
                    orig_model_run[-1] += entry['dur']

        orig_model_loading_uri /= num_repetitions
        orig_session_initialized /= num_repetitions
        orig_model_run /= num_repetitions

    for model in model_names:

        quant_model_loading_uri.append(0)
        quant_session_initialized.append(0)
        quant_model_run.append(0)

        for i in range(num_repetitions):
            result_path = os.path.join(results_dir, f"onnx_{model}_quant_profiling_{i}.json")
            with open(result_path, 'r') as file:
                data = json.load(file)

            for entry in data:
                if entry.get('name') == 'model_loading_uri':
                    quant_model_loading_uri[-1] += entry['dur']
                if entry.get('name') == 'session_initialization':
                    quant_session_initialized[-1] += entry['dur']
                if entry.get('name') == 'model_run':
                    quant_model_run[-1] += entry['dur']

        orig_model_loading_uri /= num_repetitions
        orig_session_initialized /= num_repetitions
        orig_model_run /= num_repetitions
    
    n_groups = len(model_names)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_model_loading_uri, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_model_loading_uri, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'Model Loading Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'ModelLoadingTime.png')
    plt.savefig(f'{save_dir}/ModelLoadingTime.png')
    plt.close()

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_session_initialized, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_session_initialized, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'Session Intialization Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'SessionIntializationTime.png')
    plt.savefig(f'{save_dir}/SessionIntializationTime.png')
    plt.close()

    fig, ax = plt.subplots()
    rects1 = plt.bar(index, orig_model_run, bar_width, alpha=opacity, label='Original')
    rects2 = plt.bar(index + bar_width, quant_model_run, bar_width, alpha=opacity, label='Quantized')
    plt.xlabel('Model')
    plt.ylabel(f'Model Run Time')
    plt.xticks(index + bar_width / 2, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title(f'ModelRunTime.png')
    plt.savefig(f'{save_dir}/ModelRunTime.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_dir', help='The directory where the ONNX models\' results are saved', required=True)
    parser.add_argument('--save_dir', help='The directory where the plots should be saved', required = True)
    parser.add_argument('--num_repetitions', help='The number of profiling files for each model', required = True)
    args = parser.parse_args()


    plot(args.onnx_dir, args.save_dir, args.num_repetitions)